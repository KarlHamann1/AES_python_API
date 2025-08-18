#!/usr/bin/env python3
"""
Baseline (noise-only) trace capture.

No AES is triggered — the scope relies on auto-trigger after a timeout.
Handy for building a reference noise set for filtering/SNR studies.
"""

import os, sys, csv, time, secrets   # secrets only to mimic plaintext bytes
from pathlib import Path
import numpy as np
from tqdm import trange

# ── Repo-aware imports ─────────────────────────────────────────────
# This file lives at: AES_PYTHON_API/arduino/scripts/...
REPO_ROOT   = Path(__file__).resolve().parents[2]   # AES_PYTHON_API/
ARDUINO_DIR = REPO_ROOT / "arduino"
SCOPE_DIR   = REPO_ROOT / "scope"

# Make module folders importable (no packages required)
sys.path.append(str(ARDUINO_DIR / "scripts"))
sys.path.append(str(SCOPE_DIR))

from encipher import EncipherAPI                     # Arduino UART helper
from picoscope_acquisition import DataAcquisition    # PicoScope helper


def batch_no_encryption(
    num_rounds: int      = 10_000,
    duration: float      = 0.010,            # 10 ms window
    filename_prefix: str = "noise",
    out_dir_name: str    = "data_arduino_noise",
    port: str            = "COM5",
    baud: int            = 9600
) -> None:
    """Capture *num_rounds* auto-triggered traces with no AES activity."""
    # Output folder under arduino/data
    output_dir = ARDUINO_DIR / "data" / out_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trace_overview.csv"

    # Keep UART alive so the MCU stays powered/clocked
    enc = EncipherAPI(port=port, baudrate=baud, timeout=1)

    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 43,          # ~1.56 MS/s (match your AES runs if desired)
        sampling_rate_hz       = 1.56e6,
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "EXT",       # no edge expected; auto-trigger will fire
        trigger_threshold_mV   = 100,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,        # fire after 1 s if no trigger
        coupling_mode          = "AC",
        voltage_range          = "10MV",
        output_dir             = str(output_dir),
        filename_prefix        = filename_prefix
    )

    try:
        scope.setup_picoscope()

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(["TraceIndex", "DummyPlaintextHex", "FileName", "Samples"])

            for idx in trange(num_rounds, desc="Capturing (noise)"):
                # 0) Dummy PT for consistency with encrypted runs (never used)
                dummy_pt = secrets.token_bytes(16)
                pt_hex = dummy_pt.hex().upper()

                # 1) Send state to keep the protocol identical
                enc.set_state(dummy_pt)

                # 2) Arm the scope — it will auto-trigger after timeout
                scope.prepare_block_mode()

                # 3) Do NOT call enc.encrypt(); just capture the block
                wf = scope.capture_block()
                if wf is None or len(wf) == 0:
                    print("[Warn] auto-trigger timeout — empty capture")
                    continue

                # 4) Save NPZ
                fname = f"{filename_prefix}_{idx:06d}.npz"
                np.savez(
                    output_dir / fname,
                    trace      = wf.astype(np.float32),
                    plaintext  = np.frombuffer(dummy_pt, np.uint8),
                    noise_only = True,
                    sr         = scope.sampling_rate_hz,
                    timebase   = getattr(scope, "timebase_final", None)
                )

                # 5) Log CSV
                writer.writerow([idx, pt_hex, fname, len(wf)])
                if idx % 500 == 0:
                    csv_f.flush()

        print(f"\nFinished. Overview CSV: {csv_path}")

    finally:
        # Best-effort cleanup
        try: scope.close()
        except Exception: pass
        try: enc.close()
        except Exception: pass


if __name__ == "__main__":
    batch_no_encryption()
