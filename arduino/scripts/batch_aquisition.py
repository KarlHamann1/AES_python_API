#!/usr/bin/env python3
"""
single-trace capture for the ATmega AES @ 16 MHz.

Window & rates:
- Capture window: 0.7 ms (covers ~0.6 ms AES + margin)
- Target sampling rate: ~31.25 MS/s (Pico timebase = 5)
- Resolution: 12-bit
- UART: 115200 Bd
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import trange

# ── Repo-aware imports ─────────────────────────────────────────────
# This file lives at: AES_PYTHON_API/arduino/scripts/...
REPO_ROOT   = Path(__file__).resolve().parents[2]   # AES_PYTHON_API/
ARDUINO_DIR = REPO_ROOT / "arduino"
SCOPE_DIR   = REPO_ROOT / "scope"

# Make module folders importable (no package needed)
sys.path.append(str(ARDUINO_DIR / "scripts"))
sys.path.append(str(SCOPE_DIR))

from encipher import EncipherAPI                     # Arduino UART helper
from picoscope_acquisition import DataAcquisition    # PicoScope helper


# ── Main batch function ───────────────────────────────────────────
def batch_with_encryption(
    num_rounds: int      = 10_000,
    duration: float      = 0.0007,         # 0.7 ms @ 31.25 MS/s ≈ 21,875 samples
    filename_prefix: str = "encrypt",
    out_dir_name: str    = "data_arduino_16MHz_20dB_31Msps_115200Bd",
    port: str            = "COM7",
    baud: int            = 115200
):
    """
    Capture one trace per random plaintext and store each as an NPZ.
    Output path: AES_PYTHON_API/arduino/data/<out_dir_name>
    """
    # Output folder under arduino/data
    output_dir = ARDUINO_DIR / "data" / out_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trace_overview.csv"

    # ----- UART (Arduino) -----
    # Slightly longer timeout helps during long runs
    enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # ----- Scope -----
    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 5,             # request ~31.25 MS/s
        sampling_rate_hz       = 31.25e6,       # nominal for metadata
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "EXT",         # hardware trigger from MCU
        trigger_threshold_mV   = 300,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = "AC",
        voltage_range          = "10MV",
        output_dir             = str(output_dir),
        filename_prefix        = filename_prefix
    )

    try:
        # Program the scope; the driver will report actual timebase/sample interval
        scope.setup_picoscope()

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(["TraceIndex", "PlaintextHex", "CiphertextHex", "FileName", "Samples"])

            for idx in trange(num_rounds, desc="Capturing"):
                # Keep the serial input clean for each round
                enc.ser.reset_input_buffer()

                # 1) Random 16-byte plaintext
                pt = secrets.token_bytes(16)
                pt_hex = pt.hex().upper()

                # 2) Send PT to target
                enc.set_state(pt)

                # 3) Arm scope
                scope.prepare_block_mode()

                # 4) Trigger encryption (device raises trigger)
                enc.encrypt()

                # 5) Capture waveform (retry once if empty)
                wf = None
                for attempt in (1, 2):
                    wf = scope.capture_block()
                    if wf is not None and len(wf):
                        break
                    print(f"[Warn] empty capture (attempt {attempt})")
                    time.sleep(0.05)
                if wf is None or len(wf) == 0:
                    raise RuntimeError("consecutive empty captures")

                # 6) Fetch ciphertext
                try:
                    ct = enc.get_state()
                except TimeoutError:
                    enc.ser.reset_input_buffer()
                    continue
                ct_hex = ct.hex().upper()

                # 7) Save trace + metadata
                fname = f"{filename_prefix}_{idx:06d}.npz"
                fpath = output_dir / fname
                np.savez(
                    fpath,
                    trace      = wf.astype(np.float32),
                    plaintext  = np.frombuffer(pt, np.uint8),
                    ciphertext = np.frombuffer(ct, np.uint8),
                    sr         = scope.sampling_rate_hz,     # nominal
                    timebase   = getattr(scope, "timebase_final", None)
                )

                # 8) CSV log
                writer.writerow([idx, pt_hex, ct_hex, fname, len(wf)])
                if idx % 500 == 0:
                    csv_f.flush()

        print(f"\nDone. Overview CSV: {csv_path}")

    finally:
        # Best-effort clean-up
        try: scope.close()
        except Exception: pass
        try: enc.close()
        except Exception: pass


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_with_encryption()
