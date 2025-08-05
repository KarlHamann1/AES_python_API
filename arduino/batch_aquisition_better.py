#!/usr/bin/env python3
"""
High-quality trace capture for the ATmega AES @ 16 MHz

Changes for high-rate, short-window capture:
- Capture window: 0.7 ms (covers ~0.6 ms AES window + margin)
- Target sampling rate: ~31.25 MS/s (timebase = 5, per Pico 5000A formula)
- 12-bit resolution requested
- UART at 115200 baud
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import trange

# --------------------------------------------------------------------
#  Local imports
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from arduino.encipher import EncipherAPI          # unchanged
from picoscope_acquisition import DataAcquisition # your class

# --------------------------------------------------------------------
#  Main batch function
# --------------------------------------------------------------------
def batch_with_encryption(
    num_rounds      = 10_000,
    duration        = 0.0007,      # 0.7 ms window at 16 MHz (AES ~0.60 ms)
    filename_prefix = "encrypt",
    output_dir      = "data_arduino_16MHz_20dB_31Msps_115200Bd",
    port            = "COM7",
    baud            = 115200
):
    # ---------------- UART and scope initialisation ---------------
    # Larger timeout is safer when running many rounds
    enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # NOTE:
    # - device_resolution_bits: request 12-bit (PS5000A maps allowed timebases by resolution)
    # - timebase: 5 (per your formula; driver will return the actual sample interval)
    # - sampling_rate_hz: target/nominal value for bookkeeping (driver decides final)
    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 5,            # requested timebase (aim ~31.25 MS/s)
        sampling_rate_hz       = 31.25e6,      # nominal; store for metadata, actual from driver
        capture_duration_s     = duration,     # 0.7 ms window
        capture_channel        = "A",          # PD on A
        trigger_channel        = "EXT",        # hardware trigger from MCU
        trigger_threshold_mV   = 300,          # adjust if needed for clean arming
        trigger_delay_samples  = 0,            # capture from trigger edge
        auto_trigger_ms        = 1000,         # safety auto-trigger
        coupling_mode          = "AC",         # PD is usually AC-coupled
        voltage_range          = "10MV",       # use your smallest stable range
        output_dir             = output_dir,
        filename_prefix        = filename_prefix
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "trace_overview.csv"

    try:
        scope.setup_picoscope()  # program device; your class should resolve final timebase/sr

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["TraceIndex", "PlaintextHex", "CiphertextHex",
                "FileName", "Samples"]
            )

            for idx in trange(num_rounds, desc="Capturing"):
                # --- keep serial input clean before each round ---
                enc.ser.reset_input_buffer()

                # 1) Random plaintext (16 bytes)
                pt = secrets.token_bytes(16)
                pt_hex = pt.hex().upper()

                # 2) Send plaintext to target
                enc.set_state(pt)

                # 3) Arm scope for a new block
                scope.prepare_block_mode()

                # 4) Trigger encryption window (MCU pulls trigger HIGH)
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

                # 6) Fetch ciphertext (host sends CMD_GET_STATE inside)
                try:
                    ct = enc.get_state()
                except TimeoutError:
                    # If device/host slips, clear buffer and continue
                    enc.ser.reset_input_buffer()
                    continue

                ct_hex = ct.hex().upper()

                # 7) Save trace + metadata
                #    NOTE: If your DataAcquisition returns float32 already, this cast is no-op.
                #    If you later want smaller files, consider int16 storage + scale factor.
                fname = f"{filename_prefix}_{idx:06d}.npz"
                fpath = Path(output_dir) / fname
                np.savez(
                    fpath,
                    trace      = wf.astype(np.float32),
                    plaintext  = np.frombuffer(pt, np.uint8),
                    ciphertext = np.frombuffer(ct, np.uint8),
                    sr         = scope.sampling_rate_hz,   # nominal target
                    timebase   = scope.timebase_final      # actual timebase as reported by the driver
                )

                # 8) CSV log
                writer.writerow([idx, pt_hex, ct_hex, fname, len(wf)])
                if idx % 500 == 0:
                    csv_f.flush()

        print(f"\nDone. Overview CSV: {csv_path}")

    finally:
        scope.close()
        enc.close()

# --------------------------------------------------------------------
if __name__ == "__main__":
    batch_with_encryption()
