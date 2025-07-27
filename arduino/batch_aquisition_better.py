#!/usr/bin/env python3
"""
High‑quality trace capture for the ATmega AES demo – compatible with the
unmodified DataAcquisition class you supplied.
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
#  Helper: robust serial read_exact
# --------------------------------------------------------------------
def read_exact(ser, n_bytes: int, err_msg: str) -> bytes:
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = ser.read(n_bytes - len(buf))
        if not chunk:
            raise TimeoutError(err_msg)
        buf.extend(chunk)
    return bytes(buf)

# --------------------------------------------------------------------
#  Main batch function
# --------------------------------------------------------------------
def batch_with_encryption(
    num_rounds      = 10_000,
    duration        = 0.010,         # 10 ms
    filename_prefix = "encrypt",
    output_dir      = "data_arduino_block",
    port            = "COM5",
    baud            = 9600
):
    # ---------------- UART and scope initialisation ---------------
    enc = EncipherAPI(port=port, baudrate=baud, timeout=1)

    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 43,           # ~1.56 MS/s
        sampling_rate_hz       = 1.56e6,
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "B",
        trigger_threshold_mV   = 100,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = "AC",
        voltage_range          = "10MV",
        output_dir             = output_dir,
        filename_prefix        = filename_prefix
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "trace_overview.csv"

    try:
        scope.setup_picoscope()

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["TraceIndex", "PlaintextHex", "CiphertextHex",
                "FileName",  "Samples"]
            )

            for idx in trange(num_rounds, desc="Capturing"):
                # 1  generate 16‑byte random plaintext
                pt  = secrets.token_bytes(16)
                pt_hex = pt.hex().upper()

                # 2  send plaintext
                enc.set_state(pt)

                # 3  arm scope
                scope.prepare_block_mode()

                # 4  trigger encryption
                enc.encrypt()

                # 5  capture waveform (retry once on empty data)
                for attempt in (1, 2):
                    wf = scope.capture_block()
                    if wf is not None and len(wf):
                        break
                    print(f"[Warn] empty capture (attempt {attempt})")
                    time.sleep(0.05)
                else:
                    raise RuntimeError("consecutive empty captures")

                # 6  get ciphertext
                ct = read_exact(enc.ser, 16, "ciphertext timeout")
                ct_hex = ct.hex().upper()

                # 7  save NPZ (trace + meta)
                fname = f"{filename_prefix}_{idx:06d}.npz"
                fpath = Path(output_dir) / fname
                np.savez(
                    fpath,
                    trace      = wf.astype(np.float32),
                    plaintext  = np.frombuffer(pt, np.uint8),
                    ciphertext = np.frombuffer(ct, np.uint8),
                    sr         = scope.sampling_rate_hz,
                    timebase   = scope.timebase_final 
                )

                # 8  log to CSV
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
