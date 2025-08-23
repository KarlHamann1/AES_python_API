#!/usr/bin/env python3
# capture_pi_averaged.py
"""
Averaged-trace capture for Raspberry Pi bare-metal AES (binary UART protocol).

Per block:
1) pick a random 16-byte PT
2) capture n_avg traces for that PT (store only the mean)
3) fetch one ciphertext for that PT
4) save NPZ (mean trace + PT + CT + metadata)

Scope defaults:
- 12-bit, timebase 5 (~31.25 MS/s), AC, ±20 mV, 20 MHz BW, small +offset
- Trigger on EXT (wire GPIO16 trigger to EXT or use a BNC probe)
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import trange
from contextlib import contextmanager, redirect_stdout


REPO_ROOT = Path(__file__).resolve().parents[2]     
PI_DIR    = REPO_ROOT / "pi"
SCOPE_DIR = REPO_ROOT / "scope"
sys.path.append(str(PI_DIR / "scripts"))
sys.path.append(str(SCOPE_DIR))

from pi_bare_metal_api import PiBareMetalAPI
from picoscope_acquisition import DataAcquisition

# utility to silence verbose prints from helpers 
@contextmanager
def muted_stdout(enable: bool = True):
    if not enable:
        yield; return
    import io
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            yield
    finally:
        buf.close()

def capture_mean_for_plaintext(enc: PiBareMetalAPI, scope: DataAcquisition,
                            plaintext: bytes, n_avg: int) -> np.ndarray:
    """Capture n_avg traces for a fixed PT and return their mean (float32)."""
    running_sum = None
    for _ in range(n_avg):
        enc.ser.reset_input_buffer()

        # Send PT, arm scope, trigger encryption, capture one block
        with muted_stdout(): enc.set_state(plaintext)
        with muted_stdout(): scope.prepare_block_mode()
        with muted_stdout(): enc.encrypt()

        wf = None
        for _attempt in (0, 1):
            with muted_stdout(): wf = scope.capture_block()
            if wf is not None and len(wf): break
            time.sleep(0.02)
        if wf is None or len(wf) == 0:
            raise RuntimeError("Empty capture (twice). Check trigger wiring/threshold.")

        if running_sum is None:
            running_sum = np.zeros_like(wf, dtype=np.float64)
        elif running_sum.shape[0] != wf.shape[0]:
            raise ValueError(f"Waveform length changed: {wf.shape[0]} vs {running_sum.shape[0]}")
        running_sum += wf.astype(np.float64)

    return (running_sum / float(n_avg)).astype(np.float32)

def batch_with_encryption_averaged(
    num_blocks: int      = 10_000,
    n_avg: int           = 100,
    duration: float      = 0.0015,   # 1.5 ms
    filename_prefix: str = "encrypt_mean",
    out_dir_name: str    = ("data_pi_tb5_31.25Msps_115200Bd_"
                            "avg100_1.5ms_20mV_20MHzBW_12bit_ACoff2mV"),
    port: str            = "COM7",
    baud: int            = 115200
):
    # Output folder under pi/data
    output_dir = PI_DIR / "data" / out_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trace_overview.csv"

    # UART
    with muted_stdout(): enc = PiBareMetalAPI(port=port, baudrate=baud, timeout=2)

    # Scope
    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 5,          # ~31.25 MS/s (driver returns actual dt)
        sampling_rate_hz       = 31.25e6,    # nominal for metadata
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "EXT",      # connect GPIO16 trigger
        trigger_threshold_mV   = 300,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = "AC",
        voltage_range          = "20MV",     # ±20 mV
        output_dir             = str(output_dir),
        filename_prefix        = filename_prefix,
        bandwidth_limit_20mhz  = True,
        analogue_offset_mv     = 2.0
    )

    try:
        with muted_stdout(): scope.setup_picoscope()

        t0 = time.time()
        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["BlockIndex", "PlaintextHex", "CiphertextHex",
                "FileName", "Samples", "n_avg", "dt_ns"]
            )

            desc = f"avg {n_avg} traces  ({duration*1e3:.1f} ms window)"
            for blk in trange(num_blocks, desc=desc):
                pt = secrets.token_bytes(16)

                # (1) average n_avg traces for this PT
                try:
                    mean_trace = capture_mean_for_plaintext(enc, scope, pt, n_avg)
                except Exception as e:
                    print(f"[Warn] block {blk}: capture failed ({e}) – skip")
                    enc.ser.reset_input_buffer()
                    continue

                # (2) fetch one CT for this PT
                try:
                    enc.ser.reset_input_buffer()
                    with muted_stdout():
                        enc.set_state(pt)
                        enc.encrypt()
                        ct = enc.get_state()
                except TimeoutError:
                    print(f"[Warn] block {blk}: ciphertext timeout – skip")
                    enc.ser.reset_input_buffer()
                    continue

                # (3) save NPZ
                fname = f"{filename_prefix}_{blk:06d}.npz"
                fpath = output_dir / fname
                np.savez(
                    fpath,
                    trace_mean = mean_trace,
                    plaintext  = np.frombuffer(pt, np.uint8),
                    ciphertext = np.frombuffer(ct, np.uint8),
                    sr_nominal = scope.sampling_rate_hz,
                    dt_ns      = getattr(scope, "time_interval_ns", None),
                    timebase   = getattr(scope, "timebase_final", None),
                    n_avg      = n_avg
                )

                # (4) CSV log
                writer.writerow(
                    [blk, pt.hex().upper(), ct.hex().upper(), fname,
                    len(mean_trace), n_avg,
                    getattr(scope, "time_interval_ns", None)]
                )
                if blk % 200 == 0:
                    csv_f.flush()

                if (blk + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (blk + 1) / elapsed if elapsed else 0.0
                    eta  = (num_blocks - blk - 1) / rate if rate else float('inf')
                    print(f"[{blk+1}/{num_blocks}]  rate={rate:.2f} blk/s  "
                        f"ETA={eta/3600:.2f} h")

        print(f"\nDone. Overview CSV written to: {csv_path}")

    finally:
        with muted_stdout():
            try: scope.close()
            except Exception: pass
            try: enc.close()
            except Exception: pass

if __name__ == "__main__":
    batch_with_encryption_averaged()
