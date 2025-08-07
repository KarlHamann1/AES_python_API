#!/usr/bin/env python3
"""
Generic averaged-trace capture script for an AVR-based AES target.

Workflow (per plaintext block)
--------------------------------
1. Pick one random 16-byte plaintext.
2. Capture *n_avg* power/optical traces for this plaintext.
3. Average them to a single mean trace.
4. Fetch one ciphertext for this plaintext.
5. Store everything in one *.npz* file (mean trace + PT + CT + metadata).

Console output is kept compact:
* tqdm progress bar for blocks
* heartbeat every 100 blocks

Adjusted for:
* 12-bit resolution
* 31.25 MS/s (PicoScope timebase 5)
* 57 600 Bd UART
* ±20 mV input range with 20 MHz BW-limit
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import trange
from contextlib import contextmanager, redirect_stdout

# ------------------------------------------------------------------
#  Local imports (project-specific)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from arduino.encipher import EncipherAPI           # UART helper
from picoscope_acquisition import DataAcquisition  # PicoScope helper

# ──────────────────────────────────────────────────────────────────
#  Utils
# ──────────────────────────────────────────────────────────────────
@contextmanager
def muted_stdout(enable: bool = True):
    """Suppress noisy prints from driver classes when *enable* is True."""
    if not enable:
        yield
        return
    import io
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            yield
    finally:
        buf.close()


def capture_mean_for_plaintext(enc, scope, plaintext: bytes, n_avg: int) -> np.ndarray:
    """
    Capture *n_avg* traces for a fixed plaintext and return their mean.
    The returned vector is float32; individual traces are not stored.
    """
    acc: np.ndarray | None = None  # running sum (float64 for stability)

    for _ in range(n_avg):
        enc.ser.reset_input_buffer()

        with muted_stdout():
            enc.set_state(plaintext)

        with muted_stdout():
            scope.prepare_block_mode()
        with muted_stdout():
            enc.encrypt()

        # grab one captured block (retry once on failure)
        wf = None
        for _ in (0, 1):
            with muted_stdout():
                wf = scope.capture_block()
            if wf is not None and len(wf):
                break
            time.sleep(0.02)
        if wf is None or len(wf) == 0:
            raise RuntimeError("empty capture")

        if acc is None:
            acc = np.zeros_like(wf, dtype=np.float64)
        elif acc.shape[0] != wf.shape[0]:
            raise ValueError("waveform length changed")

        acc += wf.astype(np.float64)

    return (acc / float(n_avg)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
#  Main batch routine
# ──────────────────────────────────────────────────────────────────
def batch_with_encryption_averaged(
    num_blocks      = 10_000,                       # plaintexts ↔ files
    n_avg           = 100,                          # traces averaged / PT
    duration        = 0.0015,                       # 1.5 ms capture window
    filename_prefix = "encrypt_mean",
    output_dir      = ("data_arduino_8MHz_tb5_31.25Msps_57600Bd_"
                    "avg100_1.5ms_20MHzBW_12bit_ACoff2mV"),
    port            = "COM7",
    baud            = 57_600
):
    # -------- UART ----------
    with muted_stdout():
        enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # -------- Scope ----------
    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 5,                 # Pico timebase 5 ≈ 31.25 MS/s
        sampling_rate_hz       = 31.25e6,
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "EXT",             # e.g. UART-RX on EXT
        trigger_threshold_mV   = 300,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = "AC",
        voltage_range          = "20MV",            # ±20 mV
        output_dir             = output_dir,
        filename_prefix        = filename_prefix,
        bandwidth_limit_20mhz  = True,
        analogue_offset_mv     = 2.0
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "trace_overview.csv"

    try:
        with muted_stdout():
            scope.setup_picoscope()

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
                try:
                    mean_trace = capture_mean_for_plaintext(enc, scope, pt, n_avg)
                except Exception as e:
                    print(f"[Warn] block {blk}: capture failed ({e}) – skip")
                    enc.ser.reset_input_buffer()
                    continue

                # one ciphertext for this PT
                try:
                    enc.ser.reset_input_buffer()
                    with muted_stdout():
                        enc.set_state(pt)
                        enc.encrypt()
                        ct = enc.get_state()
                except TimeoutError:
                    print(f"[Warn] block {blk}: CT timeout – skip")
                    continue

                fname = f"{filename_prefix}_{blk:06d}.npz"
                fpath = Path(output_dir) / fname
                np.savez(
                    fpath,
                    trace_mean = mean_trace,
                    plaintext  = np.frombuffer(pt, np.uint8),
                    ciphertext = np.frombuffer(ct, np.uint8),
                    sr_nominal = scope.sampling_rate_hz,
                    dt_ns      = getattr(scope, "time_interval_ns", None),
                    timebase   = scope.timebase_final,
                    n_avg      = n_avg
                )

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
            try:    scope.close()
            except Exception: pass
            try:    enc.close()
            except Exception: pass


if __name__ == "__main__":
    batch_with_encryption_averaged()
