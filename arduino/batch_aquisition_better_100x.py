#!/usr/bin/env python3
"""
Averaged trace capture for ATmega AES @ 16 MHz

Per block:
- choose one random 16-byte plaintext
- capture N_AVG traces for this plaintext (no ciphertext fetching in between)
- average them (mean trace)
- then fetch a single ciphertext for that plaintext
- save one NPZ per block (mean trace + PT + CT + metadata)

Console output is minimized:
- tqdm progress bar for outer loop (blocks)
- a short heartbeat print every 100 blocks
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import trange
from contextlib import contextmanager, redirect_stdout

# --------------------------------------------------------------------
#  Local imports
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from arduino.encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition

# ---- small helper to silence verbose prints from underlying classes ----
@contextmanager
def muted_stdout(enabled: bool = True):
    if not enabled:
        # no-op context
        yield
        return
    import io
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            yield
    finally:
        # drop buffered text
        buf.close()


def capture_mean_for_plaintext(enc, scope, plaintext: bytes, n_avg: int) -> np.ndarray:
    """
    Capture n_avg traces for a fixed plaintext and return the mean trace (float32).
    Output is averaged; individual traces are not stored.
    """
    running_sum = None  # float64 for numeric stability

    for _ in range(n_avg):
        enc.ser.reset_input_buffer()

        # Send plaintext (quiet)
        with muted_stdout(True):
            enc.set_state(plaintext)

        # Arm scope, trigger encryption, capture (quiet)
        with muted_stdout(True):
            scope.prepare_block_mode()
        with muted_stdout(True):
            enc.encrypt()

        wf = None
        for attempt in (1, 2):
            with muted_stdout(True):
                wf = scope.capture_block()
            if wf is not None and len(wf):
                break
            time.sleep(0.02)
        if wf is None or len(wf) == 0:
            raise RuntimeError("Empty capture twice in a row")

        if running_sum is None:
            running_sum = np.zeros_like(wf, dtype=np.float64)
        elif running_sum.shape[0] != wf.shape[0]:
            raise ValueError(f"Waveform length changed: got {wf.shape[0]}, expected {running_sum.shape[0]}")

        running_sum += wf.astype(np.float64)

    mean_trace = (running_sum / float(n_avg)).astype(np.float32)
    return mean_trace


def batch_with_encryption_averaged(
    num_blocks      = 10_000,      # number of distinct plaintexts (and files)
    n_avg           = 100,         # traces to average per plaintext
    duration        = 0.0001,      # 0.1 ms window (focus on Round-1 / first byte)
    filename_prefix = "encrypt_mean_b0_short",
    output_dir      = "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_ACoff4mV",
    port            = "COM7",
    baud            = 115200
):
    # --- UART init (quiet connect message) ---
    with muted_stdout(True):
        enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # --- Scope init for high-resolution short window:
    #     - 15-bit resolution
    #     - timebase = 3  (~125 MS/s; driver reports actual dt via GetTimebase2)
    #     - EXT trigger (e.g., UART-RX of 16th byte)
    #     - DC coupling (analogue offset requires DC path)
    #     - ±10 mV input range
    #     - +4 mV analogue offset (avoid lower-rail clipping)
    #     - 20 MHz bandwidth limiter enabled
    scope = DataAcquisition(
        device_resolution_bits = 15,
        timebase               = 3,              # request ~125 MS/s
        sampling_rate_hz       = 125e6,          # nominal for metadata
        capture_duration_s     = duration,       # 0.1 ms window
        capture_channel        = "A",
        trigger_channel        = "EXT",
        trigger_threshold_mV   = 300,            # e.g., 0.3 V on EXT
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = "AC",           # AC coupling for noise reduction
        voltage_range          = "10MV",         # ±10 mV
        output_dir             = output_dir,
        filename_prefix        = filename_prefix,
        bandwidth_limit_20mhz  = True,           # enable 20 MHz BW limiter
        analogue_offset_mv     = 4.0             # +4 mV DC offset
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "trace_overview.csv"

    try:
        # Setup scope (quiet)
        with muted_stdout(True):
            scope.setup_picoscope()

        t_start = time.time()
        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["BlockIndex", "PlaintextHex", "CiphertextHex",
                "FileName", "Samples", "n_avg", "dt_ns"]
            )

            for blk in trange(num_blocks, desc=f"Averaging {n_avg} traces per PT (0.1 ms @ ~125 MS/s)"):
                # 1) choose plaintext
                pt = secrets.token_bytes(16)
                pt_hex = pt.hex().upper()

                # 2) average n_avg captures for this PT
                try:
                    mean_trace = capture_mean_for_plaintext(enc, scope, pt, n_avg=n_avg)
                except Exception as e:
                    # minimal warning, continue with next block
                    print(f"[Warn] block {blk}: capture failed ({e}), skipping.")
                    enc.ser.reset_input_buffer()
                    continue

                # 3) fetch a single ciphertext for this PT (quiet)
                try:
                    enc.ser.reset_input_buffer()
                    with muted_stdout(True):
                        enc.set_state(pt)
                        enc.encrypt()
                        ct = enc.get_state()
                except TimeoutError:
                    print(f"[Warn] block {blk}: ciphertext timeout; skipping block.")
                    enc.ser.reset_input_buffer()
                    continue

                ct_hex = ct.hex().upper()

                # 4) save NPZ (mean trace only)
                fname = f"{filename_prefix}_{blk:06d}.npz"
                fpath = Path(output_dir) / fname
                np.savez(
                    fpath,
                    trace_mean = mean_trace,
                    plaintext  = np.frombuffer(pt, np.uint8),
                    ciphertext = np.frombuffer(ct, np.uint8),
                    sr_nominal = scope.sampling_rate_hz,              # 125e6 nominal
                    dt_ns      = getattr(scope, "time_interval_ns", None),  # actual from driver (ns)
                    timebase   = scope.timebase_final,
                    n_avg      = n_avg
                )

                # 5) CSV log
                writer.writerow([blk, pt_hex, ct_hex, fname, len(mean_trace),
                                n_avg, getattr(scope, "time_interval_ns", None)])
                if blk % 200 == 0:
                    csv_f.flush()

                # periodic heartbeat every 100 blocks
                if (blk + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (blk + 1) / elapsed if elapsed > 0 else 0.0
                    eta_s = (num_blocks - (blk + 1)) / rate if rate > 0 else float('inf')
                    print(f"[{blk+1}/{num_blocks}] mean_len={len(mean_trace)}  "
                        f"rate={rate:.2f} blk/s  ETA={eta_s/3600:.2f} h")

        print(f"\nDone. Averaged traces written to: {csv_path}")

    finally:
        # best-effort cleanup (quiet)
        try:
            with muted_stdout(True):
                scope.close()
        except Exception:
            pass
        try:
            with muted_stdout(True):
                enc.close()
        except Exception:
            pass


if __name__ == "__main__":
    # tip: test smaller first, e.g., num_blocks=50, n_avg=20
    batch_with_encryption_averaged(
        num_blocks=10_000,
        n_avg=100,
        duration=0.0001,  # 0.1 ms short window
        filename_prefix="encrypt_mean_b0_short",
        output_dir="data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV",
        port="COM7",
        baud=115200
    )
