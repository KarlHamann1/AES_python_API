
"""
Averaged-trace capture for an AVR-based AES target (Arduino).

Per plaintext block:
1) pick a random 16-byte PT
2) capture n_avg traces, average to a mean trace
3) fetch one CT for that PT
4) save mean trace + PT + CT + metadata as .npz

Defaults:
- PicoScope: 12-bit, timebase 5 (~31.25 MS/s), AC, ±20 mV, 20 MHz BW
- UART: 57_600 Bd
- Compact console output (tqdm + periodic heartbeat)
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import trange
from contextlib import contextmanager, redirect_stdout


REPO_ROOT = Path(__file__).resolve().parents[2] 
ARDUINO_DIR = REPO_ROOT / "arduino"
SCOPE_DIR = REPO_ROOT / "scope"


sys.path.append(str(ARDUINO_DIR / "scripts"))
sys.path.append(str(SCOPE_DIR))

from encipher import EncipherAPI                    # UART helper (Arduino)
from picoscope_acquisition import DataAcquisition   # PicoScope helper


#  utils
@contextmanager
def muted_stdout(enable: bool = True):
    """Silence noisy driver prints when *enable* is True."""
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


def capture_mean_for_plaintext(
    enc: EncipherAPI,
    scope: DataAcquisition,
    plaintext: bytes,
    n_avg: int
) -> np.ndarray:
    """Capture *n_avg* traces for a fixed PT and return their mean (float32)."""
    acc: np.ndarray | None = None  # running sum (float64 for precision)

    for _ in range(n_avg):
        enc.ser.reset_input_buffer()

        with muted_stdout(): enc.set_state(plaintext)
        with muted_stdout(): scope.prepare_block_mode()
        with muted_stdout(): enc.encrypt()

        # grab one block (retry once if empty)
        wf = None
        for _try in (0, 1):
            with muted_stdout(): wf = scope.capture_block()
            if wf is not None and len(wf):
                break
            time.sleep(0.02)

        if wf is None or len(wf) == 0:
            raise RuntimeError("empty capture")

        if acc is None:
            acc = np.zeros_like(wf, dtype=np.float64)
        elif acc.shape[0] != wf.shape[0]:
            raise ValueError(f"waveform length changed: {wf.shape[0]} vs {acc.shape[0]}")

        acc += wf.astype(np.float64)

    return (acc / float(n_avg)).astype(np.float32)


# ── Main batch routine: Adjust This!
def batch_with_encryption_averaged(
    num_blocks: int      = 10_000,                    # plaintexts ↔ files
    n_avg: int           = 100,                       # traces averaged / PT
    duration: float      = 0.00015,                    # 0.15 ms capture window
    filename_prefix: str = "encrypt_mean",
    out_dir_name: str    = ("data_arduino_8MHz_tb5_31.25Msps_57600Bd_"
                            "avg100_1.5ms_20MHzBW_12bit_ACoff2mV"),
    port: str            = "COM7",
    baud: int            = 57_600
):
    """
    Run the acquisition. Output goes to AES_PYTHON_API/arduino/data/<out_dir_name>.
    """
    # Output folder under arduino/data
    output_dir = ARDUINO_DIR / "data" / out_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trace_overview.csv"

    #UART (Arduino) 
    with muted_stdout():
        enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # Scope
    scope = DataAcquisition(
        device_resolution_bits = 12,
        timebase               = 5,          # Pico timebase 5 ≈ 31.25 MS/s
        sampling_rate_hz       = 31.25e6,    # nominal (metadata)
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "EXT",      # e.g., Arduino trigger → EXT
        trigger_threshold_mV   = 300,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = "AC",
        voltage_range          = "20MV",     # ±20 mV
        output_dir             = str(output_dir),
        filename_prefix        = filename_prefix,
        bandwidth_limit_20mhz  = True,
        analogue_offset_mv     = 10.0
    )

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

                # (1) mean trace for this PT
                try:
                    mean_trace = capture_mean_for_plaintext(enc, scope, pt, n_avg)
                except Exception as e:
                    print(f"[Warn] block {blk}: capture failed ({e}) – skip")
                    enc.ser.reset_input_buffer()
                    continue

                # (2) one ciphertext for this PT
                try:
                    enc.ser.reset_input_buffer()
                    with muted_stdout():
                        enc.set_state(pt)
                        enc.encrypt()
                        ct = enc.get_state()
                except TimeoutError:
                    print(f"[Warn] block {blk}: CT timeout – skip")
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
                    len(mean_trace), n_avg, getattr(scope, "time_interval_ns", None)]
                )
                if blk % 200 == 0:
                    csv_f.flush()

                # heartbeat
                if (blk + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (blk + 1) / elapsed if elapsed else 0.0
                    eta  = (num_blocks - blk - 1) / rate if rate else float("inf")
                    print(f"[{blk+1}/{num_blocks}]  rate={rate:.2f} blk/s  ETA={eta/3600:.2f} h")

        print(f"\nDone. Overview CSV: {csv_path}")

    finally:
        with muted_stdout():
            try: scope.close()
            except Exception: pass
            try: enc.close()
            except Exception: pass


if __name__ == "__main__":
    batch_with_encryption_averaged()
