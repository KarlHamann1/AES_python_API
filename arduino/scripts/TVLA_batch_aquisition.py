#!/usr/bin/env python3
"""
TVLA (Welch t-test) capture — interleaved RANDOM/FIXED — ATmega AES @ 16 MHz.

Writes one .npz per trace + a CSV index. Each NPZ has:
- trace (float32), plaintext (uint8[16]), group ("RANDOM"/"FIXED"),
- sr_nominal, dt_ns, timebase

How to evaluate:
1) Load RANDOM traces → X_R (n_R x T), FIXED → X_F (n_F x T)
2) For each sample t: t(t) = (mean_F - mean_R) / sqrt(var_F/n_F + var_R/n_R)
3) Plot t(t); sustained |t| > 4.5 → leakage

Tip: interleaving (done here) reduces drift. We avoid fetching CTs to keep UART quiet.
Default profile aims at a short Round-1 window (0.1 ms) with ~125 MS/s (timebase=3).
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager, redirect_stdout

REPO_ROOT   = Path(__file__).resolve().parents[2] 
ARDUINO_DIR = REPO_ROOT / "arduino"
SCOPE_DIR   = REPO_ROOT / "scope"
sys.path.append(str(ARDUINO_DIR / "scripts"))
sys.path.append(str(SCOPE_DIR))

from encipher import EncipherAPI                    # UART helper
from picoscope_acquisition import DataAcquisition   # PicoScope helper


# ── Small util: silence noisy prints when needed ───────────────────
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


def capture_single_trace(enc: EncipherAPI,
                        scope: DataAcquisition,
                        plaintext: bytes,
                        retries: int = 1) -> np.ndarray:
    """
    Arm scope → trigger one AES → capture one block (no averaging).
    Returns waveform (float32). Raises on repeated empty captures.
    """
    enc.ser.reset_input_buffer()
    with muted_stdout():
        enc.set_state(plaintext)   # send PT only; don't fetch CT here
        scope.prepare_block_mode()
        enc.encrypt()

    wf = None
    for _ in range(retries + 1):
        with muted_stdout():
            wf = scope.capture_block()
        if wf is not None and len(wf):
            break
        time.sleep(0.02)

    if wf is None or len(wf) == 0:
        raise RuntimeError("empty capture")

    return wf.astype(np.float32)


# ── Main TVLA capture (INTERLEAVED) ────────────────────────────────
def tvla_capture_interleaved(
    n_random: int     = 10_000,    # RANDOM traces
    n_fixed: int      = 10_000,    # FIXED traces
    fixed_pt_hex: str = "00000000000000000000000000000000",

    # 16 MHz “Round-1” profile (short window, high SR)
    duration: float   = 0.0001,    # 0.1 ms window
    device_bits: int  = 15,        # 12 is fine; 15 if you want max fidelity
    timebase: int     = 3,         # ~125 MS/s on many 5000A/D (driver confirms dt)
    sampling_hz: float= 125e6,     # nominal for metadata

    # paths / I/O
    out_dir_name: str = ("data_tvla_INTERLEAVED_16MHz_tb3_125Msps_115200Bd_"
                        "10kR_10kF_15bit_AC_20MHzBW"),

    # comms / scope
    port: str              = "COM7",
    baud: int              = 115200,
    trig_threshold_mV: int = 300,    # EXT threshold
    voltage_range: str     = "10MV", # ±10 mV on Channel A
    bandwidth_limit_20mhz: bool = True,
    analogue_offset_mv: float = 0.0,
    coupling_mode: str     = "AC"
):
    fixed_pt = bytes.fromhex(fixed_pt_hex)

    # Output folder under arduino/data
    output_dir = ARDUINO_DIR / "data" / out_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trace_overview.csv"

    # ---- UART ----
    with muted_stdout():
        enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # ---- Scope ----
    scope = DataAcquisition(
        device_resolution_bits = device_bits,
        timebase               = timebase,
        sampling_rate_hz       = sampling_hz,
        capture_duration_s     = duration,
        capture_channel        = "A",
        trigger_channel        = "EXT",
        trigger_threshold_mV   = trig_threshold_mV,
        trigger_delay_samples  = 0,
        auto_trigger_ms        = 1000,
        coupling_mode          = coupling_mode,
        voltage_range          = voltage_range,
        output_dir             = str(output_dir),
        filename_prefix        = "tvla",
        bandwidth_limit_20mhz  = bandwidth_limit_20mhz,
        analogue_offset_mv     = analogue_offset_mv
    )

    try:
        with muted_stdout():
            scope.setup_picoscope()

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["OrderIdx", "Group", "PerGroupIdx", "PlaintextHex",
                "FileName", "Samples", "dt_ns", "timebase"]
            )

            # Interleave: R, F, R, F, ... and stop exactly at n_random / n_fixed
            r_count = f_count = order_idx = 0
            pbar = tqdm(total=(n_random + n_fixed),
                        desc=f"TVLA interleaved  ({duration*1e3:.1f} ms window)")

            next_group = "R"  # start with RANDOM
            while r_count < n_random or f_count < n_fixed:
                # pick group; if one is done, use the other
                if next_group == "R":
                    grp = "F" if r_count >= n_random and f_count < n_fixed else "R"
                else:
                    grp = "R" if f_count >= n_fixed and r_count < n_random else "F"

                if grp == "R":
                    pt = secrets.token_bytes(16)
                    per_group_idx = r_count
                else:
                    pt = fixed_pt
                    per_group_idx = f_count

                # capture one trace
                try:
                    wf = capture_single_trace(enc, scope, pt, retries=1)
                except Exception as e:
                    print(f"[Warn] {grp}#{per_group_idx}: capture failed ({e}) — retrying next loop")
                    enc.ser.reset_input_buffer()
                    next_group = "F" if grp == "R" else "R"
                    continue

                # save NPZ
                fname = f"tvla_I_{grp}_{per_group_idx:06d}.npz"
                np.savez(
                    output_dir / fname,
                    trace      = wf,
                    plaintext  = np.frombuffer(pt, np.uint8),
                    group      = np.array(["RANDOM" if grp == "R" else "FIXED"]),
                    sr_nominal = scope.sampling_rate_hz,
                    dt_ns      = getattr(scope, "time_interval_ns", None),
                    timebase   = getattr(scope, "timebase_final", None)
                )

                # CSV log
                writer.writerow([
                    order_idx,
                    "RANDOM" if grp == "R" else "FIXED",
                    per_group_idx,
                    pt.hex().upper(),
                    fname,
                    len(wf),
                    getattr(scope, "time_interval_ns", None),
                    getattr(scope, "timebase_final", None)
                ])
                if order_idx and (order_idx % 1000 == 0):
                    csv_f.flush()

                # advance counters / flip group
                if grp == "R":
                    r_count += 1
                    next_group = "F"
                else:
                    f_count += 1
                    next_group = "R"

                order_idx += 1
                pbar.update(1)

            pbar.close()
        print(f"\nDone. TVLA (interleaved) CSV written to: {csv_path}")

    finally:
        with muted_stdout():
            try: scope.close()
            except Exception: pass
            try: enc.close()
            except Exception: pass


if __name__ == "__main__":
    tvla_capture_interleaved()
