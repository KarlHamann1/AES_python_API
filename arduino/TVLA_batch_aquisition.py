#!/usr/bin/env python3
"""
TVLA (Welch t-test) capture – interleaved RANDOM/FIXED – ATmega AES @ 16 MHz

HOW TO EVALUATE (short):
- This script writes one .NPZ per trace and a CSV index.
- Each NPZ contains: trace (float32), plaintext (uint8[16]), group ("RANDOM"/"FIXED"),
sr_nominal, dt_ns, timebase.
- To compute TVLA (first-order, fixed vs random):
    1) Load all RANDOM traces into a matrix X_R (n_R x T), all FIXED into X_F (n_F x T).
    2) For each sample t, compute Welch t-statistic:
        t(t) = (mean(X_F[:,t]) - mean(X_R[:,t])) /
                sqrt( var(X_F[:,t])/n_F + var(X_R[:,t])/n_R )
    3) Plot t(t) and mark |t|=4.5. Persistent excursions beyond 4.5 indicate leakage.
- Tips:
    * Use identical capture settings/filters for both groups.
    * Interleaving (done here) reduces drift bias.
    * Ciphertexts are NOT fetched (less UART activity in the AES window).

Default profile below targets a short Round-1 window (0.1 ms) at ~125 MS/s (timebase=3).
"""

import os, sys, csv, time, secrets
from pathlib import Path
import numpy as np
from tqdm import tqdm
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

def capture_single_trace(enc, scope, plaintext: bytes, retries: int = 1) -> np.ndarray:
    """
    Arm scope, trigger one AES, capture one block (no averaging).
    Returns waveform (float32). Raises on repeated empty captures.
    """
    enc.ser.reset_input_buffer()
    with muted_stdout():
        enc.set_state(plaintext)   # send PT only; do not fetch CT
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

# ──────────────────────────────────────────────────────────────────
#  Main TVLA capture (INTERLEAVED)
# ──────────────────────────────────────────────────────────────────
def tvla_capture_interleaved(
    n_random       = 10_000,                       # number of RANDOM traces
    n_fixed        = 10_000,                       # number of FIXED traces
    fixed_pt_hex   = "00000000000000000000000000000000",
    # --- Recommended 16 MHz "Round-1" profile (short window, high SR) ---
    duration       = 0.0001,                       # 0.1 ms window (~Round-1)
    device_bits    = 15,                           # 12 is fine; 15 if you want max fidelity
    timebase       = 3,                            # ~125 MS/s on 5000A (driver confirms dt)
    sampling_hz    = 125e6,                        # nominal (metadata)
    # ---
    output_dir     = "data_tvla_INTERLEAVED_16MHz_tb3_125Msps_115200Bd_10kR_10kF_15bit_AC_20MHzBW",
    port           = "COM7",
    baud           = 115200,
    trig_threshold_mV = 300,                       # EXT threshold (~0.8V on ±5V EXT range)
    voltage_range  = "10MV",                       # ±10 mV input on Channel A
    bandwidth_limit_20mhz = True,                  # limit analog BW to 20 MHz
    analogue_offset_mv     = 0.0,                  # TODO
    coupling_mode = "AC"                           # AC coupling (stable baseline)
):
    fixed_pt = bytes.fromhex(fixed_pt_hex)

    # -------- UART ----------
    with muted_stdout():
        enc = EncipherAPI(port=port, baudrate=baud, timeout=2)

    # -------- Scope ----------
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
        output_dir             = output_dir,
        filename_prefix        = "tvla",
        bandwidth_limit_20mhz  = bandwidth_limit_20mhz,
        analogue_offset_mv     = analogue_offset_mv
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "trace_overview.csv"

    try:
        with muted_stdout():
            scope.setup_picoscope()

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["OrderIdx", "Group", "PerGroupIdx", "PlaintextHex",
                "FileName", "Samples", "dt_ns", "timebase"]
            )

            # Interleaving engine:
            # Alternate R, F, R, F, ... but ensure we hit exact n_random and n_fixed.
            r_count, f_count, order_idx = 0, 0, 0
            pbar = tqdm(total=(n_random + n_fixed),
                        desc=f"TVLA interleaved  ({duration*1e3:.1f} ms window)")

            next_group = "R"  # start with RANDOM by default
            while r_count < n_random or f_count < n_fixed:
                # choose group: alternate, but if one is finished, pick the other
                if next_group == "R":
                    if r_count >= n_random and f_count < n_fixed:
                        grp = "F"
                    else:
                        grp = "R"
                else:
                    if f_count >= n_fixed and r_count < n_random:
                        grp = "R"
                    else:
                        grp = "F"

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
                    # soft skip on transient issues; do not increment counts
                    print(f"[Warn] {grp}#{per_group_idx}: capture failed ({e}) – retrying next loop")
                    enc.ser.reset_input_buffer()
                    # flip interleave to not get stuck
                    next_group = "F" if grp == "R" else "R"
                    continue

                # save NPZ
                fname = f"tvla_I_{grp}_{per_group_idx:06d}.npz"
                fpath = Path(output_dir) / fname
                np.savez(
                    fpath,
                    trace     = wf,
                    plaintext = np.frombuffer(pt, np.uint8),
                    group     = np.array(["RANDOM" if grp == "R" else "FIXED"]),
                    sr_nominal= scope.sampling_rate_hz,
                    dt_ns     = getattr(scope, "time_interval_ns", None),
                    timebase  = scope.timebase_final
                )

                # log CSV
                writer.writerow([
                    order_idx,
                    "RANDOM" if grp == "R" else "FIXED",
                    per_group_idx,
                    pt.hex().upper(),
                    fname,
                    len(wf),
                    getattr(scope, "time_interval_ns", None),
                    scope.timebase_final
                ])
                if order_idx and (order_idx % 1000 == 0):
                    csv_f.flush()

                # advance counters / interleave
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
            try:    scope.close()
            except Exception: pass
            try:    enc.close()
            except Exception: pass


if __name__ == "__main__":
    # Recommended defaults: short Round-1 window @ 125 MS/s (timebase=3),
    # 15-bit if available; otherwise set device_bits=12.
    tvla_capture_interleaved(
        n_random = 10_000,
        n_fixed  = 10_000,
        fixed_pt_hex = "00000000000000000000000000000000",
        duration = 0.0001,    # 0.1 ms (Round-1)
        device_bits = 15,
        timebase = 3,
        sampling_hz = 125e6,
        output_dir = "data_tvla_INTERLEAVED_16MHz_tb3_125Msps_115200Bd_10kR_10kF_15bit_AC_20MHzBW",
        port = "COM7",
        baud = 115200,
        trig_threshold_mV = 800,
        voltage_range = "10MV",
        bandwidth_limit_20mhz = True,
        analogue_offset_mv = 0.0,
        coupling_mode = "AC"
    )
