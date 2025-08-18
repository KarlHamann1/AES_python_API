#!/usr/bin/env python3
"""
avg_fft_upto_10mhz.py

- Reads a CSV index and loads traces listed in 'TraceFilePath' (or 'FileName').
- Accepts .npy and .npz; for NPZ, prefers 'trace_mean' then 'trace'. Picks up dt_ns if present.
- Crops every trace to the shortest length so shapes match.
- Optional ROI by samples or time (needs dt from NPZ or DT_NS_OVERRIDE).
- Averages all traces, computes real FFT (rfft), and plots 0..10 MHz in dB.
- Saves the PNG next to the CSV.
"""

from __future__ import annotations
import os, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------- user params ----------------
REPO_ROOT   = Path(__file__).resolve().parents[1]  # AES_PYTHON_API/
DATA_SUBDIR = "pi/data_pi_40dB_block_mult_500_micro"
CSV_NAME    = "trace_overview_mult_block.csv"

FS_HZ            = 62.5e6          # sampling rate (used for rfftfreq if no dt)
MAX_TRACES       = 10_000          # None for all
ROI_SAMPLES      = None            # e.g., (s0, s1) or None
ROI_TIME_S       = None            # e.g., (t0, t1) seconds or None
DT_NS_OVERRIDE   = None            # e.g., 16.0 to force 16 ns/sample

FFT_MAX_HZ       = 10e6            # plot limit
TITLE            = "Frequency Spectrum of Average Trace (0–10 MHz)"
LINE_LABEL       = "Average Trace FFT (up to 10 MHz)"
# ------------------------------------------------


def _csv_rows(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _resolve_trace_path(data_dir: Path, row: dict) -> Path | None:
    # Try common columns
    fname = (row.get("TraceFilePath") or row.get("FileName") or "").strip()
    if not fname:
        return None
    p = Path(fname)
    return p if p.is_file() else (data_dir / p)


def _load_waveform(path: Path, dt_holder: dict) -> np.ndarray:
    """Load .npy or .npz. For NPZ prefer 'trace_mean', then 'trace'. Pick up dt_ns if present."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".npz":
        with np.load(path) as npz:
            if "trace_mean" in npz.files:
                wf = np.asarray(npz["trace_mean"], dtype=np.float32)
            elif "trace" in npz.files:
                wf = np.asarray(npz["trace"], dtype=np.float32)
            else:
                raise ValueError(f"{path.name}: NPZ missing 'trace_mean'/'trace'")
            if dt_holder.get("dt") is None and "dt_ns" in npz.files:
                try:
                    dt_holder["dt"] = float(npz["dt_ns"].item())
                except Exception:
                    pass
        return wf
    else:
        return np.asarray(np.load(path), dtype=np.float32)


def main():
    data_dir = (REPO_ROOT / DATA_SUBDIR)
    csv_path = data_dir / CSV_NAME
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}")
        return

    # Keep dt if NPZ provides it (or override)
    dt_holder = {"dt": (DT_NS_OVERRIDE * 1e-9) if DT_NS_OVERRIDE else None}

    traces = []
    lengths = []

    # Load traces listed in CSV
    for i, row in enumerate(_csv_rows(csv_path)):
        tpath = _resolve_trace_path(data_dir, row)
        if not tpath or not tpath.suffix.lower() in (".npy", ".npz"):
            continue
        try:
            wf = _load_waveform(tpath, dt_holder)
        except Exception as e:
            print(f"[warn] skip {tpath.name}: {e}")
            continue
        if wf.ndim != 1 or wf.size == 0:
            print(f"[warn] {tpath.name}: not a 1D non-empty array")
            continue
        traces.append(wf)
        lengths.append(wf.size)
        if MAX_TRACES and len(traces) >= MAX_TRACES:
            break

    if not traces:
        print("No traces loaded.")
        return

    # Make shapes match (crop to shortest)
    min_len = int(min(lengths))
    traces = [t[:min_len] for t in traces]
    L = min_len

    # Optional ROI before averaging/FFT
    s0, s1 = 0, L
    if ROI_SAMPLES is not None:
        s0 = max(0, int(ROI_SAMPLES[0])); s1 = min(L, int(ROI_SAMPLES[1]))
        traces = [t[s0:s1] for t in traces]
        L = s1 - s0
        print(f"Applied ROI_SAMPLES=({s0},{s1}) → {L} samples.")
    elif ROI_TIME_S is not None and dt_holder["dt"]:
        dt = float(dt_holder["dt"])
        s0 = max(0, int(round(ROI_TIME_S[0] / dt)))
        s1 = min(L, int(round(ROI_TIME_S[1] / dt)))
        traces = [t[s0:s1] for t in traces]
        L = s1 - s0
        print(f"Applied ROI_TIME_S=({ROI_TIME_S[0]:.3e},{ROI_TIME_S[1]:.3e}) → samples ({s0},{s1}) → {L}.")

    if L < 8:
        print("[warn] ROI too small.")
        return

    # Average
    avg_trace = np.mean(np.stack(traces, axis=0), axis=0).astype(np.float64, copy=False)

    # FFT (real). Prefer dt if known; else use FS_HZ.
    if dt_holder["dt"]:
        d = float(dt_holder["dt"])
        freqs = np.fft.rfftfreq(L, d=d)
    else:
        freqs = np.fft.rfftfreq(L, d=1.0 / FS_HZ)
    fft_vals = np.fft.rfft(avg_trace)
    mag_db   = 20.0 * np.log10(np.maximum(np.abs(fft_vals), 1e-12))

    # Limit to <= FFT_MAX_HZ
    mask = freqs <= FFT_MAX_HZ
    freqs_plot = freqs[mask]
    mag_db_plot = mag_db[mask]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_plot, mag_db_plot, label=LINE_LABEL)
    plt.title(TITLE)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    out_png = data_dir / "AverageTrace_FFT_UpTo10MHz.png"
    plt.savefig(out_png, dpi=140)
    plt.show()
    print(f"FFT plot saved to: {out_png}")

if __name__ == "__main__":
    main()
