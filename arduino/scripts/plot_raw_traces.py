#!/usr/bin/env python3
"""

Simple visualizations of raw side-channel traces for a fixed set of datasets:
- one single trace
- overlay of N traces (no alignment)
- optional overlay after a light cross-correlation alignment
- grand mean with a ±1 SD band

Outputs are written next to CPA results, grouped by a pretty dataset label.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

#  repo layout 

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "CPA_eval").resolve()

# keep this for folder naming, even if not used inside the plots directly
BYTE_INDEX = 0

#  datasets to scan 

RAW_LABELS = [
    "1MHz",
    "8MHz",
    "16MHz",
    "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV",
]

LABEL_MAP: Dict[str, str] = {
    "1MHz": "1MHz",
    "8MHz": "8MHz",
    "16MHz": "16MHz",
    "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV": "16MHz_1round_125Msps",
}

# flat/dead-trace filter 

P2P_MIN_FRACTION = 0.20
STD_MIN_FRACTION = 0.15

def robust_filter_flags(traces: List[np.ndarray]) -> np.ndarray:
    """Flag traces that look dead/flat based on p2p and std."""
    p2p = np.array([float(np.max(t) - np.min(t)) for t in traces], dtype=np.float64)
    sd  = np.array([float(np.std(t))            for t in traces], dtype=np.float64)
    med_p2p = np.median(p2p) if np.isfinite(p2p).all() else 0.0
    med_sd  = np.median(sd)  if np.isfinite(sd).all()  else 0.0
    ok_p2p = p2p >= (P2P_MIN_FRACTION * max(med_p2p, 1e-12))
    ok_sd  = sd  >= (STD_MIN_FRACTION  * max(med_sd,  1e-12))
    return ok_p2p & ok_sd


def find_dataset_dir(label: str) -> Optional[Path]:
    # exact match first
    exact = DATA_ROOT / label
    if exact.exists() and exact.is_dir():
        return exact
    # otherwise pick the shortest directory name that contains the label
    cands = [p for p in DATA_ROOT.iterdir() if p.is_dir() and (label in p.name)]
    if not cands:
        return None
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]

def load_rows(ds_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    csv_path = ds_dir / "trace_overview.csv"
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        for p in sorted(ds_dir.glob("*.npz")):
            rows.append({"FileName": p.name})
    return rows

def open_trace_any(p: Path) -> Tuple[np.ndarray, Optional[float]]:
    # returns (waveform, dt_seconds or None)
    if p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=False)
        if "trace" in z.files:
            x = np.asarray(z["trace"], dtype=np.float64)
        elif "trace_mean" in z.files:
            x = np.asarray(z["trace_mean"], dtype=np.float64)
        else:
            raise ValueError(f"{p.name}: no 'trace'/'trace_mean'")
        dt = None
        if "dt_ns" in z.files:
            dt = float(z["dt_ns"].item()) * 1e-9
        elif "sr_nominal" in z.files:
            fs = float(z["sr_nominal"].item());  dt = 1.0 / fs if fs > 0 else None
        elif "sr" in z.files:
            fs = float(z["sr"].item());          dt = 1.0 / fs if fs > 0 else None
        return x, dt
    elif p.suffix.lower() == ".npy":
        x = np.load(p).astype(np.float64, copy=False)
        return x, None
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

def choose_file_paths(rows: List[Dict[str,str]], ds_dir: Path, limit: Optional[int]=None) -> List[Path]:
    out: List[Path] = []
    for r in rows:
        fn = r.get("FileName") or r.get("TraceFilePath")
        if not fn:
            continue
        p = Path(fn)
        if not p.exists():
            p = ds_dir / fn
        if p.exists():
            out.append(p)
        if limit and len(out) >= limit:
            break
    return out

def time_axis(n: int, dt: Optional[float]) -> Tuple[np.ndarray, str]:
    # returns (axis, unit_str)
    if dt and dt > 0:
        return np.arange(n, dtype=np.float64) * dt, "s"
    return np.arange(n, dtype=np.float64), "samples"

def detrend_mean(x: np.ndarray) -> np.ndarray:
    # keep it simple: remove per-trace DC for nicer plots
    return x - np.mean(x)

# optional alignment 

def align_xcorr(traces: List[np.ndarray], ref: np.ndarray, max_lag: int, win: Optional[Tuple[int,int]]) -> List[np.ndarray]:
    # quick-and-dirty xcorr alignment around a small window
    def cut(y: np.ndarray) -> np.ndarray:
        if not win:
            return y
        s0, s1 = max(0, win[0]), min(len(y), win[1])
        return y[s0:s1]

    ref_w = cut(ref)
    out = [ref]
    for x in traces[1:]:
        xw = cut(x)
        L = min(len(ref_w), len(xw))
        a = ref_w[:L] - np.mean(ref_w[:L])
        b = xw[:L]    - np.mean(xw[:L])
        lag = 0
        if max_lag > 0 and L >= 8:
            best = -1e18
            for k in range(-max_lag, max_lag+1):
                if k >= 0:
                    u = a[k:]; v = b[:len(a)-k]
                else:
                    u = a[:len(a)+k]; v = b[-k:]
                if len(u) < 8:
                    break
                r = float(np.dot(u, v)) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-18)
                if r > best:
                    best = r; lag = k
        out.append(np.roll(x, -lag))
    return out

# streaming mean / sd 

def streaming_mean_sd(files: List[Path]) -> Tuple[np.ndarray, np.ndarray, Optional[float], int]:
    # figure out a common length and dt first
    L_min = None
    dt = None
    for p in files:
        try:
            x, dtt = open_trace_any(p)
        except Exception:
            continue
        L_min = len(x) if L_min is None else min(L_min, len(x))
        if dt is None and dtt:
            dt = dtt
    if not L_min:
        return np.zeros(0,), np.zeros(0,), None, 0

    # Welford-style streaming stats (vectorized)
    mean = np.zeros(L_min, dtype=np.float64)
    m2   = np.zeros(L_min, dtype=np.float64)
    n    = 0
    for p in files:
        try:
            x, _ = open_trace_any(p)
        except Exception:
            continue
        y = detrend_mean(x[:L_min])
        n += 1
        delta = y - mean
        mean += delta / n
        m2   += delta * (y - mean)
    var = m2 / max(n - 1, 1)
    sd  = np.sqrt(np.maximum(var, 0.0))
    return mean, sd, dt, n


def plot_single(t: np.ndarray, unit: str, x: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(10, 3.5))
    plt.plot(t, x, linewidth=0.9)
    plt.xlabel(f"Time ({unit})")
    plt.ylabel("Voltage (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_overlay(t: np.ndarray, unit: str, traces: List[np.ndarray], out_path: Path, title: str, alpha=0.55):
    plt.figure(figsize=(10, 3.8))
    for x in traces:
        plt.plot(t[:len(x)], x, linewidth=0.8, alpha=alpha)
    plt.xlabel(f"Time ({unit})")
    plt.ylabel("Voltage (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_mean_sd(t: np.ndarray, unit: str, mean: np.ndarray, sd: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(10, 3.5))
    plt.plot(t, mean, linewidth=1.2, label="mean")
    plt.fill_between(t, mean - sd, mean + sd, alpha=0.25, label="±1 SD")
    plt.xlabel(f"Time ({unit})")
    plt.ylabel("Voltage (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def process_dataset(label: str, pretty: str, n_overlay: int, max_lag: int, win: Optional[Tuple[int,int]], out_root: Path):
    ds_dir = find_dataset_dir(label)
    if not ds_dir:
        print(f"[skip] dataset not found for label: {label}")
        return

    out_dir = out_root / pretty
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(ds_dir)
    files_all = choose_file_paths(rows, ds_dir, limit=None)
    if not files_all:
        print(f"[skip] {pretty}: no trace files")
        return

    # small subset for the overlay figures
    files_small = files_all[:max(n_overlay, 1)]
    traces_raw: List[np.ndarray] = []
    dt = None
    for p in files_small:
        try:
            x, dtt = open_trace_any(p)
        except Exception:
            continue
        traces_raw.append(x.astype(np.float64, copy=False))
        if dt is None and dtt:
            dt = dtt
    if not traces_raw:
        print(f"[skip] {pretty}: failed to load overlay traces")
        return

    # drop obviously flat traces for the overlay (keeps the picture honest)
    flags = robust_filter_flags(traces_raw)
    traces_ok = [traces_raw[i] for i, ok in enumerate(flags) if ok]
    if not traces_ok:
        traces_ok = traces_raw

    # time axis
    L0 = len(traces_ok[0])
    t, unit = time_axis(L0, dt)

    # single trace
    plot_single(t, unit, traces_ok[0], out_dir / "RAW_single.png",
                title=f"Single raw trace — {pretty}")

    # overlay (no alignment)
    n_show = min(n_overlay, len(traces_ok))
    plot_overlay(t, unit, [traces_ok[i] for i in range(n_show)],
                out_dir / f"RAW_overlay_noalign_{n_show}.png",
                title=f"Overlay of {n_show} raw traces (no alignment) — {pretty}")

    # overlay after xcorr alignment (optional)
    if max_lag > 0:
        aligned = align_xcorr(traces_ok[:n_show], traces_ok[0], max_lag=max_lag, win=win)
        wtxt = f"[{win[0]},{win[1]})" if win else "full"
        plot_overlay(t, unit, aligned,
                    out_dir / f"RAW_overlay_xcorr_{n_show}.png",
                    title=f"Overlay after xcorr resync (win={wtxt}, ±{max_lag} samples) — {pretty}")

    # grand mean (+/− SD) streaming over all files
    mean, sd, dt_all, n_used = streaming_mean_sd(files_all)
    if mean.size > 0:
        tt, unit2 = time_axis(mean.size, dt_all if dt_all else dt)
        plot_mean_sd(tt, unit2, mean, sd, out_dir / "RAW_mean_sd.png",
                    title=f"Grand mean (±1 SD) — {pretty} (n={n_used})")

    # short text summary for quick reference
    (out_dir / "raw_summary.txt").write_text(
        "\n".join([
            f"dataset_dir: {ds_dir}",
            f"pretty_label: {pretty}",
            f"overlay_traces_loaded: {len(traces_raw)}  kept_after_filter: {len(traces_ok)}",
            f"grand_mean_traces_used: {n_used}",
            f"sample_length_first: {L0}",
            f"dt_inferred: {dt if dt else 'unknown'}",
            f"alignment: {'xcorr' if max_lag>0 else 'none'}  win: {win if win else 'full'}  max_lag: {max_lag}",
        ]),
        encoding="utf-8"
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_overlay", type=int, default=10, help="number of traces to overlay")
    ap.add_argument("--max_lag",  type=int, default=0,  help="±lag (samples) for xcorr alignment; 0 disables")
    ap.add_argument("--win",      type=str, default=None, help="alignment window s0:s1 in samples")
    args = ap.parse_args()

    win = None
    if args.win:
        s0, s1 = args.win.split(":")
        win = (int(s0), int(s1))

    out_root = (EVAL_ROOT / f"RAW_viz_b{BYTE_INDEX:02d}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for label in RAW_LABELS:
        pretty = LABEL_MAP.get(label, label)
        process_dataset(label, pretty, n_overlay=args.n_overlay,
                        max_lag=args.max_lag, win=win, out_root=out_root)

    print(f"raw visualizations saved under: {out_root}")

if __name__ == "__main__":
    main()
