#!/usr/bin/env python3
"""
cpa_heatmap_metrics.py

Consume previously saved CPA heatmap arrays and emit quantitative metrics:
- ridge occupancy (fraction of ROI where argmax guess == true key)
- longest contiguous run with correct ridge (samples and time)
- number of correct segments
- peak/mean |corr| for the true key row over correct-ridge samples
- global peak |corr| and when it occurs
Also produce 1D figures and cross-dataset overlays.

Inputs per dataset directory (under arduino/data/CPA_heatmap/<Label>/):
- corr_abs_bXX.npy                  shape (256, T) of |rho|
- max_abs_corr_over_time_bXX.npy    shape (T,)     of max_g |rho(g,t)|
- argmax_guess_over_time_bXX.npy    shape (T,)     of argmax_g |rho(g,t)|
- time_axis.npy or sample_axis.npy  shape (T,)     x-axis

Outputs per dataset:
- metrics_bXX.json, metrics_bXX.txt
- max_corr_curve_bXX.png
- ridge_truekey_mask_bXX.png
- ridge_segments_bXX.csv

Cross-dataset (under CPA_heatmap/COMPARISON):
- heatmap_metrics_bXX.csv, heatmap_metrics_bXX.json
- max_corr_overlay_normalized_bXX.png
- ridge_occupancy_bar_bXX.png
"""

from __future__ import annotations
import json, csv, re, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# repo layout
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
HEAT_ROOT   = (DATA_ROOT / "CPA_heatmap").resolve()

# dataset labels expected under CPA_heatmap/
DATASET_LABELS = ["1MHz", "8MHz", "16MHz", "16MHz_round1_125Msps"]

# byte index and true key
BYTE_INDEX   = 0
TRUE_KEY_HEX = "81556407E2B7C090D23146B46CD554A2"   # set "" to skip true-key metrics

# plotting
FIG_DPI = 170

def true_key_byte(hex16: str, byte_index: int) -> Optional[int]:
    h = (hex16 or "").strip().replace(" ", "")
    if len(h) != 32:
        return None
    return int(h[2*byte_index:2*byte_index+2], 16)

def contiguous_runs(mask: np.ndarray) -> List[Tuple[int,int]]:
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    diff = np.diff(m, prepend=m[:1])
    starts = np.where(diff == 1)[0].tolist()
    ends   = np.where(diff == -1)[0].tolist()
    if mask[-1]:
        ends.append(mask.size)
    return [(s, e) for s, e in zip(starts, ends) if e > s]

def load_axis(d: Path) -> Tuple[np.ndarray, bool]:
    p_time   = d / "time_axis.npy"
    p_sample = d / "sample_axis.npy"
    if p_time.exists():
        return np.load(p_time), True
    if p_sample.exists():
        return np.load(p_sample), False
    raise FileNotFoundError(f"{d}: missing time_axis.npy/sample_axis.npy")

def eval_dataset_dir(d: Path, byte_index: int, tk: Optional[int]) -> Optional[Dict]:
    # required files
    corr_path = d / f"corr_abs_b{byte_index:02d}.npy"
    mx_path   = d / f"max_abs_corr_over_time_b{byte_index:02d}.npy"
    am_path   = d / f"argmax_guess_over_time_b{byte_index:02d}.npy"
    if not corr_path.exists() and not mx_path.exists():
        print(f"[skip] {d.name}: no heatmap arrays found")
        return None

    # axis and arrays
    axis, is_time = load_axis(d)
    if corr_path.exists():
        corr = np.load(corr_path)            # (256, T)
        if corr.ndim != 2:
            print(f"[warn] {d.name}: corr array shape unexpected, skipping")
            return None
        max_over_g = corr.max(axis=0)        # (T,)
    else:
        corr = None
        max_over_g = np.load(mx_path)

    argmax_g = np.load(am_path)              # (T,)
    T = max_over_g.shape[0]

    # mask where ridge equals true key
    tk_byte = tk
    if tk_byte is None:
        mask = np.zeros(T, dtype=bool)
    else:
        mask = (argmax_g.astype(np.int64) == int(tk_byte))

    # per-true-key correlation statistics if corr present
    peak_true = None
    mean_true = None
    if corr is not None and tk_byte is not None:
        row = corr[int(tk_byte)]
        if mask.any():
            peak_true = float(np.max(row[mask]))
            mean_true = float(np.mean(row[mask]))
        else:
            peak_true = float(np.max(row))
            mean_true = float(np.mean(row))

    # runs
    runs = contiguous_runs(mask)
    n_runs = len(runs)
    longest_samples = max((e - s) for s, e in runs) if runs else 0

    # durations
    dt = None
    longest_time = None
    occ_time = None
    if is_time and T >= 2:
        # axis is in seconds; assume uniform spacing
        dt = float(axis[1] - axis[0]) if axis.size > 1 else None
        if dt is not None:
            longest_time = float(longest_samples * dt)
            occ_time = float(mask.sum() * dt)

    # global peak time
    peak_idx = int(np.argmax(max_over_g))
    peak_time = float(axis[peak_idx])

    # prepare per-dataset metrics
    metrics = {
        "label": d.name,
        "byte": int(byte_index),
        "T": int(T),
        "axis_is_time": bool(is_time),
        "ridge_occupancy_fraction": float(mask.mean()) if T else 0.0,
        "ridge_occupancy_samples": int(mask.sum()),
        "n_correct_segments": int(n_runs),
        "longest_correct_run_samples": int(longest_samples),
        "longest_correct_run_seconds": longest_time,
        "occupancy_seconds": occ_time,
        "peak_maxcorr": float(max_over_g.max()),
        "peak_maxcorr_at_idx": peak_idx,
        "peak_maxcorr_at_axis": peak_time,
        "peak_truekey_corr": peak_true,
        "mean_truekey_corr_over_correct": mean_true,
    }

    # save per-dataset figures
    out_curve = d / f"max_corr_curve_b{byte_index:02d}.png"
    plt.figure(figsize=(10, 4))
    plt.plot(axis, max_over_g, linewidth=1.2)
    plt.xlabel("Time (s)" if is_time else "Sample index")
    plt.ylabel("max |corr| over guesses")
    plt.title(f"{d.name} – max|corr|(t) – byte {byte_index}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_curve, dpi=FIG_DPI)
    plt.close()

    out_mask = d / f"ridge_truekey_mask_b{byte_index:02d}.png"
    plt.figure(figsize=(10, 1.8))
    # step-like mask visualization
    x = axis
    y = mask.astype(float)
    plt.plot(x, y, drawstyle="steps-post", linewidth=1.2)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 1], ["other", "true"])
    plt.xlabel("Time (s)" if is_time else "Sample index")
    plt.title(f"{d.name} – argmax ridge equals true key – byte {byte_index}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_mask, dpi=FIG_DPI)
    plt.close()

    # save runs to CSV
    with open(d / f"ridge_segments_b{byte_index:02d}.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["start_idx", "end_idx", "length_samples"]
        if is_time:
            cols += ["start_time_s", "end_time_s", "length_s"]
        w.writerow(cols)
        for s, e in runs:
            row = [int(s), int(e), int(e - s)]
            if is_time:
                row += [float(axis[s]), float(axis[e-1]), float((e - s) * dt if dt else 0.0)]
            w.writerow(row)

    # per-dataset text + json
    (d / f"metrics_b{byte_index:02d}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        f"dataset: {d.name}",
        f"byte_index: {byte_index}",
        f"T: {T}",
        f"axis: {'time (s)' if is_time else 'samples'}",
        f"ridge occupancy (fraction): {metrics['ridge_occupancy_fraction']:.4f}",
        f"ridge occupancy (samples):  {metrics['ridge_occupancy_samples']}",
        f"n correct segments:         {n_runs}",
        f"longest correct run (samples): {longest_samples}",
        f"longest correct run (seconds): {metrics['longest_correct_run_seconds'] if metrics['longest_correct_run_seconds'] is not None else 'n/a'}",
        f"peak max|corr|: {metrics['peak_maxcorr']:.6f} at axis={metrics['peak_maxcorr_at_axis']:.6e}",
        f"peak true-key |corr| over correct windows: {metrics['peak_truekey_corr'] if metrics['peak_truekey_corr'] is not None else 'n/a'}",
        f"mean true-key |corr| over correct windows: {metrics['mean_truekey_corr_over_correct'] if metrics['mean_truekey_corr_over_correct'] is not None else 'n/a'}",
        f"curve_png: {out_curve}",
        f"mask_png:  {out_mask}",
    ]
    (d / f"metrics_b{byte_index:02d}.txt").write_text("\n".join(lines), encoding="utf-8")

    return metrics, axis, max_over_g

def write_comparison(metrics_list: List[Dict], curves: List[Tuple[str, np.ndarray, np.ndarray]], byte_index: int):
    comp_dir = HEAT_ROOT / "COMPARISON"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # CSV/JSON
    if metrics_list:
        keys = [
            "label","byte","T","axis_is_time",
            "ridge_occupancy_fraction","ridge_occupancy_samples",
            "n_correct_segments","longest_correct_run_samples","longest_correct_run_seconds",
            "occupancy_seconds","peak_maxcorr","peak_maxcorr_at_idx","peak_maxcorr_at_axis",
            "peak_truekey_corr","mean_truekey_corr_over_correct"
        ]
        with open(comp_dir / f"heatmap_metrics_b{byte_index:02d}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for m in metrics_list:
                w.writerow({k: m.get(k) for k in keys})
        (comp_dir / f"heatmap_metrics_b{byte_index:02d}.json").write_text(json.dumps(metrics_list, indent=2), encoding="utf-8")

    # Overlay of max|corr|(t) with normalized x-axis
    if curves:
        plt.figure(figsize=(10, 4))
        for label, axis, y in curves:
            if axis.size != y.size or axis.size == 0:
                continue
            x_norm = np.linspace(0.0, 1.0, num=axis.size)
            plt.plot(x_norm, y, linewidth=1.2, label=label)
        plt.xlabel("Normalized time (0–1)")
        plt.ylabel("max |corr| over guesses")
        plt.title(f"CPA max|corr|(t) overlay (normalized) – byte {byte_index}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / f"max_corr_overlay_normalized_b{byte_index:02d}.png", dpi=FIG_DPI)
        plt.close()

    # Bar chart of ridge occupancy
    if metrics_list:
        labels = [m["label"] for m in metrics_list]
        occ =    [m["ridge_occupancy_fraction"] for m in metrics_list]
        plt.figure(figsize=(8, 4))
        idx = np.arange(len(labels))
        plt.bar(idx, occ)
        plt.xticks(idx, labels, rotation=0)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Ridge occupancy (fraction)")
        plt.title(f"Fraction of ROI where argmax ridge = true key – byte {byte_index}")
        plt.tight_layout()
        plt.savefig(comp_dir / f"ridge_occupancy_bar_b{byte_index:02d}.png", dpi=FIG_DPI)
        plt.close()

def main():
    # optional CLI: dataset labels to include, else use defaults
    labels = sys.argv[1:] if len(sys.argv) > 1 else DATASET_LABELS
    labels = list(dict.fromkeys(labels))  # uniq, preserve order

    tk = true_key_byte(TRUE_KEY_HEX, BYTE_INDEX)

    metrics_all = []
    curves_all = []

    for lab in labels:
        d = HEAT_ROOT / lab
        if not d.exists():
            print(f"[skip] {lab}: not found under {HEAT_ROOT}")
            continue
        out = eval_dataset_dir(d, BYTE_INDEX, tk)
        if out is None:
            continue
        metrics, axis, max_over_g = out
        metrics_all.append(metrics)
        curves_all.append((lab, axis, max_over_g))

    if metrics_all:
        write_comparison(metrics_all, curves_all, BYTE_INDEX)
        print(f"Saved per-dataset metrics and comparison to {HEAT_ROOT}")
    else:
        print("No metrics produced. Check inputs under CPA_heatmap/<Label>/.")

if __name__ == "__main__":
    main()
