# Summarize TVLA evals stored as:
#   arduino/data/TVLA_eval/
#       1MHz/
#         t_statistic.npy
#         time_axis.npy    (optional; seconds)
#       8MHz/
#       16MHz/
#
# Writes into COMPARISON:
#   - tvla_percentages_over_threshold.csv
#   - tvla_percentages_over_threshold_samples.png
#   - tvla_percentages_over_threshold_time.png

from __future__ import annotations
import csv
from pathlib import Path
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

#  paths 
SCRIPT_DIR  = Path(__file__).resolve().parent          # .../arduino/scripts
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
EVAL_ROOT   = (ARDUINO_DIR / "data" / "TVLA_eval").resolve()

DATASETS = {
    "1MHz":  EVAL_ROOT / "1MHz",
    "8MHz":  EVAL_ROOT / "8MHz",
    "16MHz": EVAL_ROOT / "16MHz",
}

# comparison dir (create if missing)
COMPARISON_DIR = (EVAL_ROOT / "COMPARISON")
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

TVLA_THRESHOLD = 4.5

def pct_samples_above(t: np.ndarray, thr: float) -> float:
    """Percentage of samples with |t| >= thr."""
    return float(np.mean(np.abs(t) >= thr) * 100.0)

def pct_time_above(t: np.ndarray, thr: float, time_axis: np.ndarray | None) -> float | None:
    """
    Percentage of total time with |t| >= thr.
    If time_axis is missing or invalid, returns None.
    """
    if time_axis is None:
        return None
    if len(time_axis) != len(t):
        return None
    if len(time_axis) < 2:
        return None

    # estimate per-sample duration from forward diffs, extend last with previous
    dt_seg = np.diff(time_axis)
    if not np.all(dt_seg > 0):
        # non-monotonic or zero step -> bail out
        return None
    dt_seg = np.append(dt_seg, dt_seg[-1])

    mask = np.abs(t) >= thr
    total = float(np.sum(dt_seg))
    above = float(np.sum(dt_seg[mask]))
    if total <= 0:
        return None
    return (above / total) * 100.0

def load_time_axis(folder: Path) -> np.ndarray | None:
    # prefer time_axis.npy; fallback to sample_axis.npy; else None
    for name in ("time_axis.npy", "sample_axis.npy"):
        p = folder / name
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                pass
    return None

def main():
    print("EVAL_ROOT:", EVAL_ROOT)

    rows = []
    for tag, fdir in DATASETS.items():
        if not fdir.exists():
            print(f"  {tag}: NOT FOUND -> {fdir}")
            continue
        print(f"  {tag}: FOUND -> {fdir}")

        t_path = fdir / "t_statistic.npy"
        if not t_path.exists():
            print(f"    [skip] missing t_statistic.npy in {fdir}")
            continue

        try:
            t = np.load(t_path)
        except Exception as e:
            print(f"    [skip] failed to load {t_path}: {e}")
            continue

        time_axis = load_time_axis(fdir)
        pct_samp = pct_samples_above(t, TVLA_THRESHOLD)
        pct_time = pct_time_above(t, TVLA_THRESHOLD, time_axis)

        rows.append({
            "dataset": tag,
            "eval_dir": str(fdir),
            "threshold": TVLA_THRESHOLD,
            "num_samples": int(t.shape[0]),
            "max_abs_t": float(np.max(np.abs(t))),
            "percent_ge_threshold_samples": pct_samp,
            "percent_ge_threshold_time": (float(pct_time) if pct_time is not None else None),
        })

    if not rows:
        print("No TVLA eval folders found with t_statistic.npy; nothing to do.")
        return

    # ---- CSV ----
    csv_path = COMPARISON_DIR / "tvla_percentages_over_threshold.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Wrote:", csv_path)

    # ---- plots ----
    labels     = [r["dataset"] for r in rows]
    perc_samps = [r["percent_ge_threshold_samples"] for r in rows]
    perc_time  = [r["percent_ge_threshold_time"] for r in rows]

    # samples-based bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(labels, perc_samps)
    plt.ylabel(f"% samples with |t| ≥ {TVLA_THRESHOLD}")
    plt.title("TVLA: % of samples above threshold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png_samples = COMPARISON_DIR / "tvla_percentages_over_threshold_samples.png"
    plt.savefig(out_png_samples, dpi=150)
    plt.close()
    print("Wrote:", out_png_samples)

    # time-based bar chart (only if we have any time percentages)
    if any(p is not None for p in perc_time):
        # replace None with 0 just for plotting; annotate missing below
        to_plot = [p if p is not None else 0.0 for p in perc_time]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, to_plot)
        plt.ylabel(f"% time with |t| ≥ {TVLA_THRESHOLD}")
        plt.title("TVLA: % of time above threshold")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_png_time = COMPARISON_DIR / "tvla_percentages_over_threshold_time.png"
        plt.savefig(out_png_time, dpi=150)
        plt.close()
        print("Wrote:", out_png_time)

        # small note for missing time axes
        missing = [lab for lab, p in zip(labels, perc_time) if p is None]
        if missing:
            note = ("Some datasets had no usable time axis; "
                    "their time-percentage was omitted: " + ", ".join(missing))
            (COMPARISON_DIR / "README_time_percentage.txt").write_text(note, encoding="utf-8")
            print(note)

if __name__ == "__main__":
    main()
