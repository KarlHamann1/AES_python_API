#!/usr/bin/env python3
"""
tvla_evaluate.py

Compute TVLA (Welch t-test) for one or more interleaved datasets (RANDOM/FIXED).
This version also:
- writes per-dataset normalized |t| vectors (1000 samples) to the eval folder,
- writes a combined abs_t_normalized_all.npz for cross-dataset work,
- plots per-dataset t-curves + cross-dataset overlays (samples & normalized),
- and reports contiguous "leakage windows" where |t| >= threshold.

Notes:
- Uses absolute paths for the three default datasets under arduino/data.
- If dt_ns or sr_nominal present, time vectors are available; overlays use
normalized sample index for comparability across very different windows/SR.
"""

from __future__ import annotations
import os, re, sys, csv, json, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

#  Path config (absolute) 
SCRIPT_DIR  = Path(__file__).resolve().parent               # .../arduino/scripts
REPO_ROOT   = SCRIPT_DIR.parents[1]                          # .../AES_PYTHON_API
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()


STAMP      = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
EVAL_ROOT  = (DATA_ROOT / "TVLA_eval" / f"TVLA_eval_{STAMP}").resolve()
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

# Absolute dataset folders (edit names only if yours differ)
DATASETS = [
    DATA_ROOT / "data_tvla_INTERLEAVED_1MHz_tb14_10_416Msps_9600Bd_10kR_10kF_15bit_AC_20MHzBW",
    DATA_ROOT / "data_tvla_INTERLEAVED_8MHz_tb4_62_5Msps_57600Bd_10kR_10kF_15bit_AC_20MHzBW",
    DATA_ROOT / "data_tvla_INTERLEAVED_16MHz_tb3_125Msps_115200Bd_10kR_10kF_15bit_AC_20MHzBW",
]
# Optional ROI in samples (start, end) or None
ROI_SAMPLES: Optional[Tuple[int, int]] = None

# "flat trace" filters (relative to group medians)
P2P_MIN_FRACTION = 0.20
STD_MIN_FRACTION = 0.15

TVLA_THRESHOLD = 4.5

# Helpers 
def find_npz_files_by_group(ds_dir: Path) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    r_pat = re.compile(r"tvla_I_R_(\d{6})\.npz$", re.IGNORECASE)
    f_pat = re.compile(r"tvla_I_F_(\d{6})\.npz$", re.IGNORECASE)
    R, F = {}, {}
    for p in sorted(ds_dir.glob("*.npz")):
        mR = r_pat.search(p.name)
        mF = f_pat.search(p.name)
        if mR: R[int(mR.group(1))] = p
        elif mF: F[int(mF.group(1))] = p
    return R, F

def load_npz_trace(path: Path) -> Tuple[np.ndarray, Dict]:
    npz = np.load(path, allow_pickle=False)
    if   "trace" in npz.files:       wf = np.asarray(npz["trace"], dtype=np.float32)
    elif "trace_mean" in npz.files:  wf = np.asarray(npz["trace_mean"], dtype=np.float32)
    else: raise ValueError(f"{path.name}: no 'trace' or 'trace_mean'.")
    meta = {
        "dt_ns": float(npz["dt_ns"].item()) if "dt_ns" in npz.files else None,
        "sr_nominal": float(npz["sr_nominal"].item()) if "sr_nominal" in npz.files else None,
        "timebase": int(npz["timebase"].item()) if "timebase" in npz.files else None,
    }
    return wf, meta

def time_axis_from_meta(n: int, meta: Dict) -> Tuple[np.ndarray, Optional[float]]:
    dt_s = None
    if meta.get("dt_ns"): dt_s = meta["dt_ns"] * 1e-9
    elif meta.get("sr_nominal"): 
        sr = meta["sr_nominal"]
        if sr and sr > 0: dt_s = 1.0 / sr
    if dt_s: return (np.arange(n, dtype=np.float64) * dt_s), dt_s
    return (np.arange(n, dtype=np.float64), None)

def robust_filter_flags(traces: List[np.ndarray]) -> np.ndarray:
    if not traces: return np.array([], dtype=bool)
    p2p = np.array([float(np.max(t) - np.min(t)) for t in traces], dtype=np.float64)
    std = np.array([float(np.std(t))              for t in traces], dtype=np.float64)
    med_p2p = np.median(p2p) if np.isfinite(p2p).all() else 0.0
    med_std = np.median(std) if np.isfinite(std).all() else 0.0
    ok_p2p = p2p >= (P2P_MIN_FRACTION * max(med_p2p, 1e-12))
    ok_std = std >= (STD_MIN_FRACTION * max(med_std, 1e-12))
    return ok_p2p & ok_std

def align_length(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = min(A.shape[1], B.shape[1])
    A = A[:, :L]; B = B[:, :L]
    if ROI_SAMPLES:
        s0, s1 = ROI_SAMPLES
        s0 = max(0, int(s0)); s1 = min(L, int(s1))
        A = A[:, s0:s1]; B = B[:, s0:s1]
    return A, B

def compute_tvla(XF: np.ndarray, XR: np.ndarray) -> np.ndarray:
    nF, T = XF.shape; nR, _ = XR.shape
    muF = XF.mean(axis=0); muR = XR.mean(axis=0)
    vF  = XF.var(axis=0, ddof=1) if nF > 1 else np.zeros(T)
    vR  = XR.var(axis=0, ddof=1) if nR > 1 else np.zeros(T)
    denom = np.sqrt(vF / max(nF,1) + vR / max(nR,1))
    t = np.zeros(T, dtype=np.float64); nz = denom > 1e-18
    t[nz] = (muF[nz] - muR[nz]) / denom[nz]
    return t

def dataset_label_from_path(p: Path) -> str:
    m = re.search(r"(\d+)\s*MHz", p.name, re.IGNORECASE)
    return f"{m.group(1)}MHz" if m else p.name

def make_out_dir_for_label(label: str) -> Path:
    """
    Create/return the per-dataset output folder under the central EVAL_ROOT.
    """
    out = EVAL_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    return out

def normalize_for_overlay(x: np.ndarray, N:int=1000) -> np.ndarray:
    if x.shape[0] == N: return x.astype(np.float64, copy=True)
    src = np.linspace(0.0, 1.0, x.shape[0]); dst = np.linspace(0.0, 1.0, N)
    return np.interp(dst, src, x)

def contiguous_regions(mask: np.ndarray) -> List[Tuple[int,int]]:
    # returns list of [start, end) half-open intervals
    if mask.size == 0: return []
    diffs = np.diff(mask.astype(np.int8))
    starts = list(np.where(diffs == 1)[0] + 1)
    ends   = list(np.where(diffs == -1)[0] + 1)
    if mask[0]:  starts = [0] + starts
    if mask[-1]: ends   = ends + [mask.size]
    return list(zip(starts, ends))

def save_per_dataset_outputs(
    out_dir: Path,
    label: str,
    t: np.ndarray,
    axis: np.ndarray,
    dt_s: Optional[float],
    nF: int,
    nR: int,
    kept_idx: List[int],
    dropped_R: List[int],
    dropped_F: List[int],
    meta_any: Dict,
    # new: save normalized abs(t)
    save_abs_t_norm: Optional[np.ndarray] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # leakage windows
    thr = float(TVLA_THRESHOLD)
    mask = np.abs(t) >= thr
    windows = contiguous_regions(mask)

    # text summary
    over = np.where(mask)[0]
    msg_lines = [
        f"TVLA summary for {label}",
        f"Samples: {t.shape[0]}",
        f"Kept pairs: R={nR}, F={nF}, indices={len(kept_idx)}",
        f"Dropped R: {len(dropped_R)}  Dropped F: {len(dropped_F)}",
        f"max |t|: {np.max(np.abs(t)):.3f} at sample {int(np.argmax(np.abs(t)))}",
        f"count(|t| >= {thr}): {over.size}",
        f"contiguous windows above threshold: {len(windows)}",
    ]
    if dt_s is not None:
        t_idx = int(np.argmax(np.abs(t)))
        msg_lines.append(f"max |t| time ~ {axis[t_idx]:.6e} s")
    if meta_any.get("timebase") is not None:
        msg_lines.append(f"timebase: {meta_any['timebase']}")
    if meta_any.get("sr_nominal"):
        msg_lines.append(f"sr_nominal: {meta_any['sr_nominal']:.3f} Hz")
    if meta_any.get("dt_ns"):
        msg_lines.append(f"dt_ns: {meta_any['dt_ns']:.3f} ns")

    (out_dir / "tvla_summary.txt").write_text("\n".join(msg_lines), encoding="utf-8")

    # windows CSV
    with open(out_dir / "tvla_leakage_windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["start_idx","end_idx","length_samples","peak_abs_t","peak_at_idx"]
        if dt_s is not None:
            cols += ["start_time_s","end_time_s","length_s","peak_time_s"]
        w.writerow(cols)
        for (s,e) in windows:
            seg = np.abs(t[s:e])
            peak_i = s + int(np.argmax(seg))
            row = [s, e, e-s, float(seg.max()), peak_i]
            if dt_s is not None:
                row += [axis[s], axis[e-1], (e-s)*dt_s, axis[peak_i]]
            w.writerow(row)

    # kept/dropped CSVs
    with open(out_dir / "tvla_indices.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kept_index"]); [w.writerow([k]) for k in kept_idx]
    with open(out_dir / "tvla_dropped_R.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dropped_R_index"]); [w.writerow([k]) for k in dropped_R]
    with open(out_dir / "tvla_dropped_F.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dropped_F_index"]); [w.writerow([k]) for k in dropped_F]

    # raw arrays
    np.save(out_dir / "t_statistic.npy", t)
    np.save(out_dir / ("time_axis.npy" if dt_s else "sample_axis.npy"), axis)

    # optional: per-dataset normalized |t|
    if save_abs_t_norm is not None:
        np.save(out_dir / "abs_t_normalized_1000.npy", save_abs_t_norm)

    # per-dataset plot
    plt.figure(figsize=(10, 4))
    xlab = "Time (s)" if dt_s else "Sample index"
    plt.plot(axis, t, label=f"t-stat ({label})", linewidth=1.0)
    plt.axhline(+TVLA_THRESHOLD, linestyle="--", linewidth=0.9, label=f"+{TVLA_THRESHOLD}")
    plt.axhline(-TVLA_THRESHOLD, linestyle="--", linewidth=0.9, label=f"-{TVLA_THRESHOLD}")
    plt.xlabel(xlab); plt.ylabel("t")
    plt.title(f"TVLA – {label}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"tvla_{label}.png", dpi=150)
    plt.close()

def main():
    # allow custom dataset dirs on CLI, else use defaults above
    ds_list = [Path(p) for p in (sys.argv[1:] if len(sys.argv) > 1 else DATASETS)]

    comparison_rows = []
    overlay_curves = []  # (label, |t| normalized 1000pts)
    overlay_ts      = []  # (label, t(raw), axis(raw), has_time)

    # NEW: comparison folder is centralized under EVAL_ROOT
    comp_dir = EVAL_ROOT / "COMPARISON"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for ds in ds_list:
        ds = ds.resolve()
        if not ds.exists():
            print(f"[skip] not found: {ds}")
            continue

        label  = dataset_label_from_path(ds)
        out_dir = make_out_dir_for_label(label)   # NEW: central per-dataset dir

        Rmap, Fmap = find_npz_files_by_group(ds)
        shared_idx = sorted(set(Rmap.keys()) & set(Fmap.keys()))
        if not shared_idx:
            print(f"[skip] no matching R/F pairs in {ds}")
            continue

        R_traces, F_traces = [], []
        meta_any = {}
        for k in shared_idx:
            wfR, metaR = load_npz_trace(Rmap[k])
            wfF, metaF = load_npz_trace(Fmap[k])
            R_traces.append(wfR); F_traces.append(wfF)
            if not meta_any: meta_any = {**metaR, **metaF}

        # filter flat traces in each group
        flags_R = robust_filter_flags(R_traces)
        flags_F = robust_filter_flags(F_traces)

        kept_idx, kept_R, kept_F, drop_R, drop_F = [], [], [], [], []
        for i, k in enumerate(shared_idx):
            if flags_R[i] and flags_F[i]:
                kept_idx.append(k); kept_R.append(R_traces[i]); kept_F.append(F_traces[i])
            else:
                if not flags_R[i]: drop_R.append(k)
                if not flags_F[i]: drop_F.append(k)

        if not kept_idx:
            print(f"[warn] all pairs filtered in {ds}")
            continue

        XR = np.stack([np.asarray(x, dtype=np.float32) for x in kept_R], axis=0)
        XF = np.stack([np.asarray(x, dtype=np.float32) for x in kept_F], axis=0)
        # remove per-trace DC
        XR = XR - XR.mean(axis=1, keepdims=True)
        XF = XF - XF.mean(axis=1, keepdims=True)
        # align length (+ ROI)
        XF, XR = align_length(XF, XR)

        # compute TVLA
        t_vec = compute_tvla(XF, XR)
        # axis (time/samples)
        axis, dt_s = time_axis_from_meta(t_vec.shape[0], meta_any)

        # normalized |t| (1000pts) for overlays & saving
        abs_t_norm = normalize_for_overlay(np.abs(t_vec), N=1000)

        # save per-dataset artifacts (+ normalized vector)
        save_per_dataset_outputs(
            out_dir=out_dir,
            label=label,
            t=t_vec,
            axis=axis,
            dt_s=dt_s,
            nF=XF.shape[0],
            nR=XR.shape[0],
            kept_idx=kept_idx,
            dropped_R=drop_R,
            dropped_F=drop_F,
            meta_any=meta_any,
            save_abs_t_norm=abs_t_norm,
        )

        # comparison row
        row = {
            "dataset": label,
            "dir": str(ds),
            "pairs_kept": len(kept_idx),
            "pairs_total_candidates": len(shared_idx),
            "drop_R": len(drop_R),
            "drop_F": len(drop_F),
            "max_abs_t": float(np.max(np.abs(t_vec))),
            "count_abs_t_ge_thr": int(np.sum(np.abs(t_vec) >= TVLA_THRESHOLD)),
        }
        comparison_rows.append(row)

        overlay_curves.append((label, abs_t_norm))
        overlay_ts.append((label, t_vec, axis, dt_s is not None))

    #  outputs 
    if comparison_rows:
        # overview CSV + JSON
        with open(comp_dir / "tvla_comparison_overview.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader(); w.writerows(comparison_rows)
        (comp_dir / "tvla_comparison_overview.json").write_text(
            json.dumps(comparison_rows, indent=2), encoding="utf-8"
        )

        # save all normalized |t| curves for deeper analysis later
        labels = [lbl for (lbl, _) in overlay_curves]
        curves = np.stack([y for (_, y) in overlay_curves], axis=0)  # (K, 1000)
        np.savez(comp_dir / "abs_t_normalized_all.npz",
                labels=np.array(labels, dtype=object),
                abs_t_norm=curves)

        # overlay (|t| vs normalized sample index)
        plt.figure(figsize=(10, 5))
        x = np.linspace(0.0, 1.0, 1000)
        for label, y in overlay_curves:
            plt.plot(x, y, label=label, linewidth=1.2)
        plt.axhline(TVLA_THRESHOLD, linestyle="--", linewidth=0.9, color="k", label=f"threshold {TVLA_THRESHOLD}")
        plt.xlabel("Normalized sample index"); plt.ylabel("|t|")
        plt.title("TVLA comparison (|t| overlay, normalized index)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / "tvla_overlay_abs_t_normalized.png", dpi=150)
        plt.close()

        # optional extra: overlay raw t vs sample index (truncate to min length)
        if overlay_ts:
            min_len = min(len(t) for (_, t, _, _) in overlay_ts)
            plt.figure(figsize=(10,5))
            for label, t, _, _ in overlay_ts:
                plt.plot(np.arange(min_len), t[:min_len], label=label, linewidth=1.0)
            plt.axhline(+TVLA_THRESHOLD, linestyle="--", linewidth=0.9, color="k")
            plt.axhline(-TVLA_THRESHOLD, linestyle="--", linewidth=0.9, color="k")
            plt.xlabel("Sample index"); plt.ylabel("t")
            plt.title("TVLA comparison (raw t, truncated to min length)")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(comp_dir / "tvla_overlay_t_minlen.png", dpi=150)
            plt.close()

        print(f"\nAll evaluation artifacts stored under:\n  {EVAL_ROOT}")
    else:
        print("\nNo datasets produced results.")

if __name__ == "__main__":
    main()
