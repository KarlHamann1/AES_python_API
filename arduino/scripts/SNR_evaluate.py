#!/usr/bin/env python3
"""
snr_evaluate.py

Compute per-sample SNR via variance decomposition (between/within) for one or more datasets.

SNR(t) = Var( E[X(t) | L] ) / E[ Var( X(t) | L ) ]
where L is a discrete "leakage class". By default:
    L = HW( SBOX[ PT[BYTE_INDEX] ^ KEY[BYTE_INDEX] ] )  # 9 classes

Outputs under arduino/data/SNR_eval:
    SNR_eval/
    1MHz/ or 8MHz/ or 16MHz/
        snr_curve.npy
        time_axis.npy  (or sample_axis.npy)
        snr_plot.png
        snr_summary.txt
        class_counts.csv
        snr_normalized_1000.npy
    COMPARISON/
        snr_overlay_normalized.png
        snr_overlay_minlen.png
        snr_comparison_overview.{csv,json}
        snr_normalized_all.npz
"""

from __future__ import annotations
import os, sys, csv, json, math, re, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

#  repo layout / default dataset roots
SCRIPT_DIR  = Path(__file__).resolve().parent                 
REPO_ROOT   = SCRIPT_DIR.parents[1] 
ARDUINO_DIR = REPO_ROOT / "arduino"                            
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "SNR_eval").resolve()

# My three full-AES averaged datasets (12-bit)
DATASETS = [
    DATA_ROOT / "data_arduino_1MHz_tb16_4.81Msps_9600Bd_avg100_10ms_20MHzBW_12bit_ACoff4mV",
    DATA_ROOT / "data_arduino_8MHz_tb5_31.25Msps_57600Bd_avg100_1.5ms_20MHzBW_12bit_ACoff2mV",
    DATA_ROOT / "data_arduino_16MHz_tb5_31Msps_115200Bd_avg100_0p7ms",
]
# label map (keeps folder names readable in plots)
LABEL_FOR_PATHPART = {"1MHz": "1MHz", "8MHz": "8MHz", "16MHz": "16MHz"}

#  SNR model settings
TRUE_KEY_HEX = "81556407E2B7C090D23146B46CD554A2"  # known Arduino key
BYTE_INDEX   = 0                                    # which state byte to use
MODEL        = "SBOX_HW"                            # "SBOX_HW" | "SBOX_VALUE" | "PT_VALUE"

# ROI (optional). Use one or None.
ROI_SAMPLES: Optional[Tuple[int, int]] = None       # e.g. (2000, 15000)
ROI_TIME_S:  Optional[Tuple[float, float]] = None   # e.g. (8e-6, 20e-6) if dt known

# Filtering
P2P_MIN_FRACTION = 0.20  # drop if p2p < 20% of median p2p
STD_MIN_FRACTION = 0.15  # drop if std < 15% of median std

# Threshold for summary
SNR_THRESHOLD = 0.01  # purely for reporting area/percent over threshold

#  AES helpers
HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
SBOX = np.array([
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
], dtype=np.uint8)

#  utils
def dataset_label_from_path(p: Path) -> str:
    name = p.name
    for part, lab in LABEL_FOR_PATHPART.items():
        if part in name:
            return lab
    # fallback: try "...MHz" in name
    m = re.search(r"(\d+)\s*MHz", name, re.IGNORECASE)
    return f"{m.group(1)}MHz" if m else name

def time_axis_from_meta(n: int, dt_ns: Optional[float], sr_nominal: Optional[float]) -> Tuple[np.ndarray, Optional[float]]:
    dt_s = None
    if dt_ns and dt_ns > 0:
        dt_s = dt_ns * 1e-9
    elif sr_nominal and sr_nominal > 0:
        dt_s = 1.0 / sr_nominal
    if dt_s:
        return np.arange(n, dtype=np.float64) * dt_s, dt_s
    return np.arange(n, dtype=np.float64), None

def robust_filter_flags(traces: List[np.ndarray]) -> np.ndarray:
    if not traces: return np.array([], dtype=bool)
    p2p = np.array([float(t.max() - t.min()) for t in traces])
    s   = np.array([float(np.std(t)) for t in traces])
    medp = np.median(p2p); meds = np.median(s)
    ok1 = p2p >= (P2P_MIN_FRACTION * max(medp, 1e-12))
    ok2 = s   >= (STD_MIN_FRACTION * max(meds, 1e-12))
    return ok1 & ok2

def _hex_to_bytes16(h: str) -> bytes:
    h = h.strip().replace(" ", "")
    return bytes.fromhex(h)

def load_rows_from_csv(ds_dir: Path) -> List[Dict[str, str]]:
    # support 'trace_overview.csv' (from your averaging capture)
    csv_path = ds_dir / "trace_overview.csv"
    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    else:
        # fallback: glob npz/npy and synthesize rows
        for p in sorted(ds_dir.glob("*.npz")):
            rows.append({"FileName": p.name, "PlaintextHex": "", "CiphertextHex": ""})
        for p in sorted(ds_dir.glob("*.npy")):
            rows.append({"TraceFilePath": p.name, "PlaintextHex": "", "CiphertextHex": ""})
    return rows

def open_trace_any(path: Path):
    """
    Load either NPZ (keys: trace_mean or trace) or NPY (1D).
    Return waveform (float32), dt_ns (float|None), sr_nominal (float|None), plaintext bytes (or None).
    """
    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        if "trace_mean" in npz.files:
            wf = np.asarray(npz["trace_mean"], dtype=np.float32)
        elif "trace" in npz.files:
            wf = np.asarray(npz["trace"], dtype=np.float32)
        else:
            raise ValueError(f"{path.name}: no 'trace_mean'/'trace'.")
        dt_ns = float(npz["dt_ns"].item()) if "dt_ns" in npz.files else None
        sr_nominal = float(npz["sr_nominal"].item()) if "sr_nominal" in npz.files else (float(npz["sr"].item()) if "sr" in npz.files else None)
        pt = bytes(npz["plaintext"].astype(np.uint8).tolist()) if "plaintext" in npz.files else None
        return wf, dt_ns, sr_nominal, pt
    elif path.suffix.lower() == ".npy":
        wf = np.load(path).astype(np.float32, copy=False)
        return wf, None, None, None
    else:
        raise ValueError(f"Unsupported file: {path}")

def class_value(pt: bytes, key: bytes, model: str, byte_index: int) -> int:
    b = pt[byte_index] ^ key[byte_index]
    s = SBOX[b]
    if model.upper() == "SBOX_HW":
        return int(HW[s])
    elif model.upper() == "SBOX_VALUE":
        return int(s)
    elif model.upper() == "PT_VALUE":
        return int(pt[byte_index])
    else:
        raise ValueError(f"Unknown MODEL: {model}")

def normalize_for_overlay(y: np.ndarray) -> np.ndarray:
    N = y.shape[0]
    if N == 1000: return y.copy()
    xsrc = np.linspace(0.0, 1.0, N)
    xdst = np.linspace(0.0, 1.0, 1000)
    return np.interp(xdst, xsrc, y)

# SNR
def compute_snr_streaming(
    ds_dir: Path,
    key_bytes: bytes,
    model: str = MODEL,
    byte_index: int = BYTE_INDEX
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Stream traces and accumulate class-wise sums/sumsq.
    Returns:
    snr_curve (float64[T]),
    meta_any (dict with dt_ns/sr_nominal/timebase if found),
    class_counts (int[K])  (K=9 for SBOX_HW, else 256)
    """
    rows = load_rows_from_csv(ds_dir)
    if not rows:
        raise RuntimeError(f"No files/CSV in {ds_dir}")

    # detect K based on model
    K = 9 if model.upper() == "SBOX_HW" else 256

    # first pass: figure out length T and collect light metadata
    T = None
    meta_any = {"dt_ns": None, "sr_nominal": None, "timebase": None}
    waveforms = []   # I keep only for a quick dud filter; then stream again
    classes   = []
    kept_files = []

    for r in rows:
        fn = r.get("FileName") or r.get("TraceFilePath")
        if not fn: continue
        p = ds_dir / fn
        if not p.exists():  # sometimes CSV sits in dir above
            alt = (ds_dir / fn).resolve()
            if not alt.exists():
                # try absolute if CSV already had a path
                p = Path(fn)
        if not p.exists():
            # silently skip missing
            continue

        wf, dt_ns, sr_nom, pt_npz = open_trace_any(p)

        # stash dt/sr/timebase if present in CSV
        if meta_any["dt_ns"] is None and dt_ns: meta_any["dt_ns"] = dt_ns
        if meta_any["sr_nominal"] is None and sr_nom: meta_any["sr_nominal"] = sr_nom
        if meta_any["timebase"] is None and "timebase" in r:
            try: meta_any["timebase"] = int(r["timebase"])
            except Exception: pass

        if T is None: T = int(wf.shape[0])
        else:         T = min(T, int(wf.shape[0]))  # keep min length across set

        # plaintext: prefer CSV else NPZ field
        pt_hex = (r.get("PlaintextHex") or "").strip()
        pt = _hex_to_bytes16(pt_hex) if len(pt_hex) == 32 else pt_npz
        if pt is None or len(pt) != 16:
            # can't classify without plaintext
            continue

        waveforms.append(wf)
        classes.append(class_value(pt, key_bytes, model, byte_index))
        kept_files.append(p)

    if not waveforms:
        raise RuntimeError(f"No usable traces in {ds_dir} (missing PTs?)")

    # cut everything to min length T and dud-filter a bit
    waveforms = [wf[:T] for wf in waveforms]
    flags = robust_filter_flags(waveforms)
    waveforms = [wf for wf, ok in zip(waveforms, flags) if ok]
    classes   = [c  for c, ok  in zip(classes, flags) if ok]
    if not waveforms:
        raise RuntimeError(f"All traces filtered as duds in {ds_dir}")

    # streaming accumulators
    sum_x  = np.zeros((K, T), dtype=np.float64)
    sum_x2 = np.zeros((K, T), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    # stream again (using the filtered list I already loaded)
    for wf, c in zip(waveforms, classes):
        sum_x[c]  += wf
        sum_x2[c] += wf * wf
        counts[c] += 1

    # compute class means/vars
    with np.errstate(divide="ignore", invalid="ignore"):
        m_k = np.zeros((K, T), dtype=np.float64)
        v_k = np.zeros((K, T), dtype=np.float64)
        for k in range(K):
            n = counts[k]
            if n == 0: continue
            mk = sum_x[k] / float(n)
            # sample variance, guarded
            var_pop = (sum_x2[k] / float(n)) - mk*mk
            # debias a hair if enough samples
            if n > 1:
                var = var_pop * (n / (n - 1.0))
            else:
                var = np.zeros_like(var_pop)
            m_k[k] = mk
            v_k[k] = var

        p_k = counts / max(1, int(np.sum(counts)))
        m   = (p_k[:, None] * m_k).sum(axis=0)
        between = (p_k[:, None] * (m_k - m)**2).sum(axis=0)
        within  = (p_k[:, None] * v_k).sum(axis=0)
        snr = np.zeros(T, dtype=np.float64)
        mask = within > 1e-20
        snr[mask] = between[mask] / within[mask]

    meta_any["counts_nonzero"] = int(np.count_nonzero(counts))
    return snr, meta_any, counts

def apply_roi(y: np.ndarray, axis: np.ndarray, dt_s: Optional[float]):
    if ROI_SAMPLES is not None:
        s0, s1 = map(int, ROI_SAMPLES)
        s0 = max(0, min(s0, len(y)))
        s1 = max(s0, min(s1, len(y)))
        return y[s0:s1], axis[s0:s1], dt_s
    if ROI_TIME_S is not None and dt_s:
        t0, t1 = ROI_TIME_S
        s0 = max(0, int(round(t0 / dt_s)))
        s1 = min(len(y), int(round(t1 / dt_s)))
        return y[s0:s1], axis[s0:s1], dt_s
    return y, axis, dt_s

def save_per_dataset_outputs(
    out_dir: Path,
    label: str,
    snr: np.ndarray,
    axis: np.ndarray,
    dt_s: Optional[float],
    counts: np.ndarray
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROI view for summaries
    y_roi, x_roi, dt_roi = apply_roi(snr, axis, dt_s)

    # metrics
    peak_val = float(np.max(y_roi)) if y_roi.size else float(np.max(snr))
    peak_idx = int(np.argmax(y_roi)) if y_roi.size else int(np.argmax(snr))
    peak_time = (x_roi[peak_idx] if (y_roi.size and dt_roi) else (axis[int(np.argmax(snr))] if dt_s else None))
    avg_roi = float(np.mean(y_roi)) if y_roi.size else float(np.mean(snr))
    over = (y_roi >= SNR_THRESHOLD) if y_roi.size else (snr >= SNR_THRESHOLD)
    pct_over = 100.0 * float(np.count_nonzero(over)) / float(len(y_roi) if y_roi.size else len(snr))
    area_over = None
    if dt_roi:
        area_over = float(np.sum(over) * dt_roi)

    # save arrays
    np.save(out_dir / "snr_curve.npy", snr)
    np.save(out_dir / ("time_axis.npy" if dt_s else "sample_axis.npy"), axis)
    np.save(out_dir / "snr_normalized_1000.npy", normalize_for_overlay(snr))

    # class counts
    with open(out_dir / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class_value", "count"])
        for k, n in enumerate(counts.tolist()):
            if n > 0: w.writerow([k, n])

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(axis, snr, label=f"SNR ({label})", linewidth=1.1)
    if dt_s:
        plt.xlabel("Time (s)")
    else:
        plt.xlabel("Sample index")
    plt.axhline(SNR_THRESHOLD, linestyle="--", linewidth=0.9, label=f"thr={SNR_THRESHOLD:.3g}")
    plt.ylabel("SNR")
    plt.title(f"SNR – {label}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"snr_{label}.png", dpi=150)
    plt.close()

    # summary.txt
    lines = [
        f"SNR summary for {label}",
        f"samples: {len(snr)}",
        f"SNR_THRESHOLD: {SNR_THRESHOLD}",
        f"peak SNR (ROI if set): {peak_val:.4f}",
        f"avg SNR (ROI if set):  {avg_roi:.4f}",
        f"% samples >= thr:      {pct_over:.2f}%",
    ]
    if peak_time is not None:
        lines.append(f"peak time ~ {peak_time:.6e} s")
    if area_over is not None:
        lines.append(f"area over thr (s):  {area_over:.6e}")
    (out_dir / "snr_summary.txt").write_text("\n".join(lines), encoding="utf-8")

def main():
    key = _hex_to_bytes16(TRUE_KEY_HEX)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    comp_rows = []
    overlay_curves = []
    minlen_curves = []

    for ds in DATASETS:
        ds = ds.resolve()
        if not ds.exists():
            print(f"[skip] dataset not found: {ds}")
            continue

        label = dataset_label_from_path(ds)
        out_dir = EVAL_ROOT / label

        try:
            snr, meta_any, counts = compute_snr_streaming(ds, key, MODEL, BYTE_INDEX)
        except Exception as e:
            print(f"[warn] {label}: {e}")
            continue

        # axis
        axis, dt_s = time_axis_from_meta(len(snr), meta_any.get("dt_ns"), meta_any.get("sr_nominal"))

        # per-dataset artifacts
        save_per_dataset_outputs(out_dir, label, snr, axis, dt_s, counts)

        # comparison row
        y_roi, x_roi, dt_roi = apply_roi(snr, axis, dt_s)
        over = (y_roi >= SNR_THRESHOLD) if y_roi.size else (snr >= SNR_THRESHOLD)
        pct_over = 100.0 * float(np.count_nonzero(over)) / float(len(y_roi) if y_roi.size else len(snr))
        row = {
            "dataset": label,
            "dir": str(ds),
            "trace_count": int(np.sum(counts)),
            "class_nonzero": int(np.count_nonzero(counts)),
            "peak_snr": float(np.max(y_roi) if y_roi.size else np.max(snr)),
            "avg_snr_roi_or_all": float(np.mean(y_roi) if y_roi.size else np.mean(snr)),
            "pct_samples_over_thr": float(pct_over),
        }
        if dt_roi:
            row["area_over_thr_seconds"] = float(np.sum(over) * dt_roi)
        comp_rows.append(row)

        # curves for overlay
        overlay_curves.append((label, normalize_for_overlay(snr)))
        minlen_curves.append((label, snr))

    # comparison bundle
    comp_dir = EVAL_ROOT / "COMPARISON"
    comp_dir.mkdir(parents=True, exist_ok=True)

    if comp_rows:
        # CSV + JSON
        with open(comp_dir / "snr_comparison_overview.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
            w.writeheader(); w.writerows(comp_rows)
        (comp_dir / "snr_comparison_overview.json").write_text(json.dumps(comp_rows, indent=2), encoding="utf-8")

        # overlay (normalized length)
        plt.figure(figsize=(10, 5))
        x = np.linspace(0.0, 1.0, 1000)
        for label, y in overlay_curves:
            plt.plot(x, y, label=label, linewidth=1.2)
        plt.axhline(SNR_THRESHOLD, linestyle="--", linewidth=0.9, color="k", label=f"thr={SNR_THRESHOLD:.3g}")
        plt.xlabel("Normalized sample index"); plt.ylabel("SNR")
        plt.title("SNR comparison (normalized length)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / "snr_overlay_normalized.png", dpi=150)
        plt.close()

        # overlay (truncate to min length)
        if minlen_curves:
            minL = min(len(y) for _, y in minlen_curves)
            plt.figure(figsize=(10, 5))
            for label, y in minlen_curves:
                plt.plot(np.arange(minL), y[:minL], label=label, linewidth=1.1)
            plt.axhline(SNR_THRESHOLD, linestyle="--", linewidth=0.9, color="k", label=f"thr={SNR_THRESHOLD:.3g}")
            plt.xlabel("Sample index (truncated)"); plt.ylabel("SNR")
            plt.title("SNR comparison (min length overlay)")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(comp_dir / "snr_overlay_minlen.png", dpi=150)
            plt.close()

        np.savez(comp_dir / "snr_normalized_all.npz", **{lab: y for lab, y in overlay_curves})

        print(f"\nSNR eval saved to: {EVAL_ROOT}")
    else:
        print("Nothing to compare — no datasets produced results.")

if __name__ == "__main__":
    main()
