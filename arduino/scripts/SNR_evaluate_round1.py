#!/usr/bin/env python3
"""
snr_evaluate_first10pct.py

Compute model-based SNR per sample using the *first 10%* of each dataset’s trace length
(as a quick, consistent ROI across datasets that contain ~10 AES rounds).

What it does (per dataset: 1MHz / 8MHz / 16MHz):
1) Loads averaged-trace NPZ files (and also supports NPY if present).
    - NPZ keys handled: "trace_mean" or "trace"
    - Optional metadata: "dt_ns", "sr_nominal", "sr", "timebase"
    - Plaintext read from "plaintext" (uint8[16])
2) Drops obviously flat/bad traces via a simple amplitude heuristic (peak-to-peak + std).
  3) Stacks kept traces, crops to the min length across kept traces, then takes the *first 10%*.
4) Computes SNR per sample using variance decomposition over 9 Hamming-weight classes:
        class = HW( SBOX[ PT[BYTE_IDX] ^ TRUE_KEY[BYTE_IDX] ] )  in {0..8}
    SNR(t) = Var_c( E[X|c] )(t)  /  E_c( Var[X|c] )(t)
5) Saves per-dataset outputs under:  <repo>/arduino/data/SNR_eval/{1MHz,8MHz,16MHz}/
    - snr_curve.npy
    - time_axis.npy or sample_axis.npy
    - snr_plot.png
    - snr_summary.txt
    - kept_indices.csv, dropped_indices.csv
    - roi_info.json  (s0/s1/frac)
6) Also writes a comparison folder:  <repo>/arduino/data/SNR_eval/COMPARISON
    - snr_comparison_overview.csv / .json
    - snr_overlay_normalized.png   (SNR curves resampled to 1000 points)
    - abs_snr_normalized_all.npz   (handy for future analysis)

Notes:
- Uses only the first 10% of samples to approximate the first-round region across datasets.
- Baseline-removes traces by per-trace mean subtraction to avoid DC offset skew.
- No TVLA files required.

"""

from __future__ import annotations
import os, sys, json, csv, math, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Project paths 
SCRIPT_DIR  = Path(__file__).resolve().parent             
REPO_ROOT   = SCRIPT_DIR.parents[1]                          
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()

#  Datasets (12-bit, full AES) 
DATASETS = [
    DATA_ROOT / "data_arduino_1MHz_tb16_4.81Msps_9600Bd_avg100_10ms_20MHzBW_12bit_ACoff4mV",
    DATA_ROOT / "data_arduino_8MHz_tb5_31.25Msps_57600Bd_avg100_1.5ms_20MHzBW_12bit_ACoff2mV",
    DATA_ROOT / "data_arduino_16MHz_tb5_31Msps_115200Bd_avg100_0p7ms",
]
# If your 8MHz folder name is slightly different, fix the entry above to your exact directory.

#  Output root 
SNR_EVAL_ROOT = (DATA_ROOT / "SNR_eval_round1").resolve()

ROI_FRACTION = 0.10  # percentage of samples to use

TRUE_KEY_HEX = "81556407E2B7C090D23146B46CD554A2"  # set to "" if unknown
BYTE_IDX     = 0   # model byte index (0..15)

# S-Box + HW
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
HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

# Filtering
P2P_MIN_FRACTION = 0.20  # drop if p2p < 20% of median p2p
STD_MIN_FRACTION = 0.15  # drop if std < 15% of median std

# Threshold for summary
SNR_THRESH = 0.01  # purely for reporting area/percent over threshold


def dataset_label_from_path(p: Path) -> str:
    name = p.name
    for tok in name.split("_"):
        if tok.endswith("MHz"):
            return tok
    return name

def load_npz_or_npy(path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load waveform + meta from .npz or .npy.
    Returns (waveform float32, meta dict with keys dt_s, sr_hz, timebase, plaintext_bytes).
    """
    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        if "trace_mean" in npz.files:
            wf = np.asarray(npz["trace_mean"], dtype=np.float32)
        elif "trace" in npz.files:
            wf = np.asarray(npz["trace"], dtype=np.float32)
        else:
            raise ValueError(f"{path.name}: no 'trace_mean' or 'trace' in NPZ.")
        # metadata
        dt_ns = float(npz["dt_ns"].item()) if "dt_ns" in npz.files else None
        sr_nom = float(npz["sr_nominal"].item()) if "sr_nominal" in npz.files else None
        sr     = float(npz["sr"].item()) if "sr" in npz.files else None
        timeb  = int(npz["timebase"].item()) if "timebase" in npz.files else None
        if "plaintext" in npz.files:
            pt_bytes = bytes(npz["plaintext"].astype(np.uint8).tolist())
        else:
            pt_bytes = None

        if dt_ns and dt_ns > 0:
            dt_s = dt_ns * 1e-9
            sr_hz = 1.0 / dt_s
        elif sr and sr > 0:
            sr_hz = sr; dt_s = 1.0 / sr_hz
        elif sr_nom and sr_nom > 0:
            sr_hz = sr_nom; dt_s = 1.0 / sr_hz
        else:
            dt_s = None; sr_hz = None

        return wf, {"dt_s": dt_s, "sr_hz": sr_hz, "timebase": timeb, "plaintext": pt_bytes}

    elif path.suffix.lower() == ".npy":
        wf = np.load(path).astype(np.float32, copy=False)
        return wf, {"dt_s": None, "sr_hz": None, "timebase": None, "plaintext": None}

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def robust_filter_flags(traces: List[np.ndarray]) -> np.ndarray:
    """Flag traces that look dead/flat based on p2p and std."""
    p2p = np.array([float(np.max(t) - np.min(t)) for t in traces], dtype=np.float64)
    sd  = np.array([float(np.std(t))            for t in traces], dtype=np.float64)
    med_p2p = np.median(p2p) if np.isfinite(p2p).all() else 0.0
    med_sd  = np.median(sd)  if np.isfinite(sd).all()  else 0.0
    ok_p2p = p2p >= (P2P_MIN_FRACTION * max(med_p2p, 1e-12))
    ok_sd  = sd  >= (STD_MIN_FRACTION  * max(med_sd,  1e-12))
    return ok_p2p & ok_sd

def normalize_for_overlay(x: np.ndarray, N: int = 1000) -> np.ndarray:
    if x.shape[0] == N:
        return x.copy()
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, N)
    return np.interp(dst, src, x)

def compute_snr_matrix(X: np.ndarray, labels: np.ndarray, n_classes: int = 9) -> np.ndarray:
    """
    X: (N, L) float32 after ROI + baseline removal (per-trace mean subtracted)
    labels: (N,) in {0..n_classes-1}
    Returns SNR per sample (L,).
    """
    N, L = X.shape
    snr_num = np.zeros(L, dtype=np.float64)  # between-class variance
    snr_den = np.zeros(L, dtype=np.float64)  # within-class variance (expected)

    # compute per-class means and variances
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        Xc = X[idx, :]  # (Nc, L)
        mu_c = Xc.mean(axis=0)                     # (L,)
        var_c = Xc.var(axis=0, ddof=1) if Xc.shape[0] > 1 else np.zeros(L)

        # weight by class prior p(c) ~ Nc/N
        w = idx.size / float(N)
        snr_num += w * (mu_c ** 2)   # Var_c[mu_c] around 0 because traces already centered globally per-trace
        snr_den += w * var_c

    # A more standard form: Var_c(mu_c) / E_c[var_c]
    # To make numerator exactly Var_c(mu_c), need to subtract the (weighted) mean of mu_c over classes.
    # However, since each trace was DC-removed (per-trace), and classes cover PT space well, the mean(mu_c) is ~0.
    # For completeness, compute weighted mean of mu_c and adjust:
    # Recompute in two passes to get exact Var_c(mu_c).
    mu_acc = np.zeros(L, dtype=np.float64)
    w_tot  = 0.0
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        Xc = X[idx, :]
        mu_c = Xc.mean(axis=0)
        w = idx.size / float(N)
        mu_acc += w * mu_c
        w_tot  += w
    mu_bar = mu_acc / max(w_tot, 1e-12)

    # Now recompute numerator as weighted sum of squared deviations from mu_bar
    snr_num = np.zeros(L, dtype=np.float64)
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        Xc = X[idx, :]
        mu_c = Xc.mean(axis=0)
        w = idx.size / float(N)
        snr_num += w * ((mu_c - mu_bar) ** 2)

    snr = np.zeros(L, dtype=np.float64)
    mask = snr_den > 1e-20
    snr[mask] = snr_num[mask] / snr_den[mask]
    return snr


def main():
    SNR_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    overlay_curves  = []

    # Build key bytes
    key_bytes = bytes.fromhex(TRUE_KEY_HEX) if len(TRUE_KEY_HEX) == 32 else None

    # Comparison dir
    comp_dir = SNR_EVAL_ROOT / f"COMPARISON"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for ds_dir in DATASETS:
        ds_dir = ds_dir.resolve()
        if not ds_dir.exists():
            print(f"[skip] dataset not found: {ds_dir}")
            continue

        label = dataset_label_from_path(ds_dir)
        out_dir = SNR_EVAL_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect files
        files = sorted(list(ds_dir.glob("*.npz")) + list(ds_dir.glob("*.npy")))
        if not files:
            print(f"[skip] no NPZ/NPY in: {ds_dir}")
            continue

        # Load all traces + metadata
        wf_list: List[np.ndarray] = []
        pts: List[bytes] = []
        kept_file_indices: List[int] = []
        file_indices: List[int] = []  # parsed from filename if available, else running index

        for i, fpath in enumerate(files):
            try:
                wf, meta = load_npz_or_npy(fpath)
            except Exception as e:
                print(f"[warn] load failed {fpath.name}: {e}")
                continue

            wf_list.append(wf.astype(np.float32, copy=False))
            pts.append(meta.get("plaintext", None))
            # Try to parse a numeric tail (e.g., *_000123.npz)
            stem = fpath.stem
            tail = "".join([ch for ch in stem if ch.isdigit()])
            file_indices.append(int(tail) if tail else i)

        if not wf_list:
            print(f"[skip] nothing valid in: {ds_dir}")
            continue

        # Filter out flat traces
        flags = robust_filter_flags(wf_list)
        wf_list = [w for w, ok in zip(wf_list, flags) if ok]
        pts     = [p for p, ok in zip(pts, flags) if ok]
        kept_file_indices = [idx for idx, ok in zip(file_indices, flags) if ok]
        dropped_indices   = [idx for idx, ok in zip(file_indices, flags) if not ok]

        if len(wf_list) == 0:
            print(f"[warn] all traces dropped in {ds_dir}")
            continue

        # Align lengths and stack
        min_len = min(len(w) for w in wf_list)
        X = np.stack([w[:min_len] for w in wf_list], axis=0).astype(np.float32)  # (N, L)
        N, L = X.shape

        # ROI
        s1 = max(1, int(math.floor(ROI_FRACTION * L)))
        s0 = 0
        X = X[:, s0:s1]
        L = X.shape[1]

        # Time axis (if possible)
        # Try to estimate dt from any NPZ that had dt_s or sr; otherwise index
        dt_s_any = None
        for fpath in files:
            if fpath.suffix.lower() == ".npz":
                try:
                    npz = np.load(fpath, allow_pickle=False)
                    if "dt_ns" in npz.files:
                        dt_ns = float(npz["dt_ns"].item())
                        if dt_ns > 0:
                            dt_s_any = dt_ns * 1e-9
                            break
                    if "sr_nominal" in npz.files:
                        sr = float(npz["sr_nominal"].item())
                        if sr > 0:
                            dt_s_any = 1.0 / sr
                            break
                    if "sr" in npz.files:
                        sr = float(npz["sr"].item())
                        if sr > 0:
                            dt_s_any = 1.0 / sr
                            break
                except Exception:
                    pass
        if dt_s_any:
            axis = np.arange(L, dtype=np.float64) * dt_s_any
            axis_name = "time_axis.npy"
        else:
            axis = np.arange(L, dtype=np.float64)
            axis_name = "sample_axis.npy"

        # Baseline removal (per-trace mean)
        X = X - X.mean(axis=1, keepdims=True)

        # Build labels (9 HW classes) from plaintext + true key
        if key_bytes is None or any(p is None or len(p) != 16 for p in pts):
            print(f"[warn] missing key or plaintext; cannot compute SNR for {label}")
            continue
        key_b = key_bytes[BYTE_IDX]
        pt_first = np.array([p[BYTE_IDX] for p in pts], dtype=np.uint8)
        z = SBOX[pt_first ^ key_b]              # intermediate byte values
        labels = HW[z]                           # classes 0..8
        K = 9

        # Compute SNR
        snr = compute_snr_matrix(X, labels, n_classes=K)

        # Metrics over the ROI (which is the whole X here)
        peak_snr = float(np.max(snr))
        avg_snr  = float(np.mean(snr))
        pct_over = float(np.mean(snr >= SNR_THRESH))
        area_over = float(np.sum(snr >= SNR_THRESH) * (dt_s_any if dt_s_any else 1.0))

        # --- Save per-dataset artifacts ---
        np.save(out_dir / "snr_curve.npy", snr)
        np.save(out_dir / axis_name, axis)

        with open(out_dir / "snr_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                "\n".join([
                    f"SNR summary for {label}",
                    f"traces_kept: {N} (dropped: {len(dropped_indices)})",
                    f"ROI samples: [{s0}, {s1})  (first {int(ROI_FRACTION*100)}%)",
                    f"peak SNR: {peak_snr:.6f}",
                    f"avg SNR (ROI): {avg_snr:.6f}",
                    f"percent samples >= {SNR_THRESH:.4f}: {pct_over*100:.3f}%",
                    (f"area over threshold (seconds): {area_over:.6e}"
                    if dt_s_any else f"area over threshold (samples): {area_over:.0f}"),
                ])
            )

        with open(out_dir / "kept_indices.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["kept_index"])
            for k in kept_file_indices: w.writerow([k])
        with open(out_dir / "dropped_indices.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["dropped_index"])
            for k in dropped_indices:   w.writerow([k])

        (out_dir / "roi_info.json").write_text(
            json.dumps({"s0": s0, "s1": s1, "fraction": ROI_FRACTION}, indent=2),
            encoding="utf-8"
        )

        # Plot per dataset
        plt.figure(figsize=(10, 4))
        if axis_name == "time_axis.npy":
            plt.plot(axis, snr, label=f"{label} SNR")
            plt.xlabel("Time (s)")
        else:
            plt.plot(axis, snr, label=f"{label} SNR")
            plt.xlabel("Sample index")
        plt.ylabel("SNR")
        plt.title(f"SNR – First {int(ROI_FRACTION*100)}% ROI – {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"snr_{label}.png", dpi=150)
        plt.close()

        # Comparison rows / overlay
        comparison_rows.append({
            "dataset": label,
            "dir": str(ds_dir),
            "traces_kept": N,
            "peak_snr": peak_snr,
            "avg_snr": avg_snr,
            "pct_over_thresh": pct_over,
            "area_over_thresh_" + ("seconds" if dt_s_any else "samples"): area_over
        })
        overlay_curves.append((label, normalize_for_overlay(snr)))

    #  COMPARISON OUTPUTS 
    if comparison_rows:
        # Save CSV/JSON
        fields = list(comparison_rows[0].keys())
        with open(comp_dir / "snr_comparison_overview.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(comparison_rows)
        (comp_dir / "snr_comparison_overview.json").write_text(
            json.dumps(comparison_rows, indent=2), encoding="utf-8"
        )

        # Overlay plot (normalized sample axis)
        plt.figure(figsize=(10, 5))
        x = np.linspace(0.0, 1.0, 1000)
        for label, y in overlay_curves:
            plt.plot(x, y, label=label, linewidth=1.2)
        plt.xlabel("Normalized sample index (first 10% ROI)")
        plt.ylabel("SNR")
        plt.title("SNR comparison (normalized overlay)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / "snr_overlay_normalized.png", dpi=150)
        plt.close()

        # Save the normalized curves for future analysis
        np.savez(comp_dir / "abs_snr_normalized_all.npz", **{lab: y for lab, y in overlay_curves})

        print(f"\nSNR evaluation done. Results in: {SNR_EVAL_ROOT}")
    else:
        print("\nNo SNR results produced (check dataset paths and content).")

if __name__ == "__main__":
    main()
