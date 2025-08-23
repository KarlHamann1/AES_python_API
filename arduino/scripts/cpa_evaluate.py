#!/usr/bin/env python3
"""
cpa_evaluate.py

CPA success curves (GE/SR) on averaged AES traces stored as NPZ (trace_mean),
matching the style of your tvla_evaluate.py / snr_evaluate.py.

Inputs (per dataset under arduino/data):
- trace_overview.csv with columns:
    BlockIndex, PlaintextHex, CiphertextHex, FileName, Samples, n_avg, dt_ns, (optional) timebase
- NPZ files referenced by FileName, each containing:
    trace_mean or trace, plaintext, ciphertext, and optional dt_ns, sr_nominal, timebase, n_avg

Leakage model: first-round SubBytes, HW(SBOX(PT[byte] ^ guess))

Saves under arduino/data/CPA_eval/<Label>/ :
recovered_key_hex.txt
byte_bXX/
    Ns.npy, GE.npy, SR.npy
    GE_vs_traces_bXX.png (log-log)
    SR_vs_traces_bXX.png (log-x)
    GE_SR.json (curves + summary)
    summary_bXX.txt (traces to GE<=1, SR>=80/90%, best |corr| using all traces)
COMPARISON/ (across datasets):
GE_overlay_bXX.png, SR_overlay_bXX.png
cpa_comparison_bXX.csv / .json

Notes
- Bootstraps without replacement; default 50 resamples per grid point.
- ROI can be samples, absolute time, or fraction of trace length (e.g., 0.10 for “first 10%”).
- Dud filtering (p2p/std) matches your other tools.
"""

from __future__ import annotations
import os, sys, re, csv, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt



SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "CPA_eval").resolve()

"""
DATASETS = [
    DATA_ROOT / "data_arduino_1MHz_tb16_4.81Msps_9600Bd_avg100_10ms_20MHzBW_12bit_ACoff4mV",
    DATA_ROOT / "data_arduino_8MHz_tb5_31.25Msps_57600Bd_avg100_1.5ms_20MHzBW_12bit_ACoff2mV",
    DATA_ROOT / "data_arduino_16MHz_tb5_31Msps_115200Bd_avg100_0p7ms",
]
"""

DATASETS = [
    DATA_ROOT / "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV",
]

# Short display labels
LABEL_FOR_PATHPART = {"1MHz": "1MHz", "8MHz": "8MHz", "16MHz": "16MHz"}

# CPA settings 
TRUE_KEY_HEX   = "81556407E2B7C090D23146B46CD554A2"   # set "" to skip key checks
BYTES_TO_EVAL  = [0]                                   # add a “hard” byte if needed, e.g., [0, 7]
N_RESAMPLES    = 50                                    # bootstraps per grid point
MIN_GRID_START = 100                                   # smallest N on the grid (if enough traces)
GRID_POINTS    = 8                                     # number of grid points (geomspace)
STATISTIC      = "maxabs"                              # 'maxabs' or 'max' over time

# ROI (pick one or None)
ROI_SAMPLES: Optional[Tuple[int,int]] = None           # e.g., (2000, 15000)
ROI_TIME_S:  Optional[Tuple[float,float]] = None       # e.g., (8e-6, 20e-6) if dt known
ROI_FRACTION: Optional[float] = None                   # e.g., 0.10 for “first 10% of samples”

# Dud filtering (same spirit as your scripts)
P2P_MIN_FRACTION = 0.20
STD_MIN_FRACTION = 0.15

# Thresholds for summary
SR_THRESHOLDS = [0.8, 0.9]                             # report traces-to-SR>=thr
GE_TARGET     = 1.0                                    # report traces-to-GE<=1


# AES helpers / lookup tables

HW_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
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

"""
def dataset_label_from_path(p: Path) -> str:
    name = p.name
    for part, lab in LABEL_FOR_PATHPART.items():
        if part in name:
            return lab
    m = re.search(r"(\d+)\s*MHz", name, re.IGNORECASE)
    return f"{m.group(1)}MHz" if m else name
    
"""
def dataset_label_from_path(p: Path) -> str:
    return p.name


def load_rows_from_csv(ds_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    csv_path = ds_dir / "trace_overview.csv"
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    else:
        # fallback: synthesize from NPZ names
        for p in sorted(ds_dir.glob("*.npz")):
            rows.append({"FileName": p.name, "PlaintextHex": "", "CiphertextHex": ""})
    return rows

def open_trace_any(path: Path):
    """
    Load NPZ/NPY. Return (waveform float32, dt_ns, sr_nominal, pt bytes)
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

def hex_to_bytes16(h: str) -> bytes:
    h = h.strip().replace(" ", "")
    return bytes.fromhex(h)

def time_axis(n: int, dt_ns: Optional[float], sr_nominal: Optional[float]):
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

def apply_roi(traces: np.ndarray, dt_s: Optional[float]):
    N, L = traces.shape
    if ROI_SAMPLES is not None:
        s0, s1 = map(int, ROI_SAMPLES)
        s0 = max(0, min(s0, L))
        s1 = max(s0, min(s1, L))
        return traces[:, s0:s1], (s0, s1)
    if ROI_TIME_S is not None and dt_s:
        t0, t1 = ROI_TIME_S
        s0 = max(0, int(round(t0 / dt_s)))
        s1 = min(L, int(round(t1 / dt_s)))
        return traces[:, s0:s1], (s0, s1)
    if ROI_FRACTION is not None and 0 < ROI_FRACTION <= 1.0:
        s1 = int(round(L * ROI_FRACTION))
        s1 = max(1, min(s1, L))
        return traces[:, :s1], (0, s1)
    return traces, (0, L)


# CPA math (vectorized)

def _standardize_cols(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return Xc / sd

def cpa_corr_matrix(traces: np.ndarray, pts: np.ndarray, byte_idx: int, roi: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """
    corr[guess, t] for all 256 key guesses using HW(SBOX(PT^guess)).
    """
    n, m = traces.shape
    if roi is None:
        s, e = 0, m
    else:
        s, e = roi
    Zt = _standardize_cols(traces[:, s:e])
    ptb = pts[:, byte_idx].astype(np.uint8)
    guesses = np.arange(256, dtype=np.uint8)
    leak = HW_TABLE[SBOX[np.bitwise_xor(ptb[:, None], guesses[None, :])]].astype(np.float64)
    Zl = _standardize_cols(leak)
    corr = (Zl.T @ Zt) / (n - 1.0)  # (256, T_roi)
    return corr

def rank_from_corr(corr: np.ndarray, key_true: int, stat: str = "maxabs") -> int:
    if stat == "maxabs":
        score = np.max(np.abs(corr), axis=1)
    elif stat == "max":
        score = np.max(corr, axis=1)
    else:
        raise ValueError("Unsupported stat")
    order = np.argsort(-score)   # descending
    return int(np.where(order == int(key_true))[0][0])  # 0 = success

def bootstrap_ge_sr(
    traces: np.ndarray,
    pts: np.ndarray,
    byte_idx: int,
    key_true: int,
    Ns: Sequence[int],
    n_resamples: int = 50,
    roi: Optional[Tuple[int,int]] = None,
    stat: str = "maxabs",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    n_total = traces.shape[0]
    Ns = np.array(sorted(int(n) for n in Ns if 2 <= n <= n_total))
    ge = np.zeros_like(Ns, dtype=np.float64)
    sr = np.zeros_like(Ns, dtype=np.float64)
    for i, N in enumerate(Ns):
        ranks = []
        succ = 0
        for _ in range(int(n_resamples)):
            idx = rng.choice(n_total, size=int(N), replace=False)
            cm = cpa_corr_matrix(traces[idx], pts[idx], byte_idx=byte_idx, roi=roi)
            rank = rank_from_corr(cm, key_true=key_true, stat=stat)
            ranks.append(rank)
            if rank == 0:
                succ += 1
        ge[i] = float(np.mean(ranks))
        sr[i] = float(succ) / float(n_resamples)
    return {"Ns": Ns.astype(np.int64), "GE": ge, "SR": sr}


# Plot
def plot_ge(Ns, GE, out_png: Path, title: str):
    plt.figure(figsize=(9,5))
    plt.plot(Ns, GE, marker="o", linewidth=1.2)
    plt.xscale("log"); plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("# traces (log)")
    plt.ylabel("Guessing Entropy (log)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_sr(Ns, SR, out_png: Path, title: str):
    plt.figure(figsize=(9,5))
    plt.plot(Ns, SR, marker="o", linewidth=1.2)
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("# traces (log)")
    plt.ylabel("Success Rate")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# dataset workflow

def eval_dataset(ds_dir: Path, bytes_to_eval: Sequence[int], n_resamples: int):
    label   = dataset_label_from_path(ds_dir)
    out_dir = EVAL_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows_from_csv(ds_dir)
    if not rows:
        print(f"[skip] {label}: no rows")
        return None

    traces, pts = [], []
    dt_ns_global = None

    for r in rows:
        fn = r.get("FileName") or r.get("TraceFilePath")
        if not fn: 
            continue
        p = ds_dir / fn
        if not p.exists():
            p = Path(fn)  # allow absolute path in CSV
            if not p.exists():
                continue

        try:
            wf, dt_ns, sr_nom, pt_npz = open_trace_any(p)
        except Exception as e:
            print(f"[warn] {label}: could not load {p.name}: {e}")
            continue

        if dt_ns and (dt_ns_global is None):
            dt_ns_global = float(dt_ns)

        pt_hex = (r.get("PlaintextHex") or "").strip()
        if pt_hex and len(pt_hex) == 32:
            pt = hex_to_bytes16(pt_hex)
        else:
            pt = pt_npz

        if pt is None or len(pt) != 16:
            continue

        traces.append(wf)
        pts.append(np.frombuffer(pt, dtype=np.uint8))

    if not traces:
        print(f"[warn] {label}: no usable traces")
        return None

    # Truncate to common length & dud filter
    min_len = min(len(w) for w in traces)
    traces = [w[:min_len] for w in traces]
    flags = robust_filter_flags(traces)
    traces = [w for w, ok in zip(traces, flags) if ok]
    pts    = [p for p, ok in zip(pts, flags) if ok]

    if not traces:
        print(f"[warn] {label}: all traces filtered as duds")
        return None

    X = np.stack(traces, axis=0).astype(np.float64)   # (N, L)
    P = np.stack(pts,    axis=0).astype(np.uint8)     # (N, 16)
    N_total, L = X.shape

    # Remove per-trace DC
    X = X - X.mean(axis=1, keepdims=True)

    # Apply ROI
    _, dt_s = time_axis(L, dt_ns_global, None)
    X, (s0, s1) = apply_roi(X, dt_s)

    # Full-dataset best-guess key (informational only)
    recovered_key = np.zeros(16, dtype=np.uint8)
    best_corr     = np.zeros(16, dtype=np.float64)
    for b in range(16):
        corr = cpa_corr_matrix(X, P, byte_idx=b, roi=None)  # already sliced by ROI
        if STATISTIC == "maxabs":
            score = np.max(np.abs(corr), axis=1)
        else:
            score = np.max(corr, axis=1)
        g = int(np.argmax(score))
        recovered_key[b] = g
        best_corr[b] = float(score[g])

    (out_dir / "recovered_key_hex.txt").write_text("".join(f"{x:02X}" for x in recovered_key), encoding="utf-8")

    # Build geometric grid of Ns up to N_total
    if N_total <= 2:
        print(f"[warn] {label}: too few traces ({N_total})")
        return None
    grid_min = min(max(MIN_GRID_START, 10), N_total)
    if grid_min >= N_total:
        Ns = np.array([N_total], dtype=np.int64)
    else:
        Ns = np.unique(np.rint(np.geomspace(grid_min, N_total, num=GRID_POINTS)).astype(int))
        Ns = Ns[Ns <= N_total]

    # Evaluate bytes
    per_byte_results = {}
    rng = np.random.default_rng()

    for b in bytes_to_eval:
        key_true = None
        if len(TRUE_KEY_HEX) == 32:
            key_true = int(TRUE_KEY_HEX[b*2:(b+1)*2], 16)

        res = bootstrap_ge_sr(
            X, P, byte_idx=b, key_true=(key_true if key_true is not None else 0),
            Ns=Ns, n_resamples=n_resamples, roi=None, stat=STATISTIC, rng=rng
        )

        # Save per-byte artifacts
        bdir = out_dir / f"byte_b{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        np.save(bdir / "Ns.npy", res["Ns"])
        np.save(bdir / "GE.npy", res["GE"])
        np.save(bdir / "SR.npy", res["SR"])

        # Plots
        plot_ge(res["Ns"], res["GE"], bdir / f"GE_vs_traces_b{b:02d}.png", f"{label} – CPA GE (byte {b})")
        plot_sr(res["Ns"], res["SR"], bdir / f"SR_vs_traces_b{b:02d}.png", f"{label} – CPA SR (byte {b})")

        # Summaries: traces to SR>=thr and GE<=1
        summary = {
            "dataset": label,
            "byte": int(b),
            "N_total": int(N_total),
            "roi_samples": [int(s0), int(s1)],
            "statistic": STATISTIC,
            "n_resamples": int(n_resamples),
            "recovered_key_hex_all_traces": "".join(f"{x:02X}" for x in recovered_key),
            "best_corr_all_traces": float(best_corr[b]),
            "true_key_byte_hex": (f"{key_true:02X}" if key_true is not None else None),
        }

        def traces_to_threshold(xN, y, target, mode):
            # 'ge_to_le' (GE<=target) or 'sr_to_ge' (SR>=target)
            if len(xN) == 0: return None
            if mode == "ge_to_le":
                idx = np.where(y <= target)[0]
            else:
                idx = np.where(y >= target)[0]
            return int(xN[int(idx[0])]) if idx.size else None

        summary["traces_to_GE_le_1"] = traces_to_threshold(res["Ns"], res["GE"], GE_TARGET, "ge_to_le")
        for th in SR_THRESHOLDS:
            summary[f"traces_to_SR_ge_{int(th*100)}pct"] = traces_to_threshold(res["Ns"], res["SR"], th, "sr_to_ge")

        (bdir / "GE_SR.json").write_text(json.dumps({
            "Ns": res["Ns"].tolist(),
            "GE": res["GE"].tolist(),
            "SR": res["SR"].tolist(),
            "summary": summary
        }, indent=2), encoding="utf-8")

        lines = [
            f"CPA summary – {label} – byte {b}",
            f"N_total (after filter, ROI): {N_total}",
            f"ROI samples: [{s0}, {s1})  (length {s1 - s0})",
            f"statistic: {STATISTIC}  resamples: {n_resamples}",
            f"best |corr| (all traces, ROI): {best_corr[b]:.6f}",
        ]
        if summary["traces_to_GE_le_1"] is not None:
            lines.append(f"traces to GE<=1: {summary['traces_to_GE_le_1']}")
        else:
            lines.append("traces to GE<=1: not reached on grid")
        for th in SR_THRESHOLDS:
            key = f"traces_to_SR_ge_{int(th*100)}pct"
            val = summary[key]
            lines.append(f"traces to SR>={int(th*100)}%: {val if val is not None else 'not reached on grid'}")
        (bdir / f"summary_b{b:02d}.txt").write_text("\n".join(lines), encoding="utf-8")

        per_byte_results[b] = {"Ns": res["Ns"], "GE": res["GE"], "SR": res["SR"], "summary": summary}

    return {"label": label, "N_total": int(N_total), "per_byte": per_byte_results}

# COMPARISON overlays
def write_comparison(per_ds, byte_idx: int):
    comp_dir = EVAL_ROOT / "COMPARISON"
    comp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # GE overlay
    plt.figure(figsize=(9,5))
    for ds in per_ds:
        label = ds["label"]
        pb = ds["per_byte"].get(byte_idx)
        if not pb:
            continue
        Ns, GE = pb["Ns"], pb["GE"]
        plt.plot(Ns, GE, marker="o", linewidth=1.2, label=label)
        s = pb["summary"]
        rows.append({
            "dataset": label,
            "N_total": ds["N_total"],
            "traces_to_GE_le_1": s["traces_to_GE_le_1"],
            **{k: v for k, v in s.items() if k.startswith("traces_to_SR_ge_")}
        })
    plt.xscale("log"); plt.yscale("log"); plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("# traces (log)"); plt.ylabel("Guessing Entropy (log)")
    plt.title(f"CPA GE comparison (byte {byte_idx})"); plt.legend()
    plt.tight_layout(); plt.savefig(comp_dir / f"GE_overlay_b{byte_idx:02d}.png", dpi=150); plt.close()

    # SR overlay
    plt.figure(figsize=(9,5))
    for ds in per_ds:
        label = ds["label"]
        pb = ds["per_byte"].get(byte_idx)
        if not pb:
            continue
        Ns, SR = pb["Ns"], pb["SR"]
        plt.plot(Ns, SR, marker="o", linewidth=1.2, label=label)
    plt.xscale("log"); plt.ylim(0, 1.0); plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("# traces (log)"); plt.ylabel("Success Rate")
    plt.title(f"CPA SR comparison (byte {byte_idx})"); plt.legend()
    plt.tight_layout(); plt.savefig(comp_dir / f"SR_overlay_b{byte_idx:02d}.png", dpi=150); plt.close()

    # CSV/JSON
    if rows:
        fields = list(rows[0].keys())
        with open(comp_dir / f"cpa_comparison_b{byte_idx:02d}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        (comp_dir / f"cpa_comparison_b{byte_idx:02d}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

def main():
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    # allow custom dataset dirs on CLI
    ds_list = [Path(p) for p in (sys.argv[1:] if len(sys.argv) > 1 else DATASETS)]

    per_ds_results = []
    for ds in ds_list:
        ds = ds.resolve()
        if not ds.exists():
            print(f"[skip] dataset not found: {ds}")
            continue
        out = eval_dataset(ds, BYTES_TO_EVAL, N_RESAMPLES)
        if out:
            per_ds_results.append(out)

    # Overlays for the first requested byte (if any)
    if per_ds_results and BYTES_TO_EVAL:
        write_comparison(per_ds_results, BYTES_TO_EVAL[0])

    if per_ds_results:
        print(f"CPA evaluation saved under: {EVAL_ROOT}")
    else:
        print("No datasets produced results.")

if __name__ == "__main__":
    main()
