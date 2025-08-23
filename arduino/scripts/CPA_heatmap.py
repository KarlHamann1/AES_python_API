#!/usr/bin/env python3
"""
cpa_heatmap.py

Compute and plot a correlation heatmap for CPA:
rows = key guesses 0..255, columns = time samples within ROI.
Entries are |rho(g, t)| for the byte index of interest.

Inputs per dataset (under arduino/data):
- trace_overview.csv listing files and plaintext; or plaintext inside NPZ
- NPZ files with 'trace_mean' or 'trace' and optionally 'plaintext', 'dt_ns', 'sr_nominal'

Outputs under arduino/data/CPA_heatmap/<Label>/ :
- heatmap_abs_corr_bXX.png
- corr_abs_bXX.npy
- max_abs_corr_over_time_bXX.npy
- argmax_guess_over_time_bXX.npy
- time_axis.npy or sample_axis.npy
- summary_bXX.txt
"""

from __future__ import annotations
import sys, re, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# repo layout
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "CPA_heatmap").resolve()

# dataset selection (multiple datasets supported)
DATASETS = [
    DATA_ROOT / "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV",
    DATA_ROOT / "data_arduino_1MHz_tb16_4.81Msps_9600Bd_avg100_10ms_20MHzBW_12bit_ACoff4mV",
    DATA_ROOT / "data_arduino_8MHz_tb5_31.25Msps_57600Bd_avg100_1.5ms_20MHzBW_12bit_ACoff2mV",
    DATA_ROOT / "data_arduino_16MHz_tb5_31Msps_115200Bd_avg100_0p7ms",
]

# display labels to keep names consistent in figures
LABEL_MAP = {
    "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV": "16MHz_round1_125Msps",
    "1MHz":  "1MHz",
    "8MHz":  "8MHz",
    "16MHz": "16MHz",
}

def dataset_label_from_path(p: Path) -> str:
    name = p.name
    for key, lab in LABEL_MAP.items():
        if key in name:
            return lab
    m = re.search(r"(\d+)\s*MHz", name, re.IGNORECASE)
    return f"{m.group(1)}MHz" if m else name

# CPA parameters
BYTE_INDEX     = 0
TRUE_KEY_HEX   = "81556407E2B7C090D23146B46CD554A2"   # set "" to skip highlighting
USE_ABS        = True

# ROI choices (pick one or leave all None)
ROI_SAMPLES: Optional[Tuple[int,int]] = None
ROI_TIME_S:  Optional[Tuple[float,float]] = None
ROI_FRACTION: Optional[float] = 0.15

# speed knobs
MAX_TRACES: Optional[int] = None
DECIMATE:   int = 1

# dud filtering
P2P_MIN_FRACTION = 0.20
STD_MIN_FRACTION = 0.15

# AES lookup
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

def load_rows_from_csv(ds_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    csv_path = ds_dir / "trace_overview.csv"
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    else:
        for p in sorted(ds_dir.glob("*.npz")):
            rows.append({"FileName": p.name, "PlaintextHex": ""})
    return rows

def open_trace_any(path: Path):
    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        if "trace_mean" in npz.files:
            wf = np.asarray(npz["trace_mean"], dtype=np.float32)
        elif "trace" in npz.files:
            wf = np.asarray(npz["trace"], dtype=np.float32)
        else:
            raise ValueError(f"{path.name}: no 'trace_mean'/'trace'")
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

def apply_roi_matrix(X: np.ndarray, dt_s: Optional[float]):
    N, L = X.shape
    if ROI_SAMPLES is not None:
        s0, s1 = map(int, ROI_SAMPLES)
        s0 = max(0, min(s0, L)); s1 = max(s0, min(s1, L))
        return X[:, s0:s1], (s0, s1)
    if ROI_TIME_S is not None and dt_s:
        t0, t1 = ROI_TIME_S
        s0 = max(0, int(round(t0 / dt_s)))
        s1 = min(L, int(round(t1 / dt_s)))
        return X[:, s0:s1], (s0, s1)
    if ROI_FRACTION is not None and 0 < ROI_FRACTION <= 1.0:
        s1 = int(round(L * ROI_FRACTION))
        s1 = max(1, min(s1, L))
        return X[:, :s1], (0, s1)
    return X, (0, L)

def standardize_cols(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return Xc / sd

def cpa_corr_matrix(traces: np.ndarray, pts: np.ndarray, byte_idx: int) -> np.ndarray:
    n, m = traces.shape
    Zt = standardize_cols(traces)
    ptb = pts[:, byte_idx].astype(np.uint8)
    guesses = np.arange(256, dtype=np.uint8)
    leak = HW[SBOX[np.bitwise_xor(ptb[:, None], guesses[None, :])]].astype(np.float64)
    Zl = standardize_cols(leak)
    corr = (Zl.T @ Zt) / (n - 1.0)
    return corr

def process_dataset(ds: Path):
    label = dataset_label_from_path(ds)
    out_dir = EVAL_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows_from_csv(ds)
    traces, pts = [], []
    dt_ns_global, sr_global = None, None

    for r in rows:
        fn = r.get("FileName") or r.get("TraceFilePath")
        if not fn:
            continue
        p = ds / fn
        if not p.exists():
            p = Path(fn)
            if not p.exists():
                continue
        try:
            wf, dt_ns, sr_nom, pt_npz = open_trace_any(p)
        except Exception:
            continue

        if dt_ns_global is None and dt_ns:
            dt_ns_global = float(dt_ns)
        if sr_global is None and sr_nom:
            sr_global = float(sr_nom)

        pt_hex = (r.get("PlaintextHex") or "").strip()
        if pt_hex and len(pt_hex) == 32:
            pt = hex_to_bytes16(pt_hex)
        else:
            pt = pt_npz
        if pt is None or len(pt) != 16:
            continue

        traces.append(wf)
        pts.append(np.frombuffer(pt, dtype=np.uint8))

        if MAX_TRACES and len(traces) >= int(MAX_TRACES):
            break

    if not traces:
        print(f"[skip] no usable traces in {ds}")
        return

    min_len = min(len(w) for w in traces)
    traces = [w[:min_len] for w in traces]
    flags = robust_filter_flags(traces)
    X = np.stack([w for w, ok in zip(traces, flags) if ok], axis=0).astype(np.float64)
    P = np.stack([p for p, ok in zip(pts,    flags) if ok], axis=0).astype(np.uint8)
    if X.size == 0:
        print(f"[skip] all traces filtered in {ds}")
        return

    X = X - X.mean(axis=1, keepdims=True)

    axis, dt_s = time_axis(X.shape[1], dt_ns_global, sr_global)
    X, (s0, s1) = apply_roi_matrix(X, dt_s)

    if DECIMATE > 1:
        X = X[:, ::DECIMATE]
        axis = axis[s0:s1:DECIMATE]
    else:
        axis = axis[s0:s1]

    corr = cpa_corr_matrix(X, P, byte_idx=BYTE_INDEX)
    corr_abs = np.abs(corr) if USE_ABS else corr

    np.save(out_dir / f"corr_abs_b{BYTE_INDEX:02d}.npy", np.abs(corr))
    np.save(out_dir / ("time_axis.npy" if dt_s else "sample_axis.npy"), axis)

    max_over_g = np.max(np.abs(corr), axis=0)
    arg_over_g = np.argmax(np.abs(corr), axis=0)
    np.save(out_dir / f"max_abs_corr_over_time_b{BYTE_INDEX:02d}.npy", max_over_g)
    np.save(out_dir / f"argmax_guess_over_time_b{BYTE_INDEX:02d}.npy", arg_over_g)

    plt.figure(figsize=(10, 6))
    if dt_s:
        extent = [axis[0], axis[-1], 255, 0]
        xlab = "Time (s)"
    else:
        extent = [s0, s0 + corr.shape[1]*DECIMATE, 255, 0]
        xlab = "Sample index"
    im = plt.imshow(corr_abs, aspect="auto", interpolation="nearest", extent=extent)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="|corr|" if USE_ABS else "corr")
    plt.xlabel(xlab)
    plt.ylabel("Key guess (0..255)")
    plt.title(f"CPA correlation heatmap – {label} – byte {BYTE_INDEX}")

    if TRUE_KEY_HEX and len(TRUE_KEY_HEX) == 32:
        key_true = int(TRUE_KEY_HEX[BYTE_INDEX*2:(BYTE_INDEX+1)*2], 16)
        y = 255 - key_true
        plt.axhline(y=y, color="white", linewidth=1.2, linestyle="--", alpha=0.9)

    xs = np.linspace(extent[0], extent[1], num=arg_over_g.shape[0])
    ys = 255 - arg_over_g
    plt.plot(xs, ys, linewidth=0.8, alpha=0.9, color="white")

    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_abs_corr_b{BYTE_INDEX:02d}.png", dpi=170)
    plt.close()

    lines = [
        f"dataset: {label}",
        f"byte_index: {BYTE_INDEX}",
        f"traces_kept: {X.shape[0]}",
        f"roi_samples: [{s0}, {s1})  decimate={DECIMATE}",
        f"use_abs: {USE_ABS}",
        f"time_axis: {'seconds' if dt_s else 'samples'}",
        f"true_key_byte: {(TRUE_KEY_HEX[BYTE_INDEX*2:(BYTE_INDEX+1)*2] if TRUE_KEY_HEX else 'n/a')}",
        f"max |corr| over time: {float(max_over_g.max()):.6f}",
        f"heatmap_png: {out_dir / f'heatmap_abs_corr_b{BYTE_INDEX:02d}.png'}",
    ]
    (out_dir / f"summary_b{BYTE_INDEX:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved heatmap for {label} -> {out_dir}")

def main():
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_args = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else list(DATASETS)
    for ds in ds_args:
        ds = Path(ds).resolve()
        if not ds.exists():
            print(f"[skip] dataset not found: {ds}")
            continue
        process_dataset(ds)

if __name__ == "__main__":
    main()
