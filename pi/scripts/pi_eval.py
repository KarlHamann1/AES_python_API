#!/usr/bin/env python3
"""
pi_noleak_eval.py

Quick checks to show that the Raspberry Pi captures do not exhibit detectable first-order leakage.
It runs:
- a simple positive control on the mult/idle dataset (Welch-style t-stat on early vs late window)
- model-free tests (NICV and MI) on the AES dataset
- a CPA correlation upper bound with a Fisher confidence interval

Outputs are written under: pi/data/CPA_eval/PI_eval/
"""

from __future__ import annotations
import os, csv, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

# repo layout (script is expected in: pi/scripts/)
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
PI_DIR      = REPO_ROOT / "pi"
DATA_ROOT   = (PI_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "CPA_eval").resolve()

# dataset roots
AES_DIR      = DATA_ROOT / "data_pi_0dB_single_encryption"
MULTIDLE_DIR = DATA_ROOT / "data_pi_20dB_10MV_block_mult_500_micro_with_300_micro_idle"

RNG = np.random.default_rng(7)

def find_csv_in(dir_path: Path) -> Optional[Path]:
    # try common names first, then any .csv
    for name in ("trace_overview_block.csv", "trace_overview.csv"):
        p = dir_path / name
        if p.exists():
            return p
    cands = sorted(dir_path.glob("*.csv"))
    return cands[0] if cands else None

def load_csv_rows(csv_path: Path) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

def load_wave(ds_dir: Path, fn: str) -> np.ndarray:
    p = ds_dir / fn
    if not p.exists():
        # allow absolute/relative paths in CSV
        p = Path(fn)
    x = np.load(p).astype(np.float64, copy=False)
    return x

def center(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)

def robust_filter_flags(traces: List[np.ndarray]) -> np.ndarray:
    # drop obviously "dead" traces using peak-to-peak and std heuristics
    p2p = np.array([float(np.max(t) - np.min(t)) for t in traces], dtype=np.float64)
    sd  = np.array([float(np.std(t))            for t in traces], dtype=np.float64)
    med_p2p = np.median(p2p) if np.isfinite(p2p).all() else 0.0
    med_sd  = np.median(sd)  if np.isfinite(sd).all()  else 0.0
    ok_p2p = p2p >= (0.20 * max(med_p2p, 1e-12))
    ok_sd  = sd  >= (0.15 * max(med_sd,  1e-12))
    return ok_p2p & ok_sd

def welch_t_stat(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    # returns (t, dof). p-value is skipped to avoid SciPy.
    n1, n2 = len(a), len(b)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    denom = np.sqrt(v1 / max(n1,1) + v2 / max(n2,1) + 1e-18)
    t = (m1 - m2) / denom
    num = (v1 / max(n1,1) + v2 / max(n2,1))**2
    den = (v1**2) / (max(n1,1)**2 * max(n1-1,1)) + (v2**2) / (max(n2,1)**2 * max(n2-1,1))
    dof = num / max(den, 1e-18)
    return float(t), float(dof)

def nicv(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # traces: N x T, labels: N
    N, T = traces.shape
    grand = np.mean(traces, axis=0)
    den = np.mean((traces - grand)**2, axis=0) + 1e-18
    num = np.zeros(T, dtype=np.float64)
    for u in np.unique(labels):
        idx = labels == u
        w = np.sum(idx) / float(N)
        mu = np.mean(traces[idx, :], axis=0)
        num += w * (mu - grand)**2
    return num / den

def mi_binned(traces: np.ndarray, byte_vals: np.ndarray, nbins: int = 32) -> np.ndarray:
    """
    Estimate MI I(trace[t]; plaintext_byte0) via equal-width binning.
    Returns MI[t] in bits. Robust to empty bins and avoids shape mismatches.
    """
    N, T = traces.shape
    y = byte_vals.astype(np.uint8)         # 0..255
    MI = np.zeros(T, dtype=np.float64)

    for t in range(T):
        x = traces[:, t]

        # Bin the analog sample x into nbins bins
        edges = np.histogram_bin_edges(x, bins=nbins)
        xb = np.digitize(x, edges) - 1
        xb = np.clip(xb, 0, nbins - 1)

        # Joint histogram p(b,y) on a (nbins x 256) grid
        joint = np.zeros((nbins, 256), dtype=np.float64)
        np.add.at(joint, (xb, y), 1.0)
        joint /= float(N)

        # Marginals p(b), p(y)
        px = joint.sum(axis=1, keepdims=True)   # (nbins, 1)
        py = joint.sum(axis=0, keepdims=True)   # (1, 256)

        # Elementwise ratio p(b,y)/(p(b)p(y)) over the full matrix
        denom = px * py                          # (nbins, 256)
        mask = (joint > 0) & (denom > 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.zeros_like(joint)
            ratio[mask] = joint[mask] / denom[mask]
            MI_t = (joint[mask] * np.log2(ratio[mask])).sum()

        MI[t] = float(max(0.0, MI_t))           # numerical floor at 0
    return MI


def permute_max_stat(stat_fn, X: np.ndarray, y: np.ndarray, iters: int = 300) -> Tuple[float, float]:
    # permutation test on the max statistic over time
    obs = stat_fn(X, y)
    obs_max = float(np.max(obs))
    cnt = 0
    max_null = 0.0
    y_perm = np.array(y, copy=True)
    for _ in range(iters):
        RNG.shuffle(y_perm)
        s = stat_fn(X, y_perm)
        s_max = float(np.max(s))
        max_null = max(max_null, s_max)
        if s_max >= obs_max - 1e-12:
            cnt += 1
    p = (cnt + 1) / (iters + 1)
    return p, max_null

# AES S-box and HW model for the CPA bound
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

def hw8(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x[:, None], axis=1).sum(axis=1).astype(np.uint8)

def cpa_corr(X: np.ndarray, pt_b0: np.ndarray, key_guess: int) -> np.ndarray:
    # X: N x T, pt_b0: N (plaintext byte 0), guess: 0..255
    leak = hw8(SBOX[pt_b0 ^ key_guess]).astype(np.float64)
    leak = (leak - leak.mean()) / (leak.std() + 1e-18)
    Xc = X - X.mean(0)
    num = Xc.T @ leak
    den = np.sqrt((Xc**2).sum(0) * (len(leak) - 1))
    r = num / (den + 1e-18)
    return r

def fisher_upper_bound(r: np.ndarray, n: int) -> np.ndarray:
    # per-time 95% upper bound on |rho|
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    z_hi = z + 1.96 / max(n - 3, 1)
    return np.tanh(z_hi)

def run_mult_idle_ttest():
    if not MULTIDLE_DIR.exists():
        return
    csv_path = find_csv_in(MULTIDLE_DIR)
    if not csv_path:
        return
    out_dir = (EVAL_ROOT / "mult_idle").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(csv_path)
    waves = []
    for r in rows:
        try:
            waves.append(center(load_wave(MULTIDLE_DIR, r["TraceFilePath"])))
        except Exception:
            continue
    if not waves:
        return

    X = np.stack(waves, axis=0)
    N, T = X.shape

    # quick split: first half as "mult", second half as "idle"
    s_mid = T // 2
    A = X[:, :s_mid].mean(1)
    B = X[:, s_mid:].mean(1)
    t, dof = welch_t_stat(A, B)

    # tiny plot of the grand means to illustrate the split
    gm = X.mean(0)
    tt = np.arange(T)
    plt.figure(figsize=(8, 3))
    plt.plot(tt, gm, linewidth=1.0)
    plt.axvline(s_mid, linestyle="--", alpha=0.6)
    plt.title("Grand mean (mult/idle split marker)")
    plt.xlabel("samples")
    plt.ylabel("a.u.")
    plt.tight_layout()
    plt.savefig(out_dir / "grand_mean_split.png", dpi=170)
    plt.close()

    (out_dir / "summary.txt").write_text(
        f"traces={N}, samples={T}\n"
        f"split_index={s_mid}\n"
        f"Welch_t={t:.2f}, dof≈{dof:.1f}\n",
        encoding="utf-8"
    )

def run_aes_model_free_and_bound():
    if not AES_DIR.exists():
        return
    csv_path = find_csv_in(AES_DIR)
    if not csv_path:
        return
    out_dir = (EVAL_ROOT / "aes_single").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(csv_path)
    waves = []
    pt_b0 = []
    for r in rows:
        try:
            w = center(load_wave(AES_DIR, r["TraceFilePath"]))
            waves.append(w)
            # use the first plaintext byte (two hex chars)
            pt_b0.append(int(r["PlaintextHex"][0:2], 16))
        except Exception:
            continue
    if not waves:
        return

    L = min(len(w) for w in waves)
    X = np.stack([w[:L] for w in waves], axis=0)
    pt_b0 = np.array(pt_b0[:X.shape[0]], dtype=np.uint8)
    N, T = X.shape

    # NICV on bit-0 and on high nibble
    y_bit0 = (pt_b0 & 0x01)
    y_nib  = (pt_b0 >> 4)  # 16 classes
    nicv_bit = nicv(X, y_bit0)
    nicv_nib = nicv(X, y_nib)
    np.save(out_dir / "nicv_bit0.npy", nicv_bit)
    np.save(out_dir / "nicv_nibble.npy", nicv_nib)

    # simple plots (optional but helpful)
    x_axis = np.arange(T)
    plt.figure(figsize=(9, 3))
    plt.plot(x_axis, nicv_bit, label="NICV bit0", linewidth=1.0)
    plt.plot(x_axis, nicv_nib, label="NICV nibble", linewidth=1.0, alpha=0.8)
    plt.xlabel("samples")
    plt.ylabel("NICV")
    plt.title("NICV vs time (AES single-encryption set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nicv.png", dpi=170)
    plt.close()

    # MI (binned) + permutation test on max(MI)
    MI = mi_binned(X, pt_b0, nbins=32)
    p_mi, mi_null = permute_max_stat(lambda A,B: mi_binned(A,B,nbins=32), X, pt_b0, iters=300)
    np.save(out_dir / "mi.npy", MI)
    (out_dir / "mi_summary.json").write_text(
        json.dumps({"N": int(N), "T": int(T), "max_MI_bits": float(MI.max()),
                    "perm_p_value_on_max": float(p_mi), "max_null_maxMI_bits": float(mi_null)}, indent=2),
        encoding="utf-8"
    )
    plt.figure(figsize=(9, 3))
    plt.plot(x_axis, MI, linewidth=1.0)
    plt.xlabel("samples")
    plt.ylabel("MI (bits)")
    plt.title("Mutual Information vs time")
    plt.tight_layout()
    plt.savefig(out_dir / "mi.png", dpi=170)
    plt.close()

    # CPA correlation upper bound (byte 0, first-round S-box HW)
    Rmax = np.zeros(T, dtype=np.float64)
    for k in range(256):
        r = np.abs(cpa_corr(X, pt_b0, k))
        Rmax = np.maximum(Rmax, r)
    Rup = fisher_upper_bound(Rmax, N)
    np.save(out_dir / "cpa_rmax.npy", Rmax)
    np.save(out_dir / "cpa_rupper95.npy", Rup)
    (out_dir / "bound_summary.txt").write_text(
        f"N={N}, T={T}\n"
        f"max|r|={float(Rmax.max()):.4f}\n"
        f"per-time 95% upper bound max={float(Rup.max()):.4f}\n"
        f"(note: CI is per time sample; apply your preferred multiple-comparison control if needed)\n",
        encoding="utf-8"
    )

    plt.figure(figsize=(9, 3))
    plt.plot(x_axis, Rmax, linewidth=1.0, label="max |r| across guesses")
    plt.plot(x_axis, Rup, linewidth=1.0, alpha=0.9, label="95% upper bound")
    plt.xlabel("samples")
    plt.ylabel("|r|")
    plt.title("CPA correlation upper bound (byte 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cpa_bound.png", dpi=170)
    plt.close()

def main():
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    run_mult_idle_ttest()
    run_aes_model_free_and_bound()
    print(f"Pi no-leak evaluation saved under: {EVAL_ROOT}")

if __name__ == "__main__":
    main()
