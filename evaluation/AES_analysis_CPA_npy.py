#!/usr/bin/env python3
"""
cpa_aes_encryption.py

CPA on AES traces stored as *.npy, using a first-round SubBytes HW model.


- Repo-aware paths (arduino/pi) with a single DATASET_DIR_NAME to switch sets.
- Handles CSV with either:
    * TraceIndex, PlaintextHex, CiphertextHex, TraceFilePath
      (absolute or relative *.npy path per row), OR
    * BlockIndex, PlaintextHex, CiphertextHex, FileName
    (file inside the dataset folder, from my NPZ/CSV flow)
- Crops all traces to the shortest length so shapes line up.
- Optional ROI by samples, or by time if dt is known (see DT_NS_OVERRIDE).
- Vectorized CPA core (precompute trace centering once).
- Plots correlation curve for byte 0 (best vs. one wrong guess).
- Compares recovered key to a known reference key.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


REPO_ROOT    = Path(__file__).resolve().parents[1] 
ARDUINO_DATA = REPO_ROOT / "arduino" / "data"
PI_DATA      = REPO_ROOT / "pi" / "data"

#  Configuration 
# choose the dataset folder
DATASET_DIR_NAME = "data_arduino_16MHz_tb5_31Msps_115200Bd_avg100"
DATA_DIR         = ARDUINO_DATA / DATASET_DIR_NAME
CSV_FILE         = DATA_DIR / "trace_overview.csv"

# cap number of traces (None = use all)
MAX_TRACES = 10_000

# known true key as hex (16 bytes)
TRUE_KEY_HEX = "81556407E2B7C090D23146B46CD554A2"

# Region of interest:
ROI_SAMPLES     = None             # e.g., (start, end) in samples, or None
ROI_TIME_S      = None             # e.g., (t0, t1) seconds, needs a known dt (see below)
DT_NS_OVERRIDE  = None             # set to a number (e.g., 32.0) if CSV doesn't have dt

# plotting
SHOW_CORR_PLOT  = True
SAVE_PLOT_PATH  = None             # e.g., DATA_DIR/"corr_byte0.png" or None


# Hamming weight (0..255)
HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

# AES S-box
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

def _hex16(s: str) -> bytes:
    return bytes.fromhex(s.strip().replace(" ", ""))

def main():
    # --- read CSV and load traces ---
    csv_path = Path(CSV_FILE)
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}")
        return

    pts, traces = [], []
    dt_ns_global = None  # try to detect dt from CSV if available

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_hex = (row.get("PlaintextHex") or "").strip()
            if not pt_hex:
                continue

            # file path column name can differ
            #  - NPZ flow:   FileName (relative to DATA_DIR)
            #  - NPY flow:   TraceFilePath (absolute or relative)
            path_token = (row.get("TraceFilePath") or row.get("FileName") or "").strip()
            if not path_token:
                continue

            # make a path
            path = Path(path_token)
            if not path.is_file():
                path = DATA_DIR / path_token  # try relative to dataset dir
            if not path.is_file():
                print(f"[warn] trace not found: {path_token}")
                continue

            # load waveform (npy only)
            try:
                wf = np.asarray(np.load(path), dtype=np.float32)
            except Exception as e:
                print(f"[warn] failed to load {path}: {e}")
                continue

            # plaintext (I don't need ciphertext for round-1 CPA)
            pt = _hex16(pt_hex)
            if len(pt) != 16:
                print(f"[warn] bad PT length in CSV: {pt_hex}")
                continue

            pts.append(np.frombuffer(pt, dtype=np.uint8))
            traces.append(wf)

            # optionally grab dt_ns if CSV happens to include it
            if dt_ns_global is None:
                dt_str = row.get("dt_ns")
                if dt_str:
                    try:
                        dt_ns_global = float(dt_str)
                    except ValueError:
                        pass

            if MAX_TRACES is not None and len(traces) >= MAX_TRACES:
                break

    if not traces:
        print("No traces loaded.")
        return

    # crop to the shortest length so all shapes match
    min_len = min(len(w) for w in traces)
    traces = np.stack([w[:min_len] for w in traces], axis=0).astype(np.float64)  # (N, L)
    pts = np.stack(pts, axis=0)                                                  # (N, 16)
    N, L = traces.shape
    print(f"Loaded {N} traces, {L} samples each.")

    # --- ROI handling ---
    s0, s1 = 0, L
    if ROI_SAMPLES is not None:
        s0, s1 = max(0, int(ROI_SAMPLES[0])), min(L, int(ROI_SAMPLES[1]))
        traces = traces[:, s0:s1]; L = traces.shape[1]
        print(f"Applied ROI_SAMPLES=({s0},{s1}) → {L} samples.")
    elif ROI_TIME_S is not None:
        dt_ns = DT_NS_OVERRIDE if DT_NS_OVERRIDE is not None else dt_ns_global
        if dt_ns and dt_ns > 0:
            dt = dt_ns * 1e-9
            s0 = max(0, int(round(ROI_TIME_S[0] / dt)))
            s1 = min(L, int(round(ROI_TIME_S[1] / dt)))
            traces = traces[:, s0:s1]; L = traces.shape[1]
            print(f"Applied ROI_TIME_S=({ROI_TIME_S[0]:.3e},{ROI_TIME_S[1]:.3e}) s → samples ({s0},{s1}) → {L} samples.")
        else:
            print("[info] ROI_TIME_S set but no dt known; skipping time-ROI.")

    # --- precompute centered traces once (vectorized Pearson) ---
    t_mean = traces.mean(axis=0)        # (L,)
    t_diff = traces - t_mean            # (N,L)
    ss_t   = np.sum(t_diff * t_diff, axis=0)  # (L,)

    recovered_key = np.zeros(16, dtype=np.uint8)
    best_corrs    = np.zeros(16, dtype=np.float64)

    # for plotting byte 0
    corr_vec_best_b0 = None
    corr_vec_wrong_b0 = None
    best_guess_b0 = None
    wrong_guess_b0 = None

    # --- CPA per byte ---
    for byte_idx in range(16):
        best_g = 0
        best_c = 0.0
        corr_vec_best = None

        pt_byte = pts[:, byte_idx]  # (N,)

        # try all 256 guesses
        for g in range(256):
            # HW(SBOX(PT ^ g))
            h = HW[SBOX[pt_byte ^ g]].astype(np.float64)  # (N,)
            h_diff = h - h.mean()
            ss_h = np.dot(h_diff, h_diff)
            if ss_h <= 1e-18:
                continue

            # corr over all time samples at once
            num   = h_diff @ t_diff             # (L,)
            denom = np.sqrt(ss_h * ss_t)        # (L,)
            corr  = np.zeros(L, dtype=np.float64)
            mask  = denom > 1e-18
            corr[mask] = num[mask] / denom[mask]

            local_max = np.max(np.abs(corr))
            if local_max > best_c:
                best_c = local_max
                best_g = g
                corr_vec_best = corr

        recovered_key[byte_idx] = best_g
        best_corrs[byte_idx] = best_c
        print(f"Byte {byte_idx:2d}: guess=0x{best_g:02X}, |corr|max={best_c:.6f}")

        # keep curves for byte 0
        if byte_idx == 0 and corr_vec_best is not None:
            best_guess_b0 = best_g
            corr_vec_best_b0 = corr_vec_best
            wg = 0x00 if best_g != 0x00 else 0x01
            wrong_guess_b0 = wg

            # one wrong curve for visual comparison
            h_w   = HW[SBOX[pt_byte ^ wg]].astype(np.float64)
            h_w_d = h_w - h_w.mean()
            ss_h_w = np.dot(h_w_d, h_w_d)
            if ss_h_w > 1e-18:
                num_w   = h_w_d @ t_diff
                denom_w = np.sqrt(ss_h_w * ss_t)
                corr_w  = np.zeros(L, dtype=np.float64)
                mask_w  = denom_w > 1e-18
                corr_w[mask_w] = num_w[mask_w] / denom_w[mask_w]
                corr_vec_wrong_b0 = corr_w

    # --- print recovered key and compare ---
    key_hex = "".join(f"{b:02X}" for b in recovered_key)
    print("\n=== Recovered Key (CPA, Round-1 SubBytes) ===")
    print(key_hex)
    print("=============================================\n")

    if len(TRUE_KEY_HEX) == 32:
        if key_hex.upper() == TRUE_KEY_HEX.upper():
            print("[✓] Recovered key matches the known true key!")
        else:
            print("[✗] Recovered key does NOT match the known key.")
            print(f"    True key: {TRUE_KEY_HEX.upper()}")
            print(f"    Found   : {key_hex.upper()}")

    # --- plot byte-0 correlation (best vs wrong) ---
    if SHOW_CORR_PLOT and corr_vec_best_b0 is not None:
        # x-axis in time if dt is known, else samples
        dt_ns = DT_NS_OVERRIDE if DT_NS_OVERRIDE is not None else dt_ns_global
        if ROI_TIME_S is not None and dt_ns and dt_ns > 0:
            dt = dt_ns * 1e-9
            x = np.arange(corr_vec_best_b0.size) * dt + s0 * dt
            xlabel = "Time (s)"
        else:
            x = np.arange(corr_vec_best_b0.size) + s0
            xlabel = "Sample index"

        plt.figure(figsize=(10, 5))
        plt.plot(x, corr_vec_best_b0, label=f"Byte0 best 0x{best_guess_b0:02X}")
        if corr_vec_wrong_b0 is not None:
            plt.plot(x, corr_vec_wrong_b0, label=f"Byte0 wrong 0x{wrong_guess_b0:02X}", alpha=0.85)
        plt.axhline(0, color="k", linewidth=0.8)
        plt.grid(True, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel("Correlation")
        plt.title("CPA correlation — Byte 0 (best vs wrong)")
        plt.legend()
        plt.tight_layout()

        if SAVE_PLOT_PATH:
            plt.savefig(SAVE_PLOT_PATH, dpi=150)
        else:
            plt.show()


if __name__ == "__main__":
    main()
