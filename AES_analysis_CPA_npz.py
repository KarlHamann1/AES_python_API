#!/usr/bin/env python3
"""
CPA on averaged AES traces stored as NPZ (trace_mean)

Reads CSV from your averaging capture:
BlockIndex, PlaintextHex, CiphertextHex, FileName, Samples, n_avg, dt_ns

For each row, loads NPZ with keys:
trace_mean, plaintext, ciphertext, (optional) dt_ns, sr_nominal, timebase, n_avg

CPA model: first-round SubBytes, HW(SBOX(PT[byte] ^ guess))

NEW:
- Plots the correlation curve for byte 0 for:
    * best guess (maximum |corr|)
    * one wrong guess (auto-chosen)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ----------------- User configuration -----------------
data_dir   = "arduino/data_arduino_1MHz_tb16_4.81Msps_9600Bd_avg100_10ms_20MHzBW_12bit_ACoff4mV"
csv_file   = os.path.join(data_dir, "trace_overview.csv")
max_blocks = 10000            # None = use all rows
true_key_hex = "81556407E2B7C090D23146B46CD554A2"  # optional; "" to disable

# Optional Region Of Interest (choose one or None):
ROI_SAMPLES = None            # e.g., (start_sample, end_sample)
ROI_TIME_S  = None            # e.g., (t0, t1) seconds (needs dt_ns)

# Plot control
SHOW_CORR_PLOT = True         # show correlation curve for byte 0 (best vs wrong)
SAVE_PLOT_PATH = None         # e.g., "corr_byte0.png" or None to just show()

# ------------------------------------------------------

# Hamming Weight lookup
HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

# AES S-box (encryption)
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

def hex_to_bytes16(h: str) -> bytes:
    h = h.strip().replace(" ", "")
    return bytes.fromhex(h)

def main():
    # ---------- Load CSV + NPZ ----------
    pts, cts, traces = [], [], []
    have_dt = False
    dt_ns_global = None

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_hex = row.get("PlaintextHex", "").strip()
            ct_hex = row.get("CiphertextHex", "").strip()
            fname  = row.get("FileName", "").strip()
            if not fname:
                continue

            path = os.path.join(data_dir, fname)
            try:
                npz = np.load(path, allow_pickle=False)
            except Exception as e:
                print(f"[Warn] Could not load {path}: {e}")
                continue

            # waveform (averaged)
            if "trace_mean" in npz.files:
                wf = np.asarray(npz["trace_mean"], dtype=np.float32)
            elif "trace" in npz.files:
                wf = np.asarray(npz["trace"], dtype=np.float32)
            else:
                print(f"[Warn] {path}: no 'trace_mean' or 'trace' key.")
                continue

            # plaintext/ciphertext (prefer CSV; fallback to NPZ)
            pt = hex_to_bytes16(pt_hex) if pt_hex else bytes(npz["plaintext"].astype(np.uint8).tolist())
            ct = hex_to_bytes16(ct_hex) if ct_hex else bytes(npz["ciphertext"].astype(np.uint8).tolist())

            if ROI_TIME_S is not None and "dt_ns" in npz.files:
                dt_ns = float(npz["dt_ns"].item())
                if dt_ns > 0:
                    have_dt = True
                    if dt_ns_global is None:
                        dt_ns_global = dt_ns

            pts.append(pt)
            cts.append(ct)
            traces.append(wf)

            if max_blocks is not None and len(traces) >= max_blocks:
                break

    if not traces:
        print("No traces loaded.")
        return

    # Align lengths
    min_len = min(len(w) for w in traces)
    traces = np.stack([w[:min_len] for w in traces], axis=0).astype(np.float64)  # (N, L)
    N, L = traces.shape
    pts = np.asarray([np.frombuffer(p, dtype=np.uint8) for p in pts], dtype=np.uint8)  # (N, 16)

    print(f"Loaded {N} traces, {L} samples each.")

    # ---------- ROI ----------
    if ROI_SAMPLES is not None:
        s0, s1 = ROI_SAMPLES
        s0 = max(0, int(s0)); s1 = min(L, int(s1))
        traces = traces[:, s0:s1]; L = traces.shape[1]
        print(f"Applied ROI_SAMPLES = ({s0}, {s1}) → {L} samples.")
    elif ROI_TIME_S is not None and have_dt and dt_ns_global and dt_ns_global > 0:
        t0, t1 = ROI_TIME_S
        dt = dt_ns_global * 1e-9
        s0 = max(0, int(round(t0 / dt)))
        s1 = min(L, int(round(t1 / dt)))
        traces = traces[:, s0:s1]; L = traces.shape[1]
        print(f"Applied ROI_TIME_S = ({t0:.6e}, {t1:.6e}) s → samples ({s0}, {s1}) → {L} samples.")
    else:
        s0, s1 = 0, L  # for labeling later

    # ---------- Precompute centered traces ----------
    trace_means = np.mean(traces, axis=0)       # (L,)
    t_diff = traces - trace_means               # (N,L)
    ss_t = np.sum(t_diff * t_diff, axis=0)      # (L,)

    recovered_key = np.zeros(16, dtype=np.uint8)
    best_corrs    = np.zeros(16, dtype=np.float64)

    # For plotting (byte 0)
    corr_vec_best_b0 = None
    corr_vec_wrong_b0 = None
    best_guess_b0 = None
    wrong_guess_b0 = None

    for byte_index in range(16):
        best_g = 0
        best_c = 0.0
        corr_vec_best = None

        # CPA loop over guesses
        for g in range(256):
            h = HW[SBOX[pts[:, byte_index] ^ g]].astype(np.float64)  # (N,)
            h_diff = h - h.mean()
            ss_h   = np.dot(h_diff, h_diff)
            if ss_h <= 1e-18:
                continue

            numerator = h_diff @ t_diff
            denom = np.sqrt(ss_h * ss_t)
            valid = denom > 1e-18
            corr_vec = np.zeros_like(numerator)
            corr_vec[valid] = numerator[valid] / denom[valid]

            local_max = np.max(np.abs(corr_vec))
            if local_max > best_c:
                best_c = local_max
                best_g = g
                corr_vec_best = corr_vec  # keep the best curve for this byte

        recovered_key[byte_index] = best_g
        best_corrs[byte_index] = best_c
        print(f"Byte {byte_index:2d}: guess=0x{best_g:02X}, |corr|max={best_c:.6f}")

        # Store curves for byte 0
        if byte_index == 0 and corr_vec_best is not None:
            best_guess_b0 = best_g
            corr_vec_best_b0 = corr_vec_best

            # choose a wrong guess: 0x00 unless it equals best; else 0x01
            wg = 0x00 if best_g != 0x00 else 0x01
            wrong_guess_b0 = wg

            # compute wrong curve once
            h_w = HW[SBOX[pts[:, 0] ^ wg]].astype(np.float64)
            h_w_diff = h_w - h_w.mean()
            ss_h_w = np.dot(h_w_diff, h_w_diff)
            if ss_h_w > 1e-18:
                num_w = h_w_diff @ t_diff
                den_w = np.sqrt(ss_h_w * ss_t)
                corr_w = np.zeros_like(num_w)
                mask_w = den_w > 1e-18
                corr_w[mask_w] = num_w[mask_w] / den_w[mask_w]
                corr_vec_wrong_b0 = corr_w

    key_hex = "".join(f"{b:02X}" for b in recovered_key)
    print("\n=== Recovered Key (CPA, Round-1 SubBytes) ===")
    print(key_hex)
    print("=============================================\n")

    if len(true_key_hex) == 32:
        if key_hex.upper() == true_key_hex.upper():
            print("[✓] Recovered key matches the known true key!")
        else:
            print("[✗] Recovered key does NOT match the known key.")
            print(f"    True key: {true_key_hex.upper()}")
            print(f"    Found   : {key_hex.upper()}")

    # ---------- Plot correlation curve for byte 0 (best vs wrong) ----------
    if SHOW_CORR_PLOT and corr_vec_best_b0 is not None:
        # x-axis: time if dt known, else samples
        if ROI_TIME_S is not None and have_dt and dt_ns_global and dt_ns_global > 0:
            dt = dt_ns_global * 1e-9
            t = np.arange(corr_vec_best_b0.shape[0]) * dt + s0*dt
            x = t
            xlabel = "Time (s)"
        else:
            x = np.arange(corr_vec_best_b0.shape[0]) + s0
            xlabel = "Sample index"

        plt.figure(figsize=(10, 5))
        plt.plot(x, corr_vec_best_b0, label=f"Byte0 best guess 0x{best_guess_b0:02X}")
        if corr_vec_wrong_b0 is not None:
            plt.plot(x, corr_vec_wrong_b0, label=f"Byte0 wrong guess 0x{wrong_guess_b0:02X}", alpha=0.8)
        plt.axhline(0, color="k", linewidth=0.8)
        plt.grid(True, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel("Correlation")
        plt.title("CPA correlation curves – Byte 0 (best vs wrong)")
        plt.legend()
        plt.tight_layout()

        if SAVE_PLOT_PATH:
            plt.savefig(SAVE_PLOT_PATH, dpi=150)
        else:
            plt.show()

if __name__ == "__main__":
    main()
