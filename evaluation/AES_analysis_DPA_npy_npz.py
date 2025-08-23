#!/usr/bin/env python3
"""
DPA (first-order, bit-level) on AES traces with *.npy/*.npz inputs.

What I added vs. the original:
- Repo-aware paths (switch between arduino/pi datasets easily).
- Accepts both .npy and .npz; for NPZ picks 'trace_mean' or 'trace' and tries to read dt_ns.
- CSV compatibility: supports either 'TraceFilePath' (absolute/relative) or 'FileName'.
- Crops all traces to the shortest length so shapes match.
- Optional ROI by samples or by time (if dt is known or overridden).
- Keeps the original “group-by-bit, mean difference, max delta” DPA logic.
- Verifies the recovered key with PyCryptodome AES in ECB mode.
- Saves lightweight per-byte plots (same idea as your original).

I keep comments short and first-person.
"""

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Crypto.Cipher import AES


REPO_ROOT    = Path(__file__).resolve().parents[1]
ARDUINO_DATA = REPO_ROOT / "arduino" / "data"
PI_DATA      = REPO_ROOT / "pi" / "data"

#  config 
USE_PI_DATA       = True
DATASET_DIR_NAME  = "data_pi_40dB_single_encryption_diff"  # e.g. "data_arduino_…"
DATA_DIR          = (PI_DATA if USE_PI_DATA else ARDUINO_DATA) / DATASET_DIR_NAME
CSV_FILE          = DATA_DIR / "trace_overview.csv"        # expects PlaintextHex, CiphertextHex, and a file column

MAX_TRACES        = 6_000      # None = use all
ROI_SAMPLES       = None       # e.g., (start, end) or None
ROI_TIME_S        = None       # e.g., (t0, t1)
DT_NS_OVERRIDE    = None       # float like 32.0 

PLOTS_DIR         = DATA_DIR / "plots_dpa"
SAVE_PER_BYTE_PNG = True


# (first round key equals master key for AES-128
def verify_key(candidate_key: bytes, pt: bytes, ct: bytes) -> bool:
    if len(candidate_key) != 16:
        return False
    aes = AES.new(candidate_key, AES.MODE_ECB)
    return aes.encrypt(pt) == ct

# AES S-box
SBOX = np.array((
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
), dtype=np.uint8)

def _hex16(s: str) -> bytes:
    return bytes.fromhex((s or "").strip().replace(" ", ""))

def _load_waveform(path: Path, dt_ns_holder: dict) -> np.ndarray:
    """
    Load .npy or .npz. For NPZ prefer 'trace_mean' then 'trace'.
    If NPZ has 'dt_ns' and dt_ns_holder["dt"] is None, fill it.
    """
    if path.suffix.lower() == ".npz":
        with np.load(path) as npz:
            if "trace_mean" in npz.files:
                wf = np.asarray(npz["trace_mean"], dtype=np.float32)
            elif "trace" in npz.files:
                wf = np.asarray(npz["trace"], dtype=np.float32)
            else:
                raise ValueError("no 'trace_mean' or 'trace' key in NPZ")
            if dt_ns_holder.get("dt") is None and "dt_ns" in npz.files:
                try:
                    dt_ns_holder["dt"] = float(npz["dt_ns"].item())
                except Exception:
                    pass
        return wf
    else:
        return np.asarray(np.load(path), dtype=np.float32)

def main():
    t_start = time.time()

    if not CSV_FILE.is_file():
        print(f"CSV not found: {CSV_FILE}")
        return

    pts: list[np.ndarray] = []
    cts: list[bytes] = []
    traces: list[np.ndarray] = []
    dt_ns_holder = {"dt": DT_NS_OVERRIDE}  # pick up dt from NPZ/CSV if not overridden

    # read CSV (supports either 'TraceFilePath' or 'FileName')
    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = _hex16(row.get("PlaintextHex", ""))
            ct = _hex16(row.get("CiphertextHex", ""))

            file_token = (row.get("TraceFilePath") or row.get("FileName") or "").strip()
            if not file_token or len(pt) != 16 or len(ct) != 16:
                continue

            p = Path(file_token)
            if not p.is_file():
                p = DATA_DIR / file_token
            if not p.is_file():
                print(f"[warn] trace not found: {file_token}")
                continue

            try:
                wf = _load_waveform(p, dt_ns_holder)
            except Exception as e:
                print(f"[warn] failed to load {p}: {e}")
                continue

            pts.append(np.frombuffer(pt, dtype=np.uint8))
            cts.append(ct)
            traces.append(wf)

            # consider dt_ns from CSV if present
            if dt_ns_holder["dt"] is None:
                dt_str = row.get("dt_ns")
                if dt_str:
                    try:
                        dt_ns_holder["dt"] = float(dt_str)
                    except ValueError:
                        pass

            if MAX_TRACES is not None and len(traces) >= MAX_TRACES:
                break

    if not traces:
        print("No traces loaded.")
        return

    # crop to shortest length so shapes match
    min_len = min(len(w) for w in traces)
    traces = np.stack([w[:min_len] for w in traces], axis=0).astype(np.float32)  # (N, L)
    pts = np.stack(pts, axis=0)                                                  # (N, 16)
    N, L = traces.shape
    print(f"Loaded {N} traces, {L} samples each.")

    # apply ROI if requested
    s0, s1 = 0, L
    if ROI_SAMPLES is not None:
        s0, s1 = max(0, int(ROI_SAMPLES[0])), min(L, int(ROI_SAMPLES[1]))
        traces = traces[:, s0:s1]; L = traces.shape[1]
        print(f"Applied ROI_SAMPLES=({s0},{s1}) → {L} samples.")
    elif ROI_TIME_S is not None and dt_ns_holder["dt"]:
        dt = dt_ns_holder["dt"] * 1e-9
        s0 = max(0, int(round(ROI_TIME_S[0] / dt)))
        s1 = min(L, int(round(ROI_TIME_S[1] / dt)))
        traces = traces[:, s0:s1]; L = traces.shape[1]
        print(f"Applied ROI_TIME_S=({ROI_TIME_S[0]:.3e},{ROI_TIME_S[1]:.3e}) → samples ({s0},{s1}) → {L}.")

    # DPA core (kept the original idea: split by bit of SBOX(PT ^ guess), use max|mean1-mean0|)
    recovered_key = None
    mean_delta_accu_visualization = [np.array([])] * 16  # keep diff for guess=0 like the original
    pt0 = bytes(pts[0].tolist())  # for final verification
    ct0 = cts[0]

    for l in range(8):
        print(f"\nAnalyse für Bitposition {l}:")
        candidate_bytes = []

        for byte_index in range(16):
            delta = np.zeros(256, dtype=np.float32)

            # tight loop; keep the original grouping logic
            for guess in range(256):
                # build group means on-the-fly
                count0 = 0
                count1 = 0
                accu0 = np.zeros(L, dtype=np.float64)
                accu1 = np.zeros(L, dtype=np.float64)

                # partition traces by target bit of SBOX(PT ^ guess)
                pt_b = pts[:, byte_index]
                # v = SBOX[pt_b ^ guess]; target bit:
                v = SBOX[np.bitwise_xor(pt_b, guess)]
                mask1 = ((v >> l) & 1).astype(bool)
                mask0 = ~mask1

                # accumulate means (I could vectorize full means, but I’ll keep the same math)
                if mask0.any():
                    accu0 += traces[mask0].sum(axis=0)
                    count0 = int(mask0.sum())
                if mask1.any():
                    accu1 += traces[mask1].sum(axis=0)
                    count1 = int(mask1.sum())

                if count0 == 0 or count1 == 0:
                    continue

                mean0 = (accu0 / count0).astype(np.float64)
                mean1 = (accu1 / count1).astype(np.float64)
                diff = np.abs(mean1 - mean0)

                delta[guess] = float(np.max(diff))

                # keep a small exemplar like before (diff for guess=0)
                if guess == 0:
                    mean_delta_accu_visualization[byte_index] = diff.astype(np.float32)

            predicted = int(np.argmax(delta))
            candidate_bytes.append(predicted)
            print(f"  Byte {byte_index:2d}: {predicted:02X}")

        candidate_key = bytes(candidate_bytes)
        if verify_key(candidate_key, pt0, ct0):
            print("\nVerifizierter Schlüssel gefunden!")
            recovered_key = candidate_key
            break
        else:
            print("Schlüsselkandidat passt nicht. Weiter…")

    if recovered_key is None:
        print("\nKein gültiger Schlüssel extrahiert.")
    else:
        key_hex = recovered_key.hex().upper()
        print("\nRekonstruierter Schlüssel:", key_hex)
        # save key
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "recovered_key.txt", "w") as f:
            f.write(key_hex)
        print(f"Schlüssel gespeichert → {DATA_DIR/'recovered_key.txt'}")

    # quick per-byte plots
    if SAVE_PER_BYTE_PNG:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        for byte_index in range(16):
            if mean_delta_accu_visualization[byte_index].size == 0:
                continue
            plt.figure()
            plt.plot(mean_delta_accu_visualization[byte_index], label="|mean1 - mean0| (guess=0)")
            plt.title(f"DPA Δ für Schlüsselbyte {byte_index}")
            plt.xlabel("Sample Index" if not (ROI_TIME_S and dt_ns_holder["dt"]) else "Time (s)")
            plt.ylabel("Delta")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"DPA_byte_{byte_index:02d}.png", dpi=120)
            plt.close()

    print(f"\nDone in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
