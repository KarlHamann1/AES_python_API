"""
cpa_aes_encryption.py

Perform a simple Correlation Power Analysis (CPA) on AES encryption traces,
then compare the recovered key to a known true key.

Assumptions:
 - We have a CSV "trace_overview.csv" with columns:
     TraceIndex, PlaintextHex, CiphertextHex, TraceFilePath
   in a folder (data_dir).
 - Each row provides:
     - Plaintext in hex (16 bytes)
     - Ciphertext in hex (16 bytes)
     - The path to a .npy file with the recorded trace (1D NumPy array)
 - We do a single-round, first SubBytes model:
     s_i = SBOX[ plaintext[i][byte_idx] ^ key_guess ]
     power_model = HW( s_i )
   Then we correlate this model to the measured trace.
 - We pick the key byte that gives the highest absolute correlation.

Usage:
 1. Adjust `data_dir` and `csv_file` to match your environment.
 2. Run "python cpa_aes_encryption.py".
 3. The script prints out the best guess for each key byte,
    then checks against a known true key if available.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
data_dir = "arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100"   # Folder containing 'trace_overview.csv' + .npy files
csv_file = os.path.join(data_dir, "trace_overview.csv")
max_traces = 10000  # e.g., set to 2000 or None to use all

# Known true key (from #defines). If you have a real known key, place it here (hex string).
# KEY_BYTE_0=0x81, 1=0x55, 2=0x64, 3=0x07, 4=0xE2, 5=0xB7, 6=0xC0, 7=0x90,
# 8=0xD2, 9=0x31, 10=0x46, 11=0xB4, 12=0x6C, 13=0xD5, 14=0x54, 15=0xA2
true_key_hex = "81556407E2B7C090D23146B46CD554A2"

# =====================

# Hamming Weight lookup for 0..255
HW = [bin(x).count("1") for x in range(256)]

# AES S-box for encryption
SBOX = [
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
]

def main():
    # --- Step 1: Read CSV, load plaintexts and traces ---
    plaintexts = []
    ciphertexts = []
    traces = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_hex = row["PlaintextHex"].strip()
            ct_hex = row["CiphertextHex"].strip()
            trace_file = row["TraceFilePath"].strip()

            # Convert to bytes
            pt = bytes.fromhex(pt_hex)
            ct = bytes.fromhex(ct_hex)
            plaintexts.append(pt)
            ciphertexts.append(ct)

            # Load the trace
            trace_path = os.path.join(data_dir, trace_file)
            waveform = np.load(trace_path)
            traces.append(waveform)

            if max_traces is not None and len(traces) >= max_traces:
                break

    num_traces = len(traces)
    if num_traces == 0:
        print(f"No traces loaded from {csv_file}")
        return

    traces = np.array(traces)  # shape: (num_traces, num_samples)
    samples_per_trace = traces.shape[1]

    print(f"Loaded {num_traces} traces, each with {samples_per_trace} samples.")
    print("Starting CPA...")

    # Precompute trace means
    trace_means = np.mean(traces, axis=0)  # shape: (num_samples,)

    # --- Step 2: For each key byte, try all 256 guesses ---
    recovered_key = [0]*16
    for byte_index in range(16):
        best_guess = 0
        best_corr = 0.0

        for guess in range(256):
            # Build hypothetical intermediate for each trace:
            # h[i] = HW( SBOX[ plaintext[i][byte_index] ^ guess ] )
            h = np.array([HW[SBOX[pt[byte_index] ^ guess]] for pt in plaintexts], dtype=np.float64)

            # Pearson correlation approach:
            h_mean = np.mean(h)

            # sum of squares (h - h_mean)^2
            h_diff = h - h_mean
            ss_h = np.sum(h_diff**2)

            t_diff = traces - trace_means
            numerator = np.sum(h_diff[:, None] * t_diff, axis=0)  # shape (num_samples,)
            ss_t = np.sum(t_diff**2, axis=0)

            denom = np.sqrt(ss_h * ss_t)
            corr_vector = np.zeros(samples_per_trace, dtype=np.float64)
            valid_mask = denom > 1e-12
            corr_vector[valid_mask] = numerator[valid_mask] / denom[valid_mask]

            local_max_corr = np.max(np.abs(corr_vector))
            if local_max_corr > best_corr:
                best_corr = local_max_corr
                best_guess = guess

        recovered_key[byte_index] = best_guess
        print(f"Byte {byte_index:2d}: best guess = 0x{best_guess:02X} with correlation={best_corr:.5f}")

    # Summarize the recovered key
    key_hex = "".join(f"{b:02X}" for b in recovered_key)
    print("\n=== Recovered Key (CPA on Round 1 SubBytes) ===")
    print(key_hex)
    print("===============================================\n")

    # Compare to known true key if available
    if len(true_key_hex) == 32:  # 16 bytes in hex
        if key_hex.upper() == true_key_hex.upper():
            print("[✓] Recovered key matches the known true key!")
        else:
            print("[✗] Recovered key does NOT match the known key.")
            print(f"    True key: {true_key_hex.upper()}")
            print(f"    Found   : {key_hex.upper()}")

if __name__ == "__main__":
    main()
