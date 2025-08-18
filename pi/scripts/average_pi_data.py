#!/usr/bin/env python3
"""
Load *.npy traces from two folders, average each set, and (optionally) plot them.

Notes:
- Paths are now repo-aware: AES_PYTHON_API/pi/data/<DIR_NAME>
- Logic kept the same (just cleaned up a little + friendlier prints).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


REPO_ROOT   = Path(__file__).resolve().parents[2]  
PI_DATA_DIR = REPO_ROOT / "pi" / "data"

# Folders containing the .npy files (EDIT THESE TWO NAMES as needed)
ENC_DIR_NAME   = "data_pi_40dB_block_mult_500_micro_with_300_micro_idle"  # AES captures
DUMMY_DIR_NAME = "data_pi_0dB_dummytt"                                     # Dummy captures

# Build absolute folders
enc_folder   = PI_DATA_DIR / ENC_DIR_NAME
dummy_folder = PI_DATA_DIR / DUMMY_DIR_NAME


def load_and_average_traces(folder_path, label):
    """
    Try to load all .npy files in 'folder_path' as 1D arrays.
    Returns (average_trace, trace_length) or (None, None)
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print(f"[Info] Folder '{folder_path}' does not exist. Skipping {label}.")
        return None, None

    file_paths = glob.glob(str(folder_path / "*.npy"))
    if not file_paths:
        print(f"[Info] No .npy files in: {folder_path}. Skipping {label}.")
        return None, None

    all_traces = []
    for fpath in file_paths:
        try:
            trace = np.load(fpath)
            if trace.size == 0:
                print(f"[Warn] Empty file: {fpath} — skipped.")
                continue
            all_traces.append(trace)
        except Exception as e:
            print(f"[Warn] Failed to load {fpath}: {e}")

    if not all_traces:
        print(f"[Info] No valid {label} traces loaded from {folder_path}.")
        return None, None


    length0 = len(all_traces[0])
    for t in all_traces:
        if len(t) != length0:
            print(f"[Error] Not all {label} traces in {folder_path} have the same length!")
            return None, None

    # Average and save next to the input folder
    avg = np.mean(np.array(all_traces), axis=0)
    out_path = folder_path / f"averaged_trace_{label.lower()}.npy"
    np.save(out_path, avg.astype(np.float32))
    print(f"[OK] {label} average saved → {out_path}")

    return avg, length0


def main():
    # 1) AES traces
    average_enc, enc_length = load_and_average_traces(enc_folder, label="AES")

    # 2) Dummy traces
    average_dummy, dummy_length = load_and_average_traces(dummy_folder, label="Dummy")

    if average_enc is None and average_dummy is None:
        print("\nNo AES or Dummy data was loaded. Exiting.")
        return

    if average_enc is not None and average_dummy is None:
        plt.figure()
        plt.plot(average_enc, label="Average AES Trace")
        plt.title("Averaged AES Trace (Dummy not found)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
        return

    if average_dummy is not None and average_enc is None:
        plt.figure()
        plt.plot(average_dummy, label="Average Dummy Trace")
        plt.title("Averaged Dummy Trace (AES not found)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
        return

    if enc_length != dummy_length:
        print(f"[Error] AES length ({enc_length}) != Dummy length ({dummy_length}). Can't plot both.")
        return

    # 3) Plot both averages together
    plt.figure()
    plt.plot(average_enc,   label="Average AES Trace")
    plt.plot(average_dummy, label="Average Dummy Trace")
    plt.title("Averaged Traces: AES vs. Dummy")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
