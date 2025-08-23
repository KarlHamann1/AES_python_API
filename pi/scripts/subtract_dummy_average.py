#!/usr/bin/env python3
"""
subtract_dummy_average.py

What this does:
1) load the averaged dummy waveform
2) iterate over all AES trace files
3) subtract the dummy average from each AES trace
4) save the result into a separate folder (same filenames)
"""

import os
import glob
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] 
PI_DATA   = REPO_ROOT / "pi" / "data"

# folder names inside pi/data (change if needed)
DUMMY_DIR_NAME = "data_pi_40dB_block_dummy"
AES_DIR_NAME   = "data_pi_40dB_single_encryption"
OUT_DIR_NAME   = "data_pi_40dB_single_encryption_diff"

# build full paths
DUMMY_AVG_FILE = PI_DATA / DUMMY_DIR_NAME / "averaged_trace_dummy.npy"
AES_TRACES_DIR = PI_DATA / AES_DIR_NAME
OUT_DIFF_DIR   = PI_DATA / OUT_DIR_NAME


def main():
    # 1) load the dummy average
    if not DUMMY_AVG_FILE.is_file():
        print(f"Error: dummy average file not found: {DUMMY_AVG_FILE}")
        return
    dummy_avg = np.load(DUMMY_AVG_FILE)
    dummy_length = len(dummy_avg)
    print(f"Loaded dummy average from {DUMMY_AVG_FILE} (length={dummy_length} samples).")

    # 2) ensure output directory exists
    OUT_DIFF_DIR.mkdir(parents=True, exist_ok=True)

    # 3) find all AES .npy files
    aes_files = sorted(glob.glob(str(AES_TRACES_DIR / "*.npy")))
    if not aes_files:
        print(f"No AES .npy files found in {AES_TRACES_DIR}")
        return

    # 4) subtract dummy from each AES trace and save
    count_processed = 0
    for aes_path in aes_files:
        aes_path = Path(aes_path)
        trace_name = aes_path.name
        out_path = OUT_DIFF_DIR / trace_name

        try:
            aes_trace = np.load(aes_path)
            if len(aes_trace) != dummy_length:
                print(f"Warning: {trace_name} length={len(aes_trace)} != dummy_length={dummy_length}. Skipping.")
                continue

            # element-wise subtraction
            diff_trace = aes_trace - dummy_avg

            # save result
            np.save(out_path, diff_trace)
            count_processed += 1

        except Exception as e:
            print(f"Error processing {trace_name}: {e}")

    print(f"\nDone. Created {count_processed} diff files in {OUT_DIFF_DIR}.")


if __name__ == "__main__":
    main()
