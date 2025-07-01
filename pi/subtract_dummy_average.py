"""
subtract_dummy_average.py

This script:
1. Loads the averaged dummy waveform from "pi/data_pi_40dB_block_dummy/averaged_trace_dummy.npy"
2. Iterates over all AES trace files (e.g., 10k) in "data_pi_40dB_single_encryption/"
3. Subtracts the dummy average from each AES trace
4. Saves the result to a new folder "data_pi_40dB_single_encryption_diff/" (one .npy per trace)
"""

import os
import glob
import numpy as np

# Paths (change if needed)
DUMMY_AVG_DIR = "pi/data_pi_40dB_block_dummy"
DUMMY_AVG_FILE = os.path.join(DUMMY_AVG_DIR, "averaged_trace_dummy.npy")

AES_TRACES_DIR = "pi/data_pi_40dB_single_encryption"
OUT_DIFF_DIR = "pi/data_pi_40dB_single_encryption_diff"

def main():
    # 1) Load the dummy average
    if not os.path.isfile(DUMMY_AVG_FILE):
        print(f"Error: Dummy average file not found: {DUMMY_AVG_FILE}")
        return
    dummy_avg = np.load(DUMMY_AVG_FILE)
    dummy_length = len(dummy_avg)
    print(f"Loaded dummy average from {DUMMY_AVG_FILE} (length={dummy_length} samples).")

    # 2) Ensure output directory exists
    os.makedirs(OUT_DIFF_DIR, exist_ok=True)

    # 3) Find all AES .npy files
    aes_files = glob.glob(os.path.join(AES_TRACES_DIR, "*.npy"))
    if not aes_files:
        print(f"No AES .npy files found in {AES_TRACES_DIR}")
        return

    # 4) For each AES file, subtract dummy_avg => save to new folder
    count_processed = 0
    for aes_path in aes_files:
        trace_name = os.path.basename(aes_path)  # e.g., "aes_trace_block_123.npy"
        out_path = os.path.join(OUT_DIFF_DIR, trace_name)  # keep same name or adjust if desired

        try:
            aes_trace = np.load(aes_path)
            if len(aes_trace) != dummy_length:
                print(f"Warning: {trace_name} length={len(aes_trace)} != dummy_length={dummy_length}. Skipping.")
                continue

            # Subtract element-wise
            diff_trace = aes_trace - dummy_avg

            # Save
            np.save(out_path, diff_trace)
            count_processed += 1

        except Exception as e:
            print(f"Error processing {trace_name}: {e}")

    print(f"\nDone. Created {count_processed} diff files in {OUT_DIFF_DIR}.")

if __name__ == "__main__":
    main()
