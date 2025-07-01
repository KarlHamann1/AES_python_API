#!/usr/bin/env python3

"""
filter_single_trace_highpass.py

1) Reads a single .npy trace file (FILE_PATH).
2) Prints/plots the frequency response of a high-pass filter 
   using the cutoff: HIGH_PASS_CUTOFF_HZ.
3) Applies a forward-backward high-pass filter (filtfilt) to the trace.
4) Saves the filtered result as a new .npy file (OUTPUT_FILENAME) — 
   automatically named based on the cutoff frequency.
5) Optionally plots the filtered trace for a quick look.
"""

import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

# ======================= USER PARAMETERS =========================
FILE_PATH            = "pi/data_pi_0dB_10MV_block_mult_500_micro_with_300_micro_idle/mult_trace_block_0.npy"     # Path to your .npy trace
SAMPLING_RATE        = 62.5e6             # 62.5 MHz
HIGH_PASS_CUTOFF_HZ  = 5e6                # 5 MHz
FILTER_ORDER         = 2                  # e.g., 2nd order Butterworth
# ================================================================

def main():
    # Build label & output filename from the cutoff
    if HIGH_PASS_CUTOFF_HZ < 1e6:
        label_cutoff = f"{HIGH_PASS_CUTOFF_HZ/1e3:.1f}kHz"
    else:
        label_cutoff = f"{HIGH_PASS_CUTOFF_HZ/1e6:.1f}MHz"

    output_filename = f"highpass_{label_cutoff}.npy"
    filter_label    = f"Highpass ({label_cutoff})"

    # 1) Load the single trace
    if not os.path.isfile(FILE_PATH):
        print(f"Error: could not find the file '{FILE_PATH}'")
        return

    trace = np.load(FILE_PATH)
    if trace.size == 0:
        print(f"Error: '{FILE_PATH}' is empty. Aborting.")
        return

    print(f"Loaded trace from: {FILE_PATH}")
    print(f"Trace length: {len(trace)} samples")

    # 2) Create the high-pass filter
    nyquist = SAMPLING_RATE / 2.0
    high_norm = HIGH_PASS_CUTOFF_HZ / nyquist

    b, a = sps.butter(FILTER_ORDER, high_norm, btype='high')

    # --- Print/Plot the filter's frequency response ---
    w, h = sps.freqz(b, a, worN=4096)
    freqs_hz = w * (SAMPLING_RATE / (2.0 * np.pi))

    # Avoid log(0) by adding a small offset
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-20)

    plt.figure()
    plt.semilogx(freqs_hz, magnitude_db, label="Highpass Filter Response")
    plt.title(f"Butterworth Highpass (Order={FILTER_ORDER})\nCutoff ~ {label_cutoff}")
    plt.xlabel("Frequency [Hz] (log scale)")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.legend()
    plt.xlim([10, nyquist])
    plt.tight_layout()
    plt.show()
    # --------------------------------------------------

    # 3) Apply the high-pass filter (forward-backward)
    print("Applying high-pass filter to the trace...")
    filtered = sps.filtfilt(b, a, trace)

    # 4) Save the filtered result
    np.save(output_filename, filtered)
    print(f"Saved filtered trace to: {output_filename}")

    # 5) Plot the final filtered trace (optional)
    plt.figure()
    plt.plot(filtered, label=filter_label)
    plt.title("Filtered Trace (Highpass)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
