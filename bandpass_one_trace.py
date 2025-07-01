#!/usr/bin/env python3

"""
filter_single_trace.py

1) Reads a single .npy trace file (FILE_PATH).
2) Prints/plots the frequency response of the bandpass filter 
    using the cutoffs: LOW_CUTOFF_HZ and HIGH_CUTOFF_HZ.
3) Applies a forward-backward bandpass filter (filtfilt) to that one trace.
4) Saves the filtered result as a new .npy file (OUTPUT_FILENAME) — 
    automatically named based on the cutoff frequencies.
5) Optionally plots the filtered trace for a quick look.
"""

import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

# ======================= USER PARAMETERS =========================
FILE_PATH      = "pi/data_pi_0dB_10MV_block_mult_500_micro_with_300_micro_idle/mult_trace_block_0.npy"

SAMPLING_RATE  = 62.5e6     # 62.5 MHz
LOW_CUTOFF_HZ  = 2e3        # 2 kHz
HIGH_CUTOFF_HZ = 5e6        # 5 MHz
FILTER_ORDER   = 2          # Butterworth order
# ================================================================

def main():
    # Create a descriptive label for the bandpass filter based on the cutoff frequencies
    label_low  = (f"{LOW_CUTOFF_HZ/1e3:.1f}kHz" if LOW_CUTOFF_HZ < 1e6 else f"{LOW_CUTOFF_HZ/1e6:.1f}MHz")
    label_high = (f"{HIGH_CUTOFF_HZ/1e3:.1f}kHz" if HIGH_CUTOFF_HZ < 1e6 else f"{HIGH_CUTOFF_HZ/1e6:.1f}MHz")
    
    # Construct the output filename and label from the cutoffs
    output_filename = f"bandpass_{label_low}_{label_high}.npy"
    filter_label    = f"Bandpass ({label_low} – {label_high})"
    
    # 1) Load the single trace (1D NumPy array)
    if not os.path.isfile(FILE_PATH):
        print(f"Error: could not find the file '{FILE_PATH}'")
        return
    
    trace = np.load(FILE_PATH)
    if trace.size == 0:
        print(f"Error: '{FILE_PATH}' is empty. Aborting.")
        return
    
    print(f"Loaded trace from: {FILE_PATH}")
    print(f"Trace length: {len(trace)} samples")

    # 2) Create the bandpass filter
    nyquist = SAMPLING_RATE / 2.0
    low_norm = LOW_CUTOFF_HZ / nyquist
    high_norm = HIGH_CUTOFF_HZ / nyquist
    
    b, a = sps.butter(FILTER_ORDER, [low_norm, high_norm], btype='band')

    # --- Print/Plot the filter's frequency response ---
    w, h = sps.freqz(b, a, worN=4096)
    freqs_hz = w * (SAMPLING_RATE / (2.0 * np.pi))  # Convert rad/sample -> Hz
    
    # Avoid log(0) by adding a small offset
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-20)

    plt.figure()
    plt.semilogx(freqs_hz, magnitude_db, label="Bandpass Filter Response")
    plt.title(
        f"Butterworth BPF (Order={FILTER_ORDER})\n"
        f"Range ~ {LOW_CUTOFF_HZ/1e3:.1f} kHz - {HIGH_CUTOFF_HZ/1e6:.1f} MHz"
    )
    plt.xlabel("Frequency [Hz] (log scale)")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.legend()
    plt.xlim([10, nyquist])
    plt.tight_layout()
    plt.show()
    # --------------------------------------------------
    
    # 3) Apply the bandpass filter (forward-backward)
    print("Applying bandpass filter to the trace...")
    filtered = sps.filtfilt(b, a, trace)

    # 4) Save the filtered result
    np.save(output_filename, filtered)
    print(f"Saved filtered trace to: {output_filename}")

    # 5) Plot the final filtered trace (optional)
    plt.figure()
    plt.plot(filtered, label=filter_label)
    plt.title("Filtered Trace (Bandpass)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
