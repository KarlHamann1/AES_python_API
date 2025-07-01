"""
apply_bandpass_and_average.py

This script:
1. Reads all .npy trace files from a specified directory (e.g., encrypted traces).
2. Prints/plots the frequency response of the bandpass filter being used.
3. Applies the bandpass filter (filtfilt) to each trace in that directory.
4. Averages the filtered traces.
5. Saves the averaged result back to the same directory with a descriptive filename.

Parameters like directory, sampling rate, filter cutoffs, and filter order
are defined at the top.
"""

import os
import glob
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

# ======================= USER PARAMETERS =========================
DIRECTORY = "pi/data_pi_encryption"  # Path where unfiltered trace .npy files reside

SAMPLING_RATE = 62.5e6           # 62.5 MHz
LOW_CUTOFF_HZ = 10e3
HIGH_CUTOFF_HZ = 1000e3
FILTER_ORDER = 2                 # 4th-order Butterworth
LABELNAME = "Avg BPF (10-100 kHz)"

# Output filename for the averaged bandpass-filtered trace
OUTPUT_FILENAME = "averaged_trace_bandpass_10k_100k.npy"
# ================================================================

def main():
    # 1) Collect all .npy files in the directory
    file_paths = glob.glob(os.path.join(DIRECTORY, "*.npy"))
    if not file_paths:
        print(f"No .npy files found in directory: {DIRECTORY}")
        return

    # 2) Prepare the bandpass filter
    nyquist = SAMPLING_RATE / 2.0
    low_norm = LOW_CUTOFF_HZ / nyquist
    high_norm = HIGH_CUTOFF_HZ / nyquist
    b, a = sps.butter(FILTER_ORDER, [low_norm, high_norm], btype='band')

    # --- Print/Plot the filter's frequency response ---
    w, h = sps.freqz(b, a, worN=4096)  # Increase points for smoother curve
    freqs_hz = w * (SAMPLING_RATE / (2.0 * np.pi))  # Convert rad/sample -> Hz

    plt.figure(figsize=(8, 4))
    plt.semilogx(freqs_hz, 20 * np.log10(np.abs(h)), label="Bandpass Filter Response")
    plt.title(f"Butterworth BPF (Order={FILTER_ORDER})\nRange ~ {LOW_CUTOFF_HZ/1e3:.1f} kHz - {HIGH_CUTOFF_HZ/1e6:.1f} MHz")
    plt.xlabel("Frequency [Hz] (log scale)")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.legend()
    # Adjust x-limits to see the lower range more clearly:
    plt.xlim([10, nyquist])  # e.g., from 10 Hz up to ~31.25e6
    plt.tight_layout()
    plt.show()
    # --------------------------------------------------

    filtered_traces = []
    trace_length = None

    # 3) Process each file
    for fpath in file_paths:
        try:
            # Load the trace (1D NumPy array)
            trace = np.load(fpath)
            if trace.size == 0:
                print(f"Warning: {fpath} is empty. Skipping.")
                continue

            # Check length consistency
            if trace_length is None:
                trace_length = len(trace)
            else:
                if len(trace) != trace_length:
                    print(f"Warning: {fpath} has length {len(trace)} "
                        f"which differs from {trace_length}. Skipping.")
                    continue

            # 4) Apply the bandpass filter (forward-backward)
            filtered = sps.filtfilt(b, a, trace)
            filtered_traces.append(filtered)

        except Exception as e:
            print(f"Error loading/filtering {fpath}: {e}")

    if not filtered_traces:
        print("No valid filtered traces to average. Exiting.")
        return

    # 5) Convert to NumPy array and average
    all_filtered = np.array(filtered_traces)
    avg_filtered = np.mean(all_filtered, axis=0)

    # 6) Save the averaged result in the same directory
    out_path = os.path.join(DIRECTORY, OUTPUT_FILENAME)
    np.save(out_path, avg_filtered)
    print(f"\nSaved averaged bandpass-filtered trace to: {out_path}")

    # 7) (Optional) Plot the final averaged trace for a quick look
    plt.figure(figsize=(10, 4))
    plt.plot(avg_filtered, label=LABELNAME)
    plt.title("Averaged Bandpass-Filtered Trace")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (filtered units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
