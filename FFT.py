import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Define the sampling rate of your traces (62.5 MHz in your case).
    fs = 62.5e6  # 62.5 MHz

    # 2) Paths for your data
    data_dir = "Pi/data_pi_40dB_block_mult_500_micro"
    csv_file = os.path.join(data_dir, "trace_overview_mult_block.csv")

    # 3) Read your CSV and load all traces
    traces = []
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace_filename = row["TraceFilePath"].strip()
            trace_path = os.path.join(data_dir, trace_filename)
            trace = np.load(trace_path)  # shape = (num_samples,)
            traces.append(trace)

            # OPTIONAL: limit to a certain number of traces
            if len(traces) >= 10000:
                break

    if len(traces) == 0:
        print("No traces found. Exiting.")
        return

    # Convert from list of 1D arrays to a 2D NumPy array (num_traces x samples_per_trace)
    traces = np.array(traces)  # shape = (N, M)
    print(f"Loaded {len(traces)} traces with shape {traces.shape}.")

    # 4) Compute the average trace over all N traces
    avg_trace = np.mean(traces, axis=0)  # shape = (M,)

    # 5) Compute FFT of the average trace
    fft_values = np.fft.fft(avg_trace)
    freq_bins = np.fft.fftfreq(len(avg_trace), d=1/fs)

    # 6) Convert to magnitude (and optionally dB)
    mag = np.abs(fft_values)
    mag_db = 20 * np.log10(mag + 1e-12)  # add a tiny offset to avoid log(0)

    # 7) Plot only the positive frequencies (0..fs/2)
    pos_mask = (freq_bins >= 0)
    freqs_pos = freq_bins[pos_mask]
    mag_db_pos = mag_db[pos_mask]

    # 8) Further limit to up to 10 MHz
    up_to_10mhz_mask = (freqs_pos <= 10e6)
    freqs_limited = freqs_pos[up_to_10mhz_mask]
    mag_db_limited = mag_db_pos[up_to_10mhz_mask]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs_limited, mag_db_limited, label="Average Trace FFT (up to 10 MHz)")
    plt.title("Frequency Spectrum of Average Trace (0–10 MHz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True)

    # 9) Save or show the plot
    plot_path = os.path.join(data_dir, "AverageTrace_FFT_UpTo10MHz.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"FFT plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
