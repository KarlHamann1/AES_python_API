import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Path & input
    fs = 62.5e6  # 62.5 MHz
    data_dir = "Pi/data_pi_0dB_10MV_block_mult_500_micro_with_300_micro_idle"
    csv_file = os.path.join(data_dir, "trace_overview_mult_block.csv")

    # 2) We know each trace has 50000 samples: first 31250 => mult, next 18750 => idle
    #    If your actual numbers differ, adjust accordingly
    mult_start = 0
    mult_end = 31250  # up to but not including 31250
    idle_start = 31250
    idle_end = 50000  # total 50000 samples

    # 3) Read CSV, load all traces
    mult_minus_idle = []  # will store difference for each trace
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace_filename = row["TraceFilePath"].strip()
            trace_path = os.path.join(data_dir, trace_filename)
            trace = np.load(trace_path)  # shape = (50000,)

            # 4) Compute the mean amplitude in mult region
            mean_mult = np.mean(trace[mult_start:mult_end])

            # 5) Compute the mean amplitude in idle region
            mean_idle = np.mean(trace[idle_start:idle_end])

            # 6) Difference
            diff_val = mean_mult - mean_idle
            mult_minus_idle.append(diff_val)

            # Optional: limit if you have many traces
            # if len(mult_minus_idle) >= 1000:
            #     break

    # 7) Convert to NumPy for easy stats
    mult_minus_idle = np.array(mult_minus_idle)
    print(f"Loaded {len(mult_minus_idle)} traces.")
    avg_diff = np.mean(mult_minus_idle)
    print(f"Average (mean_mult - mean_idle) across all traces: {avg_diff:.6f}")
    print(f"Min/Max difference: {mult_minus_idle.min():.6f} / {mult_minus_idle.max():.6f}")

    # 8) Plot a histogram of differences
    plt.figure(figsize=(8, 5))
    plt.hist(mult_minus_idle, bins=50, alpha=0.7)
    plt.title("Histogram of (mult_mean - idle_mean) per trace")
    plt.xlabel("Difference in amplitude")
    plt.ylabel("Count of traces")
    plt.grid(True)

    # (Optional) Save or show
    plot_path = os.path.join(data_dir, "mult_minus_idle_hist.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Histogram saved to: {plot_path}")

if __name__ == "__main__":
    main()
