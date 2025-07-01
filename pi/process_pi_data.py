import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Folders containing the .npy files (change as needed)
enc_folder = "pi/data_pi_40dB_block_mult_500_micro_with_300_micro_idle"  # AES captures
dummy_folder = "data_pi_0dB_dummytt"     # Dummy captures

def load_and_average_traces(folder_path, label):
    """
    Attempts to load all .npy files in 'folder_path' as 1D NumPy arrays,
    and returns (average_trace, trace_length).

    If the folder doesn't exist or has no valid files,
    returns (None, None).

    'label' is a string used only for print messages (e.g. "AES" or "Dummy").
    """
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist. Skipping {label}.")
        return None, None

    file_paths = glob.glob(os.path.join(folder_path, "*.npy"))
    if not file_paths:
        print(f"No .npy files found in folder: {folder_path}. Skipping {label}.")
        return None, None

    all_traces = []
    for fpath in file_paths:
        try:
            trace = np.load(fpath)
            if trace.size == 0:
                print(f"Warning: File {fpath} is empty. Skipping.")
                continue
            all_traces.append(trace)
        except Exception as e:
            print(f"Error loading file {fpath}: {e}")

    if not all_traces:
        print(f"No valid {label} traces loaded from {folder_path}.")
        return None, None

    # Convert to numpy array
    all_traces_arr = np.array(all_traces)

    # Check if all have the same length
    length0 = len(all_traces_arr[0])
    for t in all_traces_arr:
        if len(t) != length0:
            print(f"Error: Not all {label} traces in {folder_path} have the same length!")
            return None, None

    # Compute average
    avg = np.mean(all_traces_arr, axis=0)

    # Save average
    out_path = os.path.join(folder_path, f"averaged_trace_{label.lower()}.npy")
    np.save(out_path, avg)
    print(f"{label} average saved to {out_path}")

    return avg, length0


def main():
    # 1) Try loading & averaging AES traces
    average_enc, enc_length = load_and_average_traces(enc_folder, label="AES")

    # 2) Try loading & averaging Dummy traces
    average_dummy, dummy_length = load_and_average_traces(dummy_folder, label="Dummy")

    # If both are None, no data loaded => exit
    if average_enc is None and average_dummy is None:
        print("\nNo AES or Dummy data was loaded. Exiting.")
        return

    # If only AES or only Dummy was loaded, just plot that one
    if average_enc is not None and average_dummy is None:
        plt.figure()
        plt.plot(average_enc, label="Average AES Trace")
        plt.title("Averaged AES Trace (Dummy not found)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    if average_dummy is not None and average_enc is None:
        plt.figure()
        plt.plot(average_dummy, label="Average Dummy Trace")
        plt.title("Averaged Dummy Trace (AES not found)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    # If we have both, check lengths
    if enc_length != dummy_length:
        print(f"Error: AES trace length ({enc_length}) != Dummy trace length ({dummy_length}). Cannot plot both.")
        return

    # 3) Plot both averages
    plt.figure()
    plt.plot(average_enc, label="Average AES Trace")
    plt.plot(average_dummy, label="Average Dummy Trace")
    plt.title("Averaged Traces: AES vs. Dummy")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
