import numpy as np
import matplotlib.pyplot as plt
import time

########################################
# User Configuration
########################################

# Choose your plotting mode:
#  - 1 => single file
#  - 2 => two files in one plot
PLOT_MODE = 1

# If using single-file mode, set this path:
#FILEPATH_SINGLE = "pi/data_pi_30dB_10MV_block_mult_500_micro_with_300_micro_idle/mult_trace_block_0.npy"
FILEPATH_SINGLE = "arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000000.npz"

# If using two-file mode, set these two paths:
FILEPATH_ONE = "pi/data_pi_40dB_single_encryption/aes_trace_round_0.npy"
FILEPATH_TWO = "pi/data_pi_40dB_single_encryption/aes_trace_round_1.npy"

########################################
# Utility Functions
########################################

def load_trace(file_path):
    """
    Loads data from a .npy file.
    
    Two possible formats:
    1) A simple NumPy array (e.g., your averaged_trace.npy).
        In this case, we return (waveform, None) 
        because there's no metadata.
    2) A dictionary with {"data": <array>, "metadata": <dict>},
        (the older example with metadata).
        Then we return (waveform, metadata_dict).
    """
    try:
        raw = np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Could not load {file_path}: {e}")

    # 1) If it's a dictionary with a 'data' key
    #    (like your old 'visualize_data' script),
    #    we expect raw.item() to be a dict with "data" and "metadata".
    #    We'll attempt this conditionally:
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        # Possibly an object array that might contain a dict.
        maybe_dict = raw.item()
        if isinstance(maybe_dict, dict) and "data" in maybe_dict:
            # It's the old format with 'data' and 'metadata'.
            waveform = maybe_dict["data"]
            metadata = maybe_dict.get("metadata", {})
            return waveform, metadata
        else:
            # It's just a weird object array. We'll treat it as a normal array.
            return raw, None
    elif isinstance(raw, np.ndarray):
        # It's a normal NumPy array (the average trace scenario).
        return raw, None
    else:
        # Possibly a dictionary if we used np.save with allow_pickle=True
        if isinstance(raw, dict) and "data" in raw:
            waveform = raw["data"]
            metadata = raw.get("metadata", {})
            return waveform, metadata
        else:
            raise ValueError(
                f"File {file_path} did not match expected formats (array or dict)."
            )

def plot_single_trace(waveform, metadata=None, label="Trace"):
    """
    Plot a single waveform. If metadata is given in the old format,
    we'll attempt to get a time axis from 'duration' or show a fallback x-axis.
    """
    # If there's metadata with a "duration" and the waveform length, 
    # we can create a time axis. Otherwise, just plot sample index.
    if metadata and "duration" in metadata and "sampling_rate" in metadata:
        duration = metadata["duration"]
        n = len(waveform)
        time_vector = np.linspace(0, duration, n)
        plt.plot(time_vector, waveform, label=label)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        title_str = f"{label}"
        # If there's start_time and end_time, show them in the title
        if "start_time" in metadata and "end_time" in metadata:
            try:
                start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata["start_time"]))
                end_str   = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata["end_time"]))
                title_str += f"\nStart: {start_str}, End: {end_str}"
            except:
                pass
        plt.title(title_str)
    else:
        # No metadata or no duration => plot by sample index
        plt.plot(waveform, label=label)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

def visualize_single_file(file_path):
    """
    Load one file, parse it, and plot it.
    """
    waveform, metadata = load_trace(file_path)

    plt.figure(figsize=(10, 5))
    plot_single_trace(waveform, metadata, label=f"File: {file_path}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # If metadata is present, print a summary:
    if metadata:
        print("\n=== Metadata Summary ===")
        for key, value in metadata.items():
            if key.endswith("_time"):
                try:
                    value = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))
                except:
                    pass
            print(f"{key}: {value}")

def visualize_two_files(file_path1, file_path2):
    """
    Load two files, parse them, and plot them on the same figure.
    (Commonly used to compare e.g. AES average vs. Dummy average.)
    """
    waveform1, metadata1 = load_trace(file_path1)
    waveform2, metadata2 = load_trace(file_path2)

    # Basic check that waveforms have the same length (for nicer comparison):
    if len(waveform1) != len(waveform2):
        print(f"Warning: The two waveforms have different lengths: "
            f"{len(waveform1)} vs. {len(waveform2)}")

    plt.figure(figsize=(10, 5))
    plot_single_trace(waveform1, metadata1, label=f"File1: {file_path1}")
    plot_single_trace(waveform2, metadata2, label=f"File2: {file_path2}")
    plt.title("Comparing Two Waveforms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally print metadata from both:
    if metadata1:
        print("\n=== Metadata Summary (File1) ===")
        for key, value in metadata1.items():
            if key.endswith("_time"):
                try:
                    value = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))
                except:
                    pass
            print(f"{key}: {value}")

    if metadata2:
        print("\n=== Metadata Summary (File2) ===")
        for key, value in metadata2.items():
            if key.endswith("_time"):
                try:
                    value = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))
                except:
                    pass
            print(f"{key}: {value}")

########################################
# Main Execution
########################################

def main():
    # Just pick the mode from the user config at the top:
    if PLOT_MODE == 1:
        # Single file
        visualize_single_file(FILEPATH_SINGLE)

    elif PLOT_MODE == 2:
        # Two files
        visualize_two_files(FILEPATH_ONE, FILEPATH_TWO)
    else:
        print("Invalid PLOT_MODE. Please set PLOT_MODE=1 or 2.")

if __name__ == "__main__":
    main()
