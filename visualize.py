import numpy as np
import matplotlib.pyplot as plt

def visualize_data(filepath):
    """
    Visualize waveform data and timestamps from a .npy file.
    """
    # Load the file
    data_dict = np.load(filepath, allow_pickle=True).item()
    waveform = data_dict["data"]
    metadata = data_dict["metadata"]

    # Extract metadata
    start_time = metadata["start_time"]
    end_time = metadata["end_time"]
    sampling_rate = metadata["sampling_rate"]
    duration = metadata["duration"]

    # Generate time vector
    time_vector = np.linspace(0, duration, len(waveform))

    # Plot waveform
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, waveform, label="Waveform", linewidth=1)
    plt.title(f"Waveform Data\nStart: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}, " f"End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

    # Print metadata summary
    print("\n=== Metadata Summary ===")
    for key, value in metadata.items():
        if key.endswith("_time"):
            value = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))
        print(f"{key}: {value}")


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <file_path>")
    else:
        filepath = sys.argv[1]
        visualize_data(filepath)
        
        # Allows you to specify the file to visualize directly from the command line:
        # python visualize.py data/data_round_1_20231122_143045.npy

