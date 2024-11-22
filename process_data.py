import numpy as np

# Load data file
data_file = "data/data_round_1_20231122_143045.npy"
data_dict = np.load(data_file, allow_pickle=True).item()

# Access waveform data
waveform = data_dict["data"]

# Access metadata
metadata = data_dict["metadata"]
start_time = metadata["start_time"]
end_time = metadata["end_time"]

print("Waveform:", waveform[:5])  # Show first 5 points
print("Metadata:", metadata)
