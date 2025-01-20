import os
import glob
import numpy as np
import matplotlib.pyplot as plt

folder_path = "data"

# a list to store traces
all_traces = []

# Loop rounds (1 to 100)
for round_num in range(1, 101):
    # File pattern: The wildcard (*) matches any sequence of characters for the variable part (e.g., date and time)
    file_pattern = os.path.join(folder_path, f"data_round_{round_num}_*.npy")
    
    # Find all files matching the pattern
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"Warning: No files found for round {round_num}.")
        continue

    # Load each file for the current round
    for file_path in file_paths:
        try:
            # Load the data (npy file)
            trace = np.load(file_path)
            if trace.size == 0:
                print(f"Warning: File {file_path} is empty.")
                continue
            all_traces.append(trace)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

# Check if loaded any traces
if not all_traces:
    print("Error: No valid traces loaded. Exiting.")
    exit()

# Convert the list to NumPy array
all_traces = np.array(all_traces)

# Save all traces to a file
all_traces_file = os.path.join(folder_path, "all_traces.npy") # Save in the same directory as the data
np.save(all_traces_file, all_traces)
print(f"All traces saved to {all_traces_file}")

# Check if all traces have the same length
if not all(len(trace) == len(all_traces[0]) for trace in all_traces):
    print("Error: Not all traces have the same length!")
    exit()

# Compute the average trace
average_trace = np.mean(all_traces, axis=0)

# Save the average trace to a file
average_trace_file = os.path.join(folder_path, "averaged_trace.npy") # Save in the same directory as the data
np.save(average_trace_file, average_trace)
print(f"Averaged trace saved to {average_trace_file}")

'''
# Plot the averaged trace
plt.figure()
plt.plot(average_trace, label="Average Trace")
plt.title("Averaged Trace of Side Channel Attack")
plt.xlabel("Sample Points")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.grid()
plt.show()
'''