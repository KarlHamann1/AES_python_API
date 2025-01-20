import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to the directories
folder_path_encryption = "data"
folder_path_no_encryption = "data_no_encryption"

# File names for the averaged traces
average_trace_file_encryption = os.path.join(folder_path_encryption, "averaged_trace.npy")
average_trace_file_no_encryption = os.path.join(folder_path_no_encryption, "averaged_trace.npy")

# Load the averaged traces
try:
    average_trace_encryption = np.load(average_trace_file_encryption)
    average_trace_no_encryption = np.load(average_trace_file_no_encryption)
except Exception as e:
    print(f"Error loading averaged traces: {e}")
    exit()

# Check if the traces have the same length
if len(average_trace_encryption) != len(average_trace_no_encryption):
    print("Error: Averaged traces do not have the same length.")
    exit()

# difference between the traces
difference_trace = average_trace_encryption - average_trace_no_encryption

# statistical metrics
mean_absolute_difference = np.mean(np.abs(difference_trace))
root_mean_square_error = np.sqrt(np.mean(difference_trace**2))
max_absolute_difference = np.max(np.abs(difference_trace))

# statistical metrics
print("Statistical Analysis of Differences:")
print(f"Mean Absolute Difference (MAD): {mean_absolute_difference}") # is the average of the absolute differences between the predicted values and the actual values
print(f"Root Mean Square Error (RMSE): {root_mean_square_error}") # is the square root of the average of the squared differences between the predicted values and the actual values
print(f"Max Absolute Difference: {max_absolute_difference}") # is the maximum absolute difference between the predicted values and the actual values

# Plot both traces
plt.figure(figsize=(10, 6))
plt.plot(average_trace_encryption, label="Average Trace (Encryption)", linestyle="--")
plt.plot(average_trace_no_encryption, label="Average Trace (No Encryption)", linestyle="-")
plt.title("Comparison of Averaged Traces")
plt.xlabel("Sample Points")
plt.ylabel("Signal Amplitude")
plt.legend(loc = "lower right")
plt.grid()
plt.show()

'''
# Optionally save the difference trace
output_difference_file = "difference_trace.npy"
np.save(output_difference_file, difference_trace)
print(f"Difference trace saved to {output_difference_file}")
'''
