
"""
Compare two averaged traces:
- one from AES-encrypted runs
- one from noise-only runs

Computes simple metrics (MAD, RMSE, Max |Δ|) and plots both curves.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT   = Path(__file__).resolve().parents[2] 
ARDUINO_DATA = REPO_ROOT / "arduino" / "data"

# Set the *folder names* in arduino/data (edit these two lines as needed)
ENCRYPTION_DIR_NAME   = "data_arduino_8MHz_tb5_31.25Msps_57600Bd_avg100_1.5ms_20MHzBW_12bit_ACoff2mV"
NO_ENCRYPTION_DIR_NAME = "data_arduino_noise"

# File inside each folder to load 
AVERAGED_TRACE_FILENAME = "averaged_trace.npy"

# Build full paths
enc_path  = ARDUINO_DATA / ENCRYPTION_DIR_NAME   / AVERAGED_TRACE_FILENAME
noise_path = ARDUINO_DATA / NO_ENCRYPTION_DIR_NAME / AVERAGED_TRACE_FILENAME

#  Load averaged traces 
def load_vec(p: Path) -> np.ndarray:
    if not p.exists():
        sys.exit(f"[Error] Missing file: {p}")
    try:
        # np.load supports pathlib.Path directly
        return np.load(p)
    except Exception as e:
        sys.exit(f"[Error] Failed to load {p} : {e}")

avg_enc   = load_vec(enc_path)
avg_noise = load_vec(noise_path)

if avg_enc.shape != avg_noise.shape:
    sys.exit(f"[Error] Shape mismatch: enc {avg_enc.shape} vs noise {avg_noise.shape}")

#  Differences & metrics 
diff = avg_enc - avg_noise

mad  = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean(diff**2))
mx   = np.max(np.abs(diff))

print("Statistical Analysis of Differences")
print(f"- Mean Absolute Difference (MAD): {mad:.6g}")
print(f"- Root Mean Square Error (RMSE): {rmse:.6g}")
print(f"- Max Absolute Difference:       {mx:.6g}")

#  Plot 
plt.figure(figsize=(10, 6))
plt.plot(avg_enc,   label="Average Trace (Encryption)", linestyle="--")
plt.plot(avg_noise, label="Average Trace (No Encryption)", linestyle="-")
plt.title("Comparison of Averaged Traces")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
