#!/usr/bin/env python3
"""
NPZ visualizer for averaged AES traces

- Supports the NPZ files produced by your capture scripts, e.g.:
  * Single-trace files: keys like ["trace", "plaintext", "ciphertext", "sr", "timebase"]
  * Averaged files:     keys like ["trace_mean", "plaintext", "ciphertext",
                                "sr_nominal", "dt_ns", "timebase", "n_avg"]

- Provides:
  * Single-file plot (time axis if dt/sr present, else sample index)
  * Two-file comparison plot

- Prints a concise metadata summary (PT/CT, dt, sr, timebase, n_avg, length)
"""

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
FILEPATH_SINGLE = "arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000000.npz"

# If using two-file mode, set these two paths:
FILEPATH_ONE = "arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000000.npz"
FILEPATH_TWO = "arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000001.npz"

########################################
# Utility Functions
########################################

def _to_hex(b: np.ndarray | bytes | None) -> str | None:
    if b is None:
        return None
    if isinstance(b, bytes):
        return b.hex().upper()
    # np.ndarray of dtype uint8
    try:
        return bytes(b.tolist()).hex().upper()
    except Exception:
        return None

def load_npz_trace(file_path):
    """
    Load NPZ and return (waveform, metadata_dict).
    Recognizes both 'trace' and 'trace_mean' payloads.
    Derives dt (s) and sr (Hz) if possible.
    """
    try:
        npz = np.load(file_path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Could not load NPZ {file_path}: {e}")

    # waveform
    if "trace_mean" in npz.files:
        waveform = np.asarray(npz["trace_mean"])
    elif "trace" in npz.files:
        waveform = np.asarray(npz["trace"])
    elif "data" in npz.files:
        # fallback if someone saved under 'data'
        waveform = np.asarray(npz["data"])
    else:
        raise ValueError(f"{file_path}: neither 'trace_mean' nor 'trace' found.")

    # meta fields (optional)
    pt  = npz["plaintext"]  if "plaintext"  in npz.files else None
    ct  = npz["ciphertext"] if "ciphertext" in npz.files else None
    dt_ns = npz["dt_ns"].item() if "dt_ns" in npz.files else None
    sr_nominal = npz["sr_nominal"].item() if "sr_nominal" in npz.files else None
    sr = npz["sr"].item() if "sr" in npz.files else None  # legacy single-trace
    tb = npz["timebase"].item() if "timebase" in npz.files else None
    n_avg = npz["n_avg"].item() if "n_avg" in npz.files else None

    # derive dt (seconds) and sr (Hz)
    dt = None
    if dt_ns is not None and dt_ns != 0:
        dt = dt_ns * 1e-9
        sr_eff = 1.0 / dt
    elif sr is not None and sr > 0:
        sr_eff = sr
        dt = 1.0 / sr_eff
    elif sr_nominal is not None and sr_nominal > 0:
        sr_eff = sr_nominal
        dt = 1.0 / sr_eff
    else:
        sr_eff = None

    metadata = {
        "plaintext_hex": _to_hex(pt),
        "ciphertext_hex": _to_hex(ct),
        "dt_seconds": dt,
        "sr_hz": sr_eff,
        "timebase": tb,
        "n_avg": n_avg,
        "length": int(waveform.shape[0]),
        "file": file_path,
    }
    return waveform, metadata

def plot_single_trace(waveform, metadata=None, label="Trace"):
    """
    Plot a single waveform. Time axis if dt is known; else sample index.
    """
    if metadata and metadata.get("dt_seconds"):
        n = len(waveform)
        t = np.arange(n) * metadata["dt_seconds"]
        plt.plot(t, waveform, label=label)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.title(label)
    else:
        plt.plot(waveform, label=label)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude (mV)")
        plt.title(label)

def metadata_summary(metadata: dict):
    if not metadata:
        return "No metadata."
    lines = []
    for k in ("file","length","timebase","n_avg","sr_hz","dt_seconds","plaintext_hex","ciphertext_hex"):
        if k in metadata and metadata[k] is not None:
            if k.endswith("_hex") and isinstance(metadata[k], str):
                lines.append(f"{k}: {metadata[k]}")
            elif k in ("sr_hz","dt_seconds"):
                val = metadata[k]
                if k == "sr_hz":
                    lines.append(f"sr_hz: {val:.3f}" if isinstance(val, (int,float)) else f"sr_hz: {val}")
                else:
                    lines.append(f"dt_seconds: {val:.3e}" if isinstance(val, (int,float)) else f"dt_seconds: {val}")
            else:
                lines.append(f"{k}: {metadata[k]}")
    return "\n".join(lines)

def visualize_single_file(file_path):
    waveform, metadata = load_npz_trace(file_path)

    plt.figure(figsize=(10, 5))
    plot_single_trace(waveform, metadata, label=f"{file_path}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n=== Metadata Summary ===")
    print(metadata_summary(metadata))

def visualize_two_files(file_path1, file_path2):
    wf1, m1 = load_npz_trace(file_path1)
    wf2, m2 = load_npz_trace(file_path2)

    if len(wf1) != len(wf2):
        print(f"Warning: Waveform lengths differ: {len(wf1)} vs {len(wf2)}")

    plt.figure(figsize=(10, 5))

    # Prefer plotting both on same x-scale; if both have dt, use dt1 for axis
    if m1.get("dt_seconds"):
        t = np.arange(len(wf1)) * m1["dt_seconds"]
        plt.plot(t, wf1, label=f"{file_path1}")
        # if second has different dt, still overlay on t of wf1
        plt.plot(t[:len(wf2)], wf2[:len(wf1)], label=f"{file_path2}")
        plt.xlabel("Time (s)")
    else:
        plt.plot(wf1, label=f"{file_path1}")
        plt.plot(wf2, label=f"{file_path2}")
        plt.xlabel("Sample Index")

    plt.ylabel("Amplitude (mV)")
    plt.title("Comparing Two Waveforms (NPZ)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n=== Metadata Summary (File 1) ===")
    print(metadata_summary(m1))
    print("\n=== Metadata Summary (File 2) ===")
    print(metadata_summary(m2))

########################################
# Main Execution
########################################

def main():
    if PLOT_MODE == 1:
        visualize_single_file(FILEPATH_SINGLE)
    elif PLOT_MODE == 2:
        visualize_two_files(FILEPATH_ONE, FILEPATH_TWO)
    else:
        print("Invalid PLOT_MODE. Please set PLOT_MODE=1 or 2.")

if __name__ == "__main__":
    main()
