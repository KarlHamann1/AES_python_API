#!/usr/bin/env python3
"""
trace_visualizer_np(any).py

NPZ/NPY visualizer for AES traces.

- Accepts both .npz and .npy:
  * NPZ: pick 'trace_mean' (preferred) or 'trace' (fallback); read metadata if present.
  * NPY: load the 1D array; metadata mostly unknown.
- Single-file plot or two-file comparison.
- Time axis if dt (seconds) or sr (Hz) available; otherwise sample index.
- Prints a compact metadata summary.

"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- User config -------------------------
PLOT_MODE = 1  # 1 = single, 2 = compare two files

# Single-file mode:
FILEPATH_SINGLE = Path("arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000000.npz")

# Two-file mode:
FILEPATH_ONE = Path("arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000000.npz")
FILEPATH_TWO = Path("arduino/data_arduino_16MHz_tb5_31Msps_115200Bd_avg100/encrypt_mean_000001.npz")

# Optional: override repo root (defaults to .../AES_PYTHON_API)
REPO_ROOT = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------


# ---------- small helpers ----------
def _resolve(path: Path) -> Path:
    """Resolve relative paths against repo root."""
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    cand = REPO_ROOT / path
    return cand if cand.exists() else path

def _to_hex(b: np.ndarray | bytes | None) -> str | None:
    if b is None:
        return None
    if isinstance(b, bytes):
        return b.hex().upper()
    try:
        return bytes(b.tolist()).hex().upper()
    except Exception:
        return None

def _load_any(file_path: Path):
    """
    Load .npz or .npy and return (waveform: float32[Ns], metadata: dict).
    NPZ: prefer 'trace_mean', fallback to 'trace'/'data'. Try dt_ns/sr/sr_nominal/timebase/n_avg.
    NPY: just load the vector; metadata minimal.
    """
    suffix = file_path.suffix.lower()
    md = {"file": str(file_path)}
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as z:
            if   "trace_mean" in z.files: wf = np.asarray(z["trace_mean"], dtype=np.float32)
            elif "trace"      in z.files: wf = np.asarray(z["trace"],      dtype=np.float32)
            elif "data"       in z.files: wf = np.asarray(z["data"],       dtype=np.float32)
            else:
                raise ValueError(f"{file_path}: no 'trace_mean'/'trace'/'data' array found")
            # optional fields
            pt  = z["plaintext"]  if "plaintext"  in z.files else None
            ct  = z["ciphertext"] if "ciphertext" in z.files else None
            dt_ns = float(z["dt_ns"].item()) if "dt_ns" in z.files else None
            sr_nom = float(z["sr_nominal"].item()) if "sr_nominal" in z.files else None
            sr = float(z["sr"].item()) if "sr" in z.files else None
            tb = int(z["timebase"].item()) if "timebase" in z.files else None
            n_avg = int(z["n_avg"].item()) if "n_avg" in z.files else None

            # pick dt/sr
            dt_s = None; sr_eff = None
            if dt_ns and dt_ns != 0.0:
                dt_s = dt_ns * 1e-9
                sr_eff = 1.0 / dt_s
            elif sr and sr > 0:
                sr_eff = sr; dt_s = 1.0 / sr_eff
            elif sr_nom and sr_nom > 0:
                sr_eff = sr_nom; dt_s = 1.0 / sr_eff

            md.update({
                "plaintext_hex": _to_hex(pt),
                "ciphertext_hex": _to_hex(ct),
                "dt_seconds": dt_s,
                "sr_hz": sr_eff,
                "timebase": tb,
                "n_avg": n_avg,
                "length": int(wf.shape[0]),
            })
            return wf, md
    elif suffix == ".npy":
        wf = np.asarray(np.load(file_path), dtype=np.float32)
        md.update({
            "plaintext_hex": None,
            "ciphertext_hex": None,
            "dt_seconds": None,
            "sr_hz": None,
            "timebase": None,
            "n_avg": None,
            "length": int(wf.shape[0]),
        })
        return wf, md
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def _meta_str(md: dict) -> str:
    lines = []
    for k in ("file","length","timebase","n_avg","sr_hz","dt_seconds","plaintext_hex","ciphertext_hex"):
        v = md.get(k, None)
        if v is None: 
            continue
        if k == "sr_hz" and isinstance(v, (int,float)):
            lines.append(f"{k}: {v:.3f}")
        elif k == "dt_seconds" and isinstance(v, (int,float)):
            lines.append(f"{k}: {v:.3e}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else "No metadata."

def _plot_single(wf: np.ndarray, md: dict, label: str):
    if md.get("dt_seconds"):
        t = np.arange(len(wf)) * float(md["dt_seconds"])
        plt.plot(t, wf, label=label)
        plt.xlabel("Time (s)")
    else:
        plt.plot(wf, label=label)
        plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (mV)")
    plt.title(label)

# ---------- modes ----------
def visualize_single_file(path: Path):
    f = _resolve(path)
    wf, md = _load_any(f)

    plt.figure(figsize=(10, 5))
    _plot_single(wf, md, label=f.name)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    print("\n=== Metadata ===")
    print(_meta_str(md))

def visualize_two_files(p1: Path, p2: Path):
    f1, f2 = _resolve(p1), _resolve(p2)
    wf1, m1 = _load_any(f1)
    wf2, m2 = _load_any(f2)

    if len(wf1) != len(wf2):
        print(f"[warn] length mismatch: {len(wf1)} vs {len(wf2)}")

    plt.figure(figsize=(10, 5))
    # If first has time info, use its dt for the x-axis; otherwise sample index
    if m1.get("dt_seconds"):
        t = np.arange(len(wf1)) * float(m1["dt_seconds"])
        plt.plot(t, wf1, label=f1.name)
        plt.plot(t[:len(wf2)], wf2[:len(wf1)], label=f2.name)
        plt.xlabel("Time (s)")
    else:
        plt.plot(wf1, label=f1.name)
        plt.plot(wf2, label=f2.name)
        plt.xlabel("Sample Index")

    plt.ylabel("Amplitude (mV)")
    plt.title("Comparing Two Waveforms")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    print("\n=== Metadata (File 1) ===")
    print(_meta_str(m1))
    print("\n=== Metadata (File 2) ===")
    print(_meta_str(m2))


# ------------------------- main -------------------------
def main():
    if PLOT_MODE == 1:
        visualize_single_file(FILEPATH_SINGLE)
    elif PLOT_MODE == 2:
        visualize_two_files(FILEPATH_ONE, FILEPATH_TWO)
    else:
        print("Invalid PLOT_MODE. Use 1 (single) or 2 (compare).")

if __name__ == "__main__":
    main()
