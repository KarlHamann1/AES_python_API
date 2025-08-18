#!/usr/bin/env python3
"""
Aggregate traces in a folder and compute an average trace.

- Works with both .npz (expects key 'trace' or 'trace_mean') and .npy files
- Resolves paths relative to repo: AES_PYTHON_API/arduino/data/<DIR_NAME>
- Saves: all_traces.npy (N x L) and averaged_trace.npy (L,)
"""

from pathlib import Path
import sys
import numpy as np

# ── Repo-aware paths ───────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[2]   
ARDUINO_DATA  = REPO_ROOT / "arduino" / "data"

# pick your data folder name
DIR_NAME      = "data_arduino_noise"                       
DATA_DIR      = ARDUINO_DATA / DIR_NAME

# ── File loading helpers ───────────────────────────────────────────
_PREFERRED_KEYS = ("trace", "trace_mean", "waveform", "data")

def _load_trace_file(p: Path) -> np.ndarray:
    """Return a 1-D float32 trace from .npz or .npy file."""
    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=False)
        return np.asarray(arr, dtype=np.float32).ravel()

    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            # Try common keys first, else take the first array in the archive
            key = next((k for k in _PREFERRED_KEYS if k in z.files), z.files[0])
            arr = z[key]
            return np.asarray(arr, dtype=np.float32).ravel()

    raise ValueError(f"Unsupported file type: {p}")

# ── Collect files ─────────────────────────────────────────────────
if not DATA_DIR.exists():
    sys.exit(f"[Error] Data folder not found: {DATA_DIR}")

files = sorted(list(DATA_DIR.glob("*.npz")) + list(DATA_DIR.glob("*.npy")))
if not files:
    sys.exit(f"[Error] No .npz/.npy files found in: {DATA_DIR}")

# ── Load all traces ───────────────────────────────────────────────
traces = []
lengths = []
for f in files:
    try:
        tr = _load_trace_file(f)
    except Exception as e:
        print(f"[Warn] Skipping {f.name}: {e}")
        continue
    traces.append(tr)
    lengths.append(tr.size)

if not traces:
    sys.exit("[Error] No valid traces loaded.")

# Enforce equal length (trim to shortest if needed)
min_len = min(lengths)
if any(L != min_len for L in lengths):
    print(f"[Info] Lengths differ; trimming all traces to {min_len} samples.")
    traces = [t[:min_len] for t in traces]

all_traces = np.vstack(traces)                 # shape (N, L)
avg_trace  = all_traces.mean(axis=0)           # shape (L,)

# ── Save outputs next to the inputs ───────────────────────────────
all_traces_path = DATA_DIR / "all_traces.npy"
avg_trace_path  = DATA_DIR / "averaged_trace.npy"

np.save(all_traces_path, all_traces.astype(np.float32))
np.save(avg_trace_path,  avg_trace.astype(np.float32))

print(f"[OK] Loaded {all_traces.shape[0]} traces from {DATA_DIR.name}")
print(f"[OK] Saved: {all_traces_path.name} (shape {all_traces.shape})")
print(f"[OK] Saved: {avg_trace_path.name}  (length {avg_trace.size})")


# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(avg_trace)
# plt.title(f"Averaged Trace · {DIR_NAME}")
# plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.grid(True); plt.tight_layout(); plt.show()
