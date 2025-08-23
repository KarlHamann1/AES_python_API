"""
apply_bandpass_and_average.py

Band-pass all traces in a folder (.npy or .npz), average them, and save the mean.

- Accepts *.npy and *.npz (for NPZ, picks 'trace_mean' or 'trace' key; grabs dt_ns if present).
- Uses Butterworth in SOS form + sosfiltfilt for stability.
- Plots the filter response (freqz_sos / sosfreqz) on a log X axis.
- Crops every trace to the shortest length so shapes match.
- Optional ROI by samples or by time (needs dt_ns or DT_NS_OVERRIDE).
- Saves the averaged result to the same folder with a descriptive filename.
"""

import os
import glob
from pathlib import Path
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

REPO_ROOT    = Path(__file__).resolve().parents[1] 
ARDU_DATA_DIR= REPO_ROOT / "arduino" / "data"

USE_PI_DATA       = True
DATASET_DIR_NAME  = "data_pi_encryption"  # e.g. "data_pi_encryption" or "data_arduino_…"
DIRECTORY         = (PI_DATA_DIR if USE_PI_DATA else ARDU_DATA_DIR) / DATASET_DIR_NAME

# Sampling info
SAMPLING_RATE = 62.5e6            # Hz (used for filter design/plots)
DT_NS_OVERRIDE = None             # e.g., 32.0 to force dt if NPZ misses dt_ns

# Filter spec (Butterworth band-pass)
LOW_CUTOFF_HZ  = 10e3             # Hz
HIGH_CUTOFF_HZ = 1000e3           # Hz
FILTER_ORDER   = 2                # filter order for butter

# Optional ROI (crop before averaging)
ROI_SAMPLES = None                # (start_idx, end_idx) or None
ROI_TIME_S  = None                # (t0, t1) seconds or None; needs dt

LABELNAME        = f"Avg BPF ({int(LOW_CUTOFF_HZ/1e3)}–{int(HIGH_CUTOFF_HZ/1e3)} kHz)"
OUTPUT_FILENAME  = f"averaged_trace_bpf_{int(LOW_CUTOFF_HZ)}_{int(HIGH_CUTOFF_HZ)}Hz.npy"
SAVE_RESP_PNG    = False          # save the filter response plot


def _load_waveform(path: Path, dt_holder: dict) -> np.ndarray:
    """Load *.npy or *.npz. For NPZ prefer 'trace_mean', then 'trace'. Optionally pick up dt_ns."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as npz:
            if "trace_mean" in npz.files:
                wf = np.asarray(npz["trace_mean"], dtype=np.float32)
            elif "trace" in npz.files:
                wf = np.asarray(npz["trace"], dtype=np.float32)
            else:
                raise ValueError("NPZ missing 'trace_mean'/'trace'")
            if dt_holder.get("dt") is None and "dt_ns" in npz.files:
                try:
                    dt_holder["dt"] = float(npz["dt_ns"].item())
                except Exception:
                    pass
        return wf
    else:
        return np.asarray(np.load(path), dtype=np.float32)


def _design_bandpass_sos(fs: float, f_lo: float, f_hi: float, order: int):
    """Design a Butterworth BPF in SOS form. Clamp edges if needed."""
    if f_lo <= 0:
        f_lo = max(1.0, f_lo)  # keep strictly >0
    nyq = fs * 0.5
    if f_hi >= nyq:
        f_hi = nyq * 0.999
        print(f"[warn] high cutoff clamped to {f_hi:.1f} Hz (near Nyquist)")
    if not (0 < f_lo < f_hi < nyq):
        raise ValueError("cutoffs must satisfy 0 < LOW < HIGH < Nyquist")
    # Use fs=... so I pass physical Hz, not normalized
    sos = sps.butter(order, [f_lo, f_hi], btype="band", output="sos", fs=fs)
    return sos


def _plot_response(sos, fs: float, title: str):
    """Plot SOS frequency response on a log X-axis. Try freqz_sos, fallback to sosfreqz."""
    try:
        # SciPy ≥1.15
        w, h = sps.freqz_sos(sos, worN=4096, fs=fs)
    except Exception:
        # Legacy alias exists widely
        w, h = sps.sosfreqz(sos, worN=4096, fs=fs)
    plt.figure(figsize=(8.5, 4))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    plt.semilogx(w, mag_db, label="Band-pass response")
    plt.title(title)
    plt.xlabel("Frequency [Hz] (log)")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.xlim([max(1.0, w.min()), fs * 0.5])  # show up to Nyquist
    plt.tight_layout()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)

    # collect *.npy and *.npz
    files = sorted(
        list(glob.glob(str(DIRECTORY / "*.npy"))) +
        list(glob.glob(str(DIRECTORY / "*.npz")))
    )
    if not files:
        print(f"No .npy/.npz files found in: {DIRECTORY}")
        return

    # design filter (SOS + zero-phase via sosfiltfilt)
    try:
        sos = _design_bandpass_sos(SAMPLING_RATE, LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ, FILTER_ORDER)
    except Exception as e:
        print(f"Filter design failed: {e}")
        return

    _plot_response(
        sos, SAMPLING_RATE,
        f"Butterworth BPF (order={FILTER_ORDER}) ~ {LOW_CUTOFF_HZ/1e3:.1f}–{HIGH_CUTOFF_HZ/1e3:.1f} kHz"
    )
    if SAVE_RESP_PNG:
        plt.savefig(DIRECTORY / f"bpf_response_{int(LOW_CUTOFF_HZ)}_{int(HIGH_CUTOFF_HZ)}Hz.png", dpi=140)
    plt.show()

    # load all waveforms, remember dt if NPZ provides it
    dt_holder = {"dt": (DT_NS_OVERRIDE * 1e-9) if DT_NS_OVERRIDE else None}
    traces = []
    lengths = []

    for f in files:
        p = Path(f)
        try:
            wf = _load_waveform(p, dt_holder)
        except Exception as e:
            print(f"[warn] skipping {p.name}: {e}")
            continue
        if wf.ndim != 1 or wf.size == 0:
            print(f"[warn] {p.name}: not a 1D non-empty array")
            continue
        traces.append(np.asarray(wf, dtype=np.float32))
        lengths.append(wf.size)

    if not traces:
        print("No valid traces loaded.")
        return

    # crop to shortest length so shapes match (filtfilt keeps same length)
    min_len = int(min(lengths))
    traces = [t[:min_len] for t in traces]
    L = min_len

    # apply optional ROI before filtering (less work if cropping a lot)
    s0, s1 = 0, L
    if ROI_SAMPLES is not None:
        s0 = max(0, int(ROI_SAMPLES[0])); s1 = min(L, int(ROI_SAMPLES[1]))
        traces = [t[s0:s1] for t in traces]
        L = s1 - s0
        print(f"Applied ROI_SAMPLES=({s0},{s1}) → {L} samples.")
    elif ROI_TIME_S is not None and dt_holder["dt"]:
        dt = float(dt_holder["dt"])
        s0 = max(0, int(round(ROI_TIME_S[0] / dt)))
        s1 = min(L, int(round(ROI_TIME_S[1] / dt)))
        traces = [t[s0:s1] for t in traces]
        L = s1 - s0
        print(f"Applied ROI_TIME_S=({ROI_TIME_S[0]:.3e},{ROI_TIME_S[1]:.3e}) → samples ({s0},{s1}) → {L}.")

    if L < 8:
        print("[warn] ROI too small for filtering.")
        return

    # filter each trace (zero-phase)
    filtered = []
    for idx, t in enumerate(traces):
        try:
            y = sps.sosfiltfilt(sos, t, padtype="odd", padlen=None)  # default padlen; robust with SOS
        except ValueError as e:
            # fallback with smaller padlen if signal is very short
            safe_pad = max(1, min(len(t) // 2 - 1, 63))
            print(f"[warn] {files[idx]}: {e} → retry with padlen={safe_pad}")
            y = sps.sosfiltfilt(sos, t, padtype="odd", padlen=safe_pad)
        filtered.append(y.astype(np.float32, copy=False))

    # average
    avg = np.mean(np.stack(filtered, axis=0), axis=0).astype(np.float32)

    # save
    out_path = DIRECTORY / OUTPUT_FILENAME
    np.save(out_path, avg)
    print(f"\nSaved averaged band-pass trace: {out_path}")

    # quick look plot
    plt.figure(figsize=(10, 4))
    plt.plot(avg, label=LABELNAME)
    plt.title("Averaged Band-pass Filtered Trace")
    plt.xlabel("Sample Index" if not (ROI_TIME_S and dt_holder["dt"]) else "Time (s)")
    plt.ylabel("Amplitude (filtered)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
