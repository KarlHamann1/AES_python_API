#!/usr/bin/env python3
"""
filter_single_trace_highpass.py

Single-trace high-pass
- Accept .npy or .npz (for NPZ I pick 'trace_mean' or 'trace'; also try 'dt_ns').
- Use Butterworth in SOS form + sosfiltfilt for stable zero-phase filtering.  # docs: butter(output='sos'), sosfiltfilt
- If FILE_PATH is relative, resolve it from repo root (AES_PYTHON_API/).
- Optional ROI by samples or by time (needs dt from NPZ or DT_NS_OVERRIDE).
- Save filtered trace next to the input with a descriptive filename.
"""

from pathlib import Path
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

REPO_ROOT        = Path(__file__).resolve().parents[1]

# Point to a single trace
FILE_PATH        = Path("pi/data_pi_0dB_10MV_block_mult_500_micro_with_300_micro_idle/mult_trace_block_0.npy")

SAMPLING_RATE    = 62.5e6     # Hz (used for design/plots)
HIGH_PASS_CUTOFF = 5e6        # Hz
FILTER_ORDER     = 2          # butter

# Optional ROI
ROI_SAMPLES      = None       # e.g., (start_idx, end_idx)
ROI_TIME_S       = None       # e.g., (t0, t1), needs dt
DT_NS_OVERRIDE   = None       # e.g., 16.0 to force 16 ns/sample

# leave False to only show
SAVE_RESP_PNG    = False
SAVE_TRACE_PNG   = False



def _resolve_input(path: Path) -> Path:
    """Resolve against repo root if not absolute and not found as-is."""
    if path.is_absolute() and path.is_file():
        return path
    if path.is_file():
        return path
    cand = (REPO_ROOT / path)
    return cand if cand.is_file() else path 

def _load_waveform(path: Path, dt_holder: dict) -> np.ndarray:
    """
    Load .npy or .npz. For NPZ prefer 'trace_mean', then 'trace'.
    If NPZ has 'dt_ns' and dt is unset, remember it.  # numpy.load docs
    """
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


def _design_highpass_sos(fs: float, f_cut: float, order: int):
    """Butterworth HPF (SOS). Clamp edges if needed.  # butter(fs=..., output='sos')"""
    nyq = fs * 0.5
    f_cut = max(1.0, min(f_cut, nyq * 0.999))  # keep inside (0, Nyquist)
    return sps.butter(order, f_cut, btype="high", output="sos", fs=fs)


def _plot_response(sos, fs: float, cutoff_hz: float, order: int, save_to: Path | None):
    """Plot SOS frequency response on log X; try freqz_sos, fallback to sosfreqz."""
    try:
        w, h = sps.freqz_sos(sos, worN=4096, fs=fs)
    except Exception:
        w, h = sps.sosfreqz(sos, worN=4096, fs=fs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    label_cut = f"{cutoff_hz/1e3:.1f} kHz" if cutoff_hz < 1e6 else f"{cutoff_hz/1e6:.1f} MHz"

    plt.figure(figsize=(8.5, 4))
    plt.semilogx(w, mag_db, label="High-pass response")
    plt.title(f"Butterworth HPF (order={order}) — cutoff ~ {label_cut}")
    plt.xlabel("Frequency [Hz] (log)")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.xlim([max(1.0, w.min()), fs * 0.5])
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=140)
    plt.show()


def main():
    # resolve and load
    in_path = _resolve_input(Path(FILE_PATH))
    if not in_path.is_file():
        print(f"Error: file not found: {in_path}")
        return

    dt_holder = {"dt": (DT_NS_OVERRIDE * 1e-9) if DT_NS_OVERRIDE else None}
    trace = _load_waveform(in_path, dt_holder)
    if trace.ndim != 1 or trace.size == 0:
        print("Error: input must be a 1D non-empty array.")
        return
    L = trace.size
    print(f"Loaded: {in_path}  (length={L} samples)")

    # design + response
    sos = _design_highpass_sos(SAMPLING_RATE, HIGH_PASS_CUTOFF, FILTER_ORDER)
    resp_png = in_path.with_suffix("").with_name(in_path.stem + f"__hpf_{int(HIGH_PASS_CUTOFF)}Hz__resp.png") if SAVE_RESP_PNG else None
    _plot_response(sos, SAMPLING_RATE, HIGH_PASS_CUTOFF, FILTER_ORDER, resp_png)

    # optional ROI (crop before filtering)
    s0, s1 = 0, L
    if ROI_SAMPLES is not None:
        s0 = max(0, int(ROI_SAMPLES[0])); s1 = min(L, int(ROI_SAMPLES[1]))
        trace = trace[s0:s1]; L = trace.size
        print(f"Applied ROI_SAMPLES=({s0},{s1}) → {L} samples.")
    elif ROI_TIME_S is not None and dt_holder["dt"]:
        dt = float(dt_holder["dt"])
        s0 = max(0, int(round(ROI_TIME_S[0] / dt)))
        s1 = min(L, int(round(ROI_TIME_S[1] / dt)))
        trace = trace[s0:s1]; L = trace.size
        print(f"Applied ROI_TIME_S=({ROI_TIME_S[0]:.3e},{ROI_TIME_S[1]:.3e}) → samples ({s0},{s1}) → {L}.")

    if L < 8:
        print("[warn] ROI too small for filtering.")
        return

    # filter
    try:
        filtered = sps.sosfiltfilt(sos, trace, padtype="odd", padlen=None)
    except ValueError as e:
        safe_pad = max(1, min(L // 2 - 1, 63))
        print(f"[warn] {e} → retry with padlen={safe_pad}")
        filtered = sps.sosfiltfilt(sos, trace, padtype="odd", padlen=safe_pad)

    # save next to input
    label_cut = (f"{HIGH_PASS_CUTOFF/1e3:.0f}k" if HIGH_PASS_CUTOFF < 1e6 else f"{HIGH_PASS_CUTOFF/1e6:.1f}M")
    out_name  = in_path.stem + f"__hpf_{label_cut}.npy"
    out_path  = in_path.with_name(out_name)
    np.save(out_path, filtered.astype(np.float32, copy=False))
    print(f"Saved filtered trace → {out_path}")

    # quick look plot
    plt.figure(figsize=(10, 4))
    plt.plot(filtered, label=f"High-pass ({label_cut})")
    plt.title("Filtered Trace (High-pass)")
    plt.xlabel("Sample Index" if not (ROI_TIME_S and dt_holder['dt']) else "Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if SAVE_TRACE_PNG:
        plt.savefig(out_path.with_suffix(".png"), dpi=140)
    plt.show()


if __name__ == "__main__":
    main()
