#!/usr/bin/env python3
"""
spectro_evaluate.py

Compute per-dataset spectrograms (STFT) and spectra of the grand-averaged trace.
Outputs under arduino/data/SPECTRO_eval/<Label>/ and a COMPARISON overlay of spectra.

Inputs per dataset (under arduino/data):
- trace_overview.csv (preferred) or *.npz files with:
    trace_mean or trace, optional dt_ns or sr_nominal, plaintext (ignored here)

Label mapping:
- data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV -> "16MHz_round1_125Msps"
- folders with "1MHz", "8MHz", "16MHz" -> "1MHz", "8MHz", "16MHz"

Notes
- Streaming average avoids storing all traces at once.
- Dud filtering by p2p/std, consistent with the other scripts.
- STFT parameters are chosen from fs to target ~50 kHz bin width, clipped to [256, 8192], power-of-two.
"""

from __future__ import annotations
import os, re, csv, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# repo layout
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "SPECTRO_eval").resolve()

# dataset selection (four sets)
DATASETS = [
    DATA_ROOT / "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV",
    DATA_ROOT / "data_arduino_1MHz_tb16_4.81Msps_9600Bd_avg100_10ms_20MHzBW_12bit_ACoff4mV",
    DATA_ROOT / "data_arduino_8MHz_tb5_31.25Msps_57600Bd_avg100_1.5ms_20MHzBW_12bit_ACoff2mV",
    DATA_ROOT / "data_arduino_16MHz_tb5_31Msps_115200Bd_avg100_0p7ms",
]

# label mapping to keep plots tidy
LABEL_MAP = {
    "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV": "16MHz_round1_125Msps",
    "1MHz": "1MHz",
    "8MHz": "8MHz",
    "16MHz": "16MHz",
}
def dataset_label_from_path(p: Path) -> str:
    name = p.name
    if "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV" in name:
        return "16MHz_round1_125Msps"
    for k in ("1MHz","8MHz","16MHz"):
        if k in name:
            return k
    m = re.search(r"(\d+)\s*MHz", name, re.IGNORECASE)
    return f"{m.group(1)}MHz" if m else name

# ROI (optional)
ROI_SAMPLES: Optional[Tuple[int,int]] = None      # (s0, s1)
ROI_TIME_S:  Optional[Tuple[float,float]] = None  # (t0, t1) if dt known
ROI_FRACTION: Optional[float] = None              # e.g., 0.20 for first 20%

# dud filtering
P2P_MIN_FRACTION = 0.20
STD_MIN_FRACTION = 0.15

# STFT / spectrum parameters
TARGET_DF_HZ   = 50e3     # target frequency-bin width for STFT
NFFT_MIN       = 256
NFFT_MAX       = 8192
OVERLAP_FRAC   = 0.5
FMAX_PLOT_HZ   = 10e6     # cap overlay plot to this frequency if Nyquist allows
SPEC_DB_FLOOR  = -120.0   # floor for dB scaling in plots

# speed knobs
MAX_TRACES_FOR_AVG: Optional[int] = None  # cap traces processed per dataset
DECIMATE: int = 1                         # sample decimation inside ROI (optional)



def load_rows_from_csv(ds_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    csv_path = ds_dir / "trace_overview.csv"
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    else:
        for p in sorted(ds_dir.glob("*.npz")):
            rows.append({"FileName": p.name})
    return rows

def open_trace_any(path: Path):
    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        wf = None
        if "trace_mean" in npz.files:
            wf = np.asarray(npz["trace_mean"], dtype=np.float64)
        elif "trace" in npz.files:
            wf = np.asarray(npz["trace"], dtype=np.float64)
        else:
            raise ValueError(f"{path.name}: no 'trace_mean'/'trace'")
        dt_ns = float(npz["dt_ns"].item()) if "dt_ns" in npz.files else None
        sr_nominal = float(npz["sr_nominal"].item()) if "sr_nominal" in npz.files else (float(npz["sr"].item()) if "sr" in npz.files else None)
        return wf, dt_ns, sr_nominal
    elif path.suffix.lower() == ".npy":
        wf = np.load(path).astype(np.float64, copy=False)
        return wf, None, None
    else:
        raise ValueError(f"Unsupported file: {path}")

def time_axis(n: int, dt_ns: Optional[float], sr_nominal: Optional[float]):
    dt_s = None
    if dt_ns and dt_ns > 0:
        dt_s = dt_ns * 1e-9
    elif sr_nominal and sr_nominal > 0:
        dt_s = 1.0 / sr_nominal
    if dt_s:
        return np.arange(n, dtype=np.float64) * dt_s, dt_s
    return np.arange(n, dtype=np.float64), None

def apply_roi_1d(x: np.ndarray, dt_s: Optional[float]):
    L = x.shape[0]
    s0, s1 = 0, L
    if ROI_SAMPLES is not None:
        s0, s1 = map(int, ROI_SAMPLES)
        s0 = max(0, min(s0, L)); s1 = max(s0, min(s1, L))
    elif ROI_TIME_S is not None and dt_s:
        t0, t1 = ROI_TIME_S
        s0 = max(0, int(round(t0 / dt_s)))
        s1 = min(L, int(round(t1 / dt_s)))
    elif ROI_FRACTION is not None and 0 < ROI_FRACTION <= 1.0:
        s1 = int(round(L * ROI_FRACTION))
        s1 = max(1, min(s1, L))
    if DECIMATE > 1:
        return x[s0:s1:DECIMATE], (s0, s1, DECIMATE)
    return x[s0:s1], (s0, s1, 1)

#  filtering and averaging 

def _trace_stats(x: np.ndarray) -> Tuple[float, float]:
    return float(x.max() - x.min()), float(np.std(x))

def robust_filter_decisions(ds_dir: Path, rows: List[Dict[str,str]]) -> Tuple[List[Tuple[Path,int]], float, Optional[float]]:
    stats = []
    T_min = None
    dt_ns_global = None
    for i, r in enumerate(rows):
        fn = r.get("FileName") or r.get("TraceFilePath")
        if not fn: 
            continue
        p = ds_dir / fn
        if not p.exists():
            p = Path(fn)
            if not p.exists():
                continue
        try:
            wf, dt_ns, sr_nom = open_trace_any(p)
        except Exception:
            continue
        if dt_ns_global is None and dt_ns: dt_ns_global = float(dt_ns)
        if T_min is None:
            T_min = int(wf.shape[0])
        else:
            T_min = min(T_min, int(wf.shape[0]))
        p2p, sd = _trace_stats(wf[:T_min])
        stats.append((p, p2p, sd))
        if MAX_TRACES_FOR_AVG and len(stats) >= MAX_TRACES_FOR_AVG:
            break
    if not stats:
        return [], 0.0, None
    p2p_vals = np.array([s[1] for s in stats], dtype=np.float64)
    sd_vals  = np.array([s[2] for s in stats], dtype=np.float64)
    med_p2p  = float(np.median(p2p_vals))
    med_sd   = float(np.median(sd_vals))
    keep = []
    for p, p2p, sd in stats:
        if p2p >= P2P_MIN_FRACTION * max(med_p2p, 1e-12) and sd >= STD_MIN_FRACTION * max(med_sd, 1e-12):
            keep.append((p, T_min))
    return keep, float(T_min), dt_ns_global

def streaming_grand_average(keep: List[Tuple[Path,int]], T_common: int) -> np.ndarray:
    if not keep:
        return np.zeros(0, dtype=np.float64)
    acc = np.zeros(T_common, dtype=np.float64)
    n   = 0
    for p, T in keep:
        try:
            wf, _, _ = open_trace_any(p)
        except Exception:
            continue
        x = wf[:T].astype(np.float64, copy=False)
        x = x - np.mean(x)  # per-trace DC removal
        acc += x
        n += 1
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return acc / float(n)

#  STFT and spectrum 

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2**int(np.ceil(np.log2(n)))

def choose_nperseg(fs: float) -> int:
    if fs <= 0:
        return 1024
    target = int(np.ceil(fs / TARGET_DF_HZ))
    n = min(max(target, NFFT_MIN), NFFT_MAX)
    return next_pow2(n)

def stft_spectrogram(x: np.ndarray, fs: float, nperseg: int, noverlap: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    if x.size == 0:
        return np.zeros((1,1)), np.array([0.0]), np.array([0.0])
    step = nperseg - noverlap
    if step <= 0:
        step = max(1, int(nperseg * (1.0 - OVERLAP_FRAC)))
    nwin = 1 + (len(x) - nperseg) // step if len(x) >= nperseg else 1
    # zero-pad short signals to at least one window
    pad = 0
    if len(x) < nperseg:
        pad = nperseg - len(x)
        x = np.pad(x, (0, pad), mode="constant")
        nwin = 1
    w = np.hanning(nperseg).astype(np.float64)
    win_pow = (w**2).sum()
    S = []
    tvec = []
    for i in range(nwin):
        s = i * step
        e = s + nperseg
        seg = x[s:e] * w
        X = np.fft.rfft(seg, n=nperseg)
        P = (np.abs(X)**2) / max(win_pow, 1e-18)   # power per bin
        S.append(P)
        t_center = (s + nperseg/2.0) / fs
        tvec.append(t_center)
    S = np.stack(S, axis=1) if len(S) > 1 else np.expand_dims(S[0], axis=1)
    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    t = np.array(tvec, dtype=np.float64)
    return S, f, t

def simple_periodogram_db(x: np.ndarray, fs: float) -> Tuple[np.ndarray,np.ndarray]:
    if x.size == 0:
        return np.array([0.0]), np.array([SPEC_DB_FLOOR])
    n = next_pow2(len(x))
    w = np.hanning(len(x))
    xw = x * w
    X = np.fft.rfft(xw, n=n)
    ps = (np.abs(X)**2) / max((w**2).sum(), 1e-18)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    ps_db = 10.0 * np.log10(ps + 1e-20)
    return f, ps_db

#  per-dataset workflow 

def eval_dataset(ds_dir: Path):
    label = dataset_label_from_path(ds_dir)
    out_dir = EVAL_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows_from_csv(ds_dir)
    keep, T_common, dt_ns = robust_filter_decisions(ds_dir, rows)
    if not keep:
        print(f"[skip] {label}: no usable traces")
        return None

    # sample rate
    fs = None
    if dt_ns and dt_ns > 0:
        fs = 1.0 / (dt_ns * 1e-9)
    else:
        # try to infer sr_nominal from any file
        for p, _ in keep:
            _, _, sr_nom = open_trace_any(p)
            if sr_nom and sr_nom > 0:
                fs = float(sr_nom); break
    if not fs:
        print(f"[warn] {label}: unknown sampling rate")
        return None

    # grand-average trace (per-trace DC removed)
    avg = streaming_grand_average(keep, int(T_common))
    if avg.size == 0:
        print(f"[skip] {label}: empty average")
        return None

    # axis and ROI
    axis, dt_s = time_axis(avg.shape[0], dt_ns, fs)
    avg_roi, (s0, s1, step) = apply_roi_1d(avg, dt_s)
    axis_roi = axis[s0:s1:step]

    # decimation is already applied inside apply_roi_1d

    # STFT spectrogram
    nperseg = choose_nperseg(fs)
    noverlap = int(OVERLAP_FRAC * nperseg)
    S, f, t = stft_spectrogram(avg_roi, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S_db = 10.0 * np.log10(S + 1e-20)

    # spectrum of average trace
    f_spec, ps_db = simple_periodogram_db(avg_roi, fs)

    # save arrays
    np.save(out_dir / "avg_trace.npy", avg_roi)
    np.save(out_dir / ("time_axis.npy" if dt_s else "sample_axis.npy"), axis_roi)
    np.save(out_dir / "spectrogram_power_db.npy", S_db)
    np.save(out_dir / "spectrogram_freq_hz.npy", f)
    np.save(out_dir / "spectrogram_time_s.npy", t)
    np.save(out_dir / "spectrum_freq_hz.npy", f_spec)
    np.save(out_dir / "spectrum_db.npy", ps_db)

    # plots
    # spectrogram (limit freq axis to available Nyquist and optional FMAX_PLOT_HZ)
    fmax = min(f.max(), FMAX_PLOT_HZ) if FMAX_PLOT_HZ else f.max()
    fmask = f <= fmax
    vmin = np.percentile(S_db[fmask, :], 5) if np.any(fmask) else np.percentile(S_db, 5)
    vmax = np.percentile(S_db[fmask, :], 99) if np.any(fmask) else np.percentile(S_db, 99)
    vmin = max(vmin, SPEC_DB_FLOOR)

    plt.figure(figsize=(11, 5.6))
    extent = [t[0], t[-1] if t.size>0 else 0.0, f[fmask][0] / 1e6, f[fmask][-1] / 1e6] if np.any(fmask) else [0,1,0,1]
    plt.imshow(S_db[fmask, :], origin="lower", aspect="auto",
            extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = plt.colorbar()
    cbar.set_label("Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"Spectrogram (STFT) – {label}  nperseg={nperseg}, overlap={int(OVERLAP_FRAC*100)}%")
    plt.tight_layout()
    plt.savefig(out_dir / "spectrogram.png", dpi=170)
    plt.close()

    # single-curve spectrum
    plt.figure(figsize=(10,5))
    fmask2 = f_spec <= fmax
    plt.plot(f_spec[fmask2] * 1e-6, ps_db[fmask2], linewidth=1.1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title(f"Spectrum of averaged trace – {label}")
    plt.tight_layout()
    plt.savefig(out_dir / "spectrum.png", dpi=170)
    plt.close()

    # summary
    (out_dir / "spectro_summary.txt").write_text(
        "\n".join([
            f"dataset: {label}",
            f"fs: {fs:.3f} Hz",
            f"avg_length_samples: {avg.shape[0]}  roi: [{s0}, {s1}) decim={step}",
            f"stft_nperseg: {nperseg}  overlap: {noverlap}",
            f"freq_plot_max_hz: {fmax:.3f}",
            f"vmin/vmax dB (5th/99th pct): {vmin:.1f}/{vmax:.1f}",
        ]), encoding="utf-8"
    )

    return {
        "label": label,
        "freq_hz": f_spec,
        "spectrum_db": ps_db,
        "fs": fs,
        "fmax_plotted": fmax
    }



def write_spectrum_overlay(results: List[Dict]):
    if not results:
        return
    comp_dir = EVAL_ROOT / "COMPARISON"
    comp_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12,5))
    for r in results:
        f = r["freq_hz"]; P = r["spectrum_db"]
        fmax = r["fmax_plotted"]
        mask = (f > 0) & (f <= fmax)
        plt.plot(f[mask] * 1e6 / 1e6, P[mask], linewidth=1.2, label=r["label"])  # MHz on x
    plt.grid(True, alpha=0.3)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title("Spectrum overlay (averaged trace)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(comp_dir / "spectrum_overlay.png", dpi=170)
    plt.close()

def main():
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for ds in DATASETS:
        ds = ds.resolve()
        if not ds.exists():
            print(f"[skip] dataset not found: {ds}")
            continue
        out = eval_dataset(ds)
        if out:
            results.append(out)
    write_spectrum_overlay(results)
    if results:
        print(f"spectrograms and spectra saved under: {EVAL_ROOT}")
    else:
        print("no outputs generated")

if __name__ == "__main__":
    main()
