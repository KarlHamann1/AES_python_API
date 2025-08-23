#!/usr/bin/env python3
"""
cpa_compare_collect_four.py

Collect GE/SR curves for byte 0 from four CPA_eval datasets and produce a single
comparison with cleaned labels, plus a combined JSON that includes:
- Ns/GE/SR arrays
- parsed JSON summary (from GE_SR.json)
- raw text summary (summary_b00.txt)
- recovered_key_hex.txt contents

Directory layout expected under arduino/data/CPA_eval:
1MHz/
8MHz/
16MHz/
data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV/
Each has byte_b00/{Ns.npy,GE.npy,SR.npy,GE_SR.json,summary_b00.txt} and recovered_key_hex.txt

Writes outputs to CPA_eval/COMPARISON_ALL_<STAMP>/:
GE_overlay_b00.png
SR_overlay_b00.png
cpa_comparison_b00.csv
cpa_comparison_b00.json
combined_b00.json
labels_map.json
"""

from __future__ import annotations
import json, csv, datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ARDUINO_DIR = REPO_ROOT / "arduino"
DATA_ROOT   = (ARDUINO_DIR / "data").resolve()
EVAL_ROOT   = (DATA_ROOT / "CPA_eval").resolve()

BYTE_INDEX = 0

# Fixed set of labels to collect
RAW_LABELS = [
    "1MHz",
    "8MHz",
    "16MHz",
    "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV",
]

# Pretty-name mapping for plots/tables
LABEL_MAP: Dict[str, str] = {
    "1MHz": "1MHz",
    "8MHz": "8MHz",
    "16MHz": "16MHz",
    "data_arduino_16MHz_tb3_125Msps_115200Bd_avg100_0p1ms_20MHzBW_15bit_DCoff4mV": "16MHz_1round_125Msps",
}

def load_curves(label_dir: Path, byte_idx: int):
    bdir = label_dir / f"byte_b{byte_idx:02d}"
    Ns_path = bdir / "Ns.npy"
    GE_path = bdir / "GE.npy"
    SR_path = bdir / "SR.npy"
    js_path = bdir / "GE_SR.json"
    tx_path = bdir / "summary_b00.txt"
    key_path = label_dir / "recovered_key_hex.txt"

    if not (Ns_path.exists() and GE_path.exists() and SR_path.exists()):
        return None

    Ns = np.load(Ns_path)
    GE = np.load(GE_path)
    SR = np.load(SR_path)

    js = None
    if js_path.exists():
        try:
            js = json.loads(js_path.read_text(encoding="utf-8"))
        except Exception:
            js = None

    txt_summary = None
    if tx_path.exists():
        try:
            txt_summary = tx_path.read_text(encoding="utf-8")
        except Exception:
            txt_summary = None

    key_hex = None
    if key_path.exists():
        try:
            key_hex = key_path.read_text(encoding="utf-8").strip()
        except Exception:
            key_hex = None

    return {
        "Ns": Ns,
        "GE": GE,
        "SR": SR,
        "json_summary": js.get("summary", None) if js else None,
        "text_summary": txt_summary,
        "recovered_key_hex": key_hex,
    }

def first_N_meeting_threshold(Ns: np.ndarray, y: np.ndarray, thr: float, mode: str) -> Optional[int]:
    if mode == "ge_le":
        idx = np.where(y <= thr)[0]
    else:
        idx = np.where(y >= thr)[0]
    return int(Ns[int(idx[0])]) if idx.size else None

def main():
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_ROOT / f"COMPARISON_ALL_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist mapping used in this run
    (out_dir / "labels_map.json").write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")

    rows: List[Dict] = []
    combined: List[Dict] = []

    # GE overlay
    plt.figure(figsize=(9,5))
    for raw in RAW_LABELS:
        label_dir = EVAL_ROOT / raw
        if not label_dir.exists():
            print(f"[skip] missing: {label_dir}")
            continue
        data = load_curves(label_dir, BYTE_INDEX)
        if data is None:
            print(f"[skip] incomplete byte_b{BYTE_INDEX:02d} in: {label_dir}")
            continue

        pretty = LABEL_MAP.get(raw, raw)
        Ns, GE, SR = data["Ns"], data["GE"], data["SR"]
        plt.plot(Ns, GE, marker="o", linewidth=1.2, label=pretty)

        ge1  = first_N_meeting_threshold(Ns, GE, 1.0, "ge_le")
        sr80 = first_N_meeting_threshold(Ns, SR, 0.8, "sr_ge")
        sr90 = first_N_meeting_threshold(Ns, SR, 0.9, "sr_ge")

        row = {
            "label_raw": raw,
            "label_pretty": pretty,
            "byte": BYTE_INDEX,
            "N_points": int(len(Ns)),
            "traces_to_GE_le_1": ge1,
            "traces_to_SR_ge_80pct": sr80,
            "traces_to_SR_ge_90pct": sr90,
        }

        # Copy selected fields from per-dataset JSON summary if available
        js = data["json_summary"] or {}
        for k in ("roi_samples", "decimate", "statistic", "n_resamples", "N_total", "N_total_after_filter"):
            if k in js:
                row[k] = js[k]

        rows.append(row)

        # Build combined entry with all requested details
        combined.append({
            "label_raw": raw,
            "label_pretty": pretty,
            "byte": BYTE_INDEX,
            "Ns": Ns.tolist(),
            "GE": GE.tolist(),
            "SR": SR.tolist(),
            "json_summary": js or None,
            "text_summary": data["text_summary"],
            "recovered_key_hex": data["recovered_key_hex"],
        })

    plt.xscale("log"); plt.yscale("log"); plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("# traces (log)")
    plt.ylabel("Guessing Entropy (log)")
    plt.title(f"CPA GE comparison (byte {BYTE_INDEX})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"GE_overlay_b{BYTE_INDEX:02d}.png", dpi=150)
    plt.close()

    # SR overlay
    plt.figure(figsize=(9,5))
    # re-iterate to keep legend ordering identical to GE plot
    for raw in RAW_LABELS:
        label_dir = EVAL_ROOT / raw
        if not label_dir.exists():
            continue
        data = load_curves(label_dir, BYTE_INDEX)
        if data is None:
            continue
        pretty = LABEL_MAP.get(raw, raw)
        plt.plot(data["Ns"], data["SR"], marker="o", linewidth=1.2, label=pretty)
    plt.xscale("log"); plt.ylim(0.0, 1.0); plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("# traces (log)")
    plt.ylabel("Success Rate")
    plt.title(f"CPA SR comparison (byte {BYTE_INDEX})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"SR_overlay_b{BYTE_INDEX:02d}.png", dpi=150)
    plt.close()

    # CSV and per-row JSON table
    if rows:
        fields = [
            "label_pretty", "label_raw", "byte", "N_points",
            "traces_to_GE_le_1", "traces_to_SR_ge_80pct", "traces_to_SR_ge_90pct",
            "roi_samples", "decimate", "statistic", "n_resamples", "N_total", "N_total_after_filter"
        ]
        with open(out_dir / f"cpa_comparison_b{BYTE_INDEX:02d}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                # ensure all fields present
                w.writerow({k: r.get(k, "") for k in fields})
        (out_dir / f"cpa_comparison_b{BYTE_INDEX:02d}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Combined rich JSON with curves + summaries + recovered keys
    (out_dir / "combined_b00.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(f"Combined comparison written to: {out_dir}")

if __name__ == "__main__":
    main()
