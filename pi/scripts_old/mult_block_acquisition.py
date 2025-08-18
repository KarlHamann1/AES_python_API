"""
    mult_block_acquisition.py

    This script captures traces in **Block Mode** from a PicoScope
    while the Raspberry Pi runs a "mult" command (e.g., 'mult 1').

    It is analogous to your dummy_block_acquisition.py script, but here we:
    - Issue the "mult <num>" command to the Pi (bare-metal multiplication code).
    - Each command triggers the PicoScope to do a single block capture.
    - We save each waveform to an .npy file in a specified folder.
    - We also log the trace index and command into a CSV.

    Usage:
    1) Ensure your Pi firmware is running the multiplication-based kernel_main,
        with a UART command "mult <num>" that toggles a trigger line once.
    2) Adjust paths and parameters (e.g. sampling_rate_hz, capture_duration_s, output_dir).
    3) Run this script to capture multiple block-mode traces in a loop.
"""

import time
import threading
import csv
import os
import sys
import numpy as np

# Adjust imports to your local setup:
from pi_bare_metal_api import PiBareMetalAPI

# Assuming picoscope_acquisition.py is in the parent directory:
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from picoscope_acquisition import DataAcquisition

def main():
    """
    Perform block-mode acquisitions while the Pi runs 'mult' operations.
    - Each iteration triggers a single block capture on the PicoScope.
    - The captured waveform is saved as an .npy file.
    - A CSV logs the run index and the corresponding trace filename.
    """

    # 1) Connect to Pi (adjust port as needed)
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)
    time.sleep(0.2)  # Give Pi a moment to be ready

    # 2) Create the DataAcquisition object for **Block Mode**
    data_acq = DataAcquisition(
        device_resolution_bits=12,
        timebase=4,                 # ~16 ns sample interval at 12-bit
        sampling_rate_hz=62.5e6,    # ~16 ns per sample
        capture_duration_s=0.0008,  #  microseconds
        capture_channel="A",
        trigger_channel="EXT",       # External trigger
        trigger_threshold_mV=100,
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="10MV",
        output_dir="data_pi_0dB_10MV_block_mult_500_micro_with_300_micro_idle",  # Folder for block-mode multiplication traces
        filename_prefix="mult_trace_block"
    )

    try:
        # 3) Setup the PicoScope hardware
        data_acq.setup_picoscope()

        # Ensure output directory exists and create a CSV file for logging
        csv_filename = "trace_overview_mult_block.csv"
        csv_path = os.path.join(data_acq.output_dir, csv_filename)
        os.makedirs(data_acq.output_dir, exist_ok=True)

        # 4) Decide how many traces (block captures) we want
        num_traces = 1000  # Adjust as desired

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write header row
            csv_writer.writerow(["TraceIndex", "MultCommand", "TraceFilePath"])

            for trace_idx in range(num_traces):
                # a) Prepare the PicoScope for a single block capture
                data_acq.prepare_block_mode()

                # b) Send "mult 1" command to the Pi.
                #    The Pi should toggle the trigger line once.
                mult_cmd = "mult 1"
                pi_api.send_command(mult_cmd)

                # c) Optionally wait a moment if needed
                time.sleep(0.05)

                # d) Perform the actual block capture
                waveform = data_acq.capture_block()

                # e) Save the waveform to an .npy file
                trace_filename = f"mult_trace_block_{trace_idx}.npy"
                full_path = os.path.join(data_acq.output_dir, trace_filename)
                np.save(full_path, waveform)
                print(f"[Trace {trace_idx}] Waveform saved as {trace_filename}")

                # f) Log it to the CSV
                csv_writer.writerow([
                    trace_idx,
                    mult_cmd,
                    trace_filename
                ])

        print(f"\nAll {num_traces} block-mode multiplication traces captured.")
        print(f"CSV log: {csv_path}")

    finally:
        data_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
