"""
dummy_block_acquisition.py

This script captures traces in **Block Mode** from a PicoScope
while the Raspberry Pi runs a "dummy" routine (e.g., 'dummy 100').

    It is analogous to the block-mode AES example, but here we:
    - Do NOT set a plaintext (dummy doesn't need it).
    - Do NOT parse a ciphertext.
    - Just record the waveform each time we call 'dummy <num>'.

    Output:
    - A series of .npy files containing the waveforms.
    - An optional CSV listing the run index and the corresponding trace filename.
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
    Perform block-mode acquisitions while the Pi runs 'dummy' operations.
    - Each run triggers a block capture on the PicoScope.
    - The captured waveform is saved as an .npy file.
    - A CSV tracks which trace index corresponds to which file.
    """

    # 1) Connect to Pi
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)
    time.sleep(0.2)  # give Pi a moment

    # 2) Create the DataAcquisition object for **Block Mode**
    data_acq = DataAcquisition(
        device_resolution_bits=12,
        timebase=4,                 # ~16 ns sample interval at 12-bit
        sampling_rate_hz=62.5e6,    # ~16 ns per sample
        capture_duration_s=0.00023, # 230 microseconds
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="20MV",
        output_dir="data_pi_40dB_block_dummy",  # Folder for block-mode dummy traces
        filename_prefix="dummy_trace_block"
    )

    try:
        # 3) Setup the PicoScope hardware
        data_acq.setup_picoscope()

        # Create output directory and a CSV for logging
        csv_filename = "trace_overview_dummy_block.csv"
        csv_path = os.path.join(data_acq.output_dir, csv_filename)
        os.makedirs(data_acq.output_dir, exist_ok=True)

        # 4) How many traces do we want?
        num_traces = 10000  # Adjust as needed

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write header row
            csv_writer.writerow(["TraceIndex", "DummyCommand", "TraceFilePath"])

            for trace_idx in range(num_traces):
                # a) Prepare for block capture
                data_acq.prepare_block_mode()

                # b) Pi executes 'dummy 100' => triggers the scope
                dummy_cmd = "dummy 1"
                pi_api.send_command(dummy_cmd)
                # Alternatively, if you implemented run_dummy(...) in pi_bare_metal_api, do:
                # pi_api.run_dummy(100)

                # c) Wait a brief moment if needed
                time.sleep(0.05)

                # d) Perform the block capture
                waveform = data_acq.capture_block()

                # e) Save the waveform
                trace_filename = f"dummy_trace_block_{trace_idx}.npy"
                full_path = os.path.join(data_acq.output_dir, trace_filename)
                np.save(full_path, waveform)
                print(f"[Trace {trace_idx}] Waveform saved as {trace_filename}")

                # f) Log to CSV (no plaintext/ciphertext here, just store the command used)
                csv_writer.writerow([
                    trace_idx,
                    dummy_cmd,
                    trace_filename
                ])

        print(f"\nAll {num_traces} block-mode dummy traces captured. CSV at: {csv_path}")

    finally:
        data_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
