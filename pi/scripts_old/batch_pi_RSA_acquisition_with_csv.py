import time
import os
import sys
import numpy as np
import csv

from pi_bare_metal_api import PiBareMetalAPI
from picoscope_acquisition import DataAcquisition

def main():
    # 1) Connect to Pi (UART)
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)
    time.sleep(0.2)  # brief pause

    # 2) Set up DataAcquisition in block mode (similar to your AES script)
    data_acq = DataAcquisition(
        device_resolution_bits=12,
        timebase=4,
        sampling_rate_hz=62.5e6,
        capture_duration_s=0.0005,  # e.g. 500 µs, adjust for RSA duration
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="DC",         # Possibly DC if you want low-freq signals
        voltage_range="20MV",
        output_dir="data_rsa_block",
        filename_prefix="rsa_trace_block"
    )

    try:
        data_acq.setup_picoscope()

        # Optional: set up a CSV for logging
        csv_filename = "rsa_trace_overview_block.csv"
        csv_path = os.path.join(data_acq.output_dir, csv_filename)
        os.makedirs(data_acq.output_dir, exist_ok=True)

        num_traces = 50  # or however many you want
        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["TraceIndex", "TraceFilePath"])  # minimal info

            for trace_idx in range(num_traces):
                # a) Prepare block mode
                data_acq.prepare_block_mode()

                # b) Trigger Pi to do one RSA exponentiation
                pi_api.send_command(f"rsa 1")

                # c) Capture block
                waveform = data_acq.capture_block()

                # d) Save to .npy
                trace_filename = f"rsa_trace_block_{trace_idx}.npy"
                full_path = os.path.join(data_acq.output_dir, trace_filename)
                np.save(full_path, waveform)
                print(f"[Trace {trace_idx}] saved as {trace_filename}")

                # e) CSV log row
                csv_writer.writerow([trace_idx, trace_filename])

    finally:
        data_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
