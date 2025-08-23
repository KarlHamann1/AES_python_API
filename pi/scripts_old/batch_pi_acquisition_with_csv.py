import time
import threading
import secrets  # random plaintexts
import csv
import os
import sys
import numpy as np

from pi_bare_metal_api import PiBareMetalAPI

# Path setup: assume picoscope_acquisition.py lives in the parent dir.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from picoscope_acquisition import DataAcquisition

def main():
    """
    Runs a block-mode acquisition for AES side-channel captures:
    - send a plaintext to the Pi
    - arm the PicoScope in block mode
    - start 'run_aes(1)' on the Pi (one trigger)
    - read the waveform via block mode and save as .npy
    - read the ciphertext from Pi and write it together with the plaintext to a CSV
    """

    # 1) connect to Pi (UART)
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)

    # (optional) background thread to watch live UART output
    # def continuous_read(pi):
    #     ...
    # reader_thread = threading.Thread(target=continuous_read, args=(pi_api,), daemon=True)
    # reader_thread.start()

    # small pause so the Pi is ready
    time.sleep(0.2)

    # 2) build the DataAcquisition object for block mode
    data_acq = DataAcquisition(
        device_resolution_bits=12,
        timebase=4,                 # ~16 ns sample interval at 12-bit
        sampling_rate_hz=62.5e6,    # ~16 ns per sample
        capture_duration_s=0.00023, # 230 us
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="20MV",
        output_dir="data_pi_40dB_block_encryption",  # new folder for block-mode data
        filename_prefix="aes_trace_block"
    )

    try:
        # 3) set up the PicoScope
        data_acq.setup_picoscope()

        # 4) CSV file for the whole run
        csv_filename = "trace_overview_block.csv"
        csv_path = os.path.join(data_acq.output_dir, csv_filename)
        os.makedirs(data_acq.output_dir, exist_ok=True)  # create folder if missing

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # header
            csv_writer.writerow(["TraceIndex", "PlaintextHex", "CiphertextHex", "TraceFilePath"])

            # 5) number of traces (e.g., 10_000)
            num_traces = 10
            for trace_idx in range(num_traces):
                # a) make a random plaintext
                plaintext_bytes = secrets.token_bytes(16)
                plaintext_hex = "".join(f"{b:02X}" for b in plaintext_bytes)

                # b) send plaintext to Pi
                pi_api.set_plaintext(plaintext_hex)

                # c) prep scope block capture
                #    do this every loop so we get one block per AES run
                data_acq.prepare_block_mode()

                # d) Pi runs AES -> scope gets the trigger
                pi_api.run_aes(1)

                # e) pull the block capture (blocks until getValues is done)
                waveform = data_acq.capture_block()

                # f) save waveform as .npy
                trace_filename = f"aes_trace_block_{trace_idx}.npy"
                full_path = os.path.join(data_acq.output_dir, trace_filename)
                np.save(full_path, waveform)
                print(f"[Trace {trace_idx}] Wellenform gespeichert als {trace_filename}")

                # g) read ciphertext from Pi over UART
                time.sleep(0.05)  # small wait so the Pi can send
                lines = pi_api.read_lines(max_lines=20)
                ciphertext_hex = None
                for line in lines:
                    if "Final Ciphertext: " in line:
                        parts = line.split(": ")
                        if len(parts) == 2:
                            ciphertext_hex = parts[1].strip()
                            break

                # h) write one row into the CSV
                csv_writer.writerow([
                    trace_idx,
                    plaintext_hex,
                    ciphertext_hex if ciphertext_hex else "UNKNOWN",
                    trace_filename
                ])

        print(f"\nAlle {num_traces} Block-Mode-Traces wurden aufgezeichnet. CSV in {csv_path}.")

    finally:
        data_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
