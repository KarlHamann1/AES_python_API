import os
import sys
import csv
import time
import secrets  # for random plaintext
import numpy as np

# If "picoscope_acquisition.py" and "encipher.py" are in the same directory or parent directory:
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from arduino.encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition

def batch_with_encryption(
    num_rounds=10000,
    duration=0.01,           # 10 ms of capture
    filename_prefix="encrypt",
    output_dir="data_arduino_30dB_block_encryption"
):
    """
    Performs multiple rounds of data acquisition on an Arduino running AES encryption.

    For each round:
        1) Generate a random 16-byte plaintext and send it to the Arduino.
        2) Prepare the PicoScope in block mode (which waits for a trigger).
        3) Trigger AES encryption on the Arduino (pin A0 → scope's trigger channel).
        4) Capture the scope trace once triggered.
        5) Read the ciphertext from the Arduino.
        6) Save the waveform to a .npy file.
        7) Write plaintext/ciphertext/filename to a CSV log.

    :param num_rounds: How many traces/encryptions to perform.
    :param duration: Capture duration in seconds (10 ms).
    :param filename_prefix: Prefix for the .npy files we save.
    :param output_dir: Directory to store waveforms and CSV logs.
    """

    # 1) Create an instance of the EncipherAPI to control the Arduino.
    #    Adjust the COM port to match your Arduino.
    encipher = EncipherAPI(port="COM5", baudrate=9600, timeout=1)

    # 2) Create a DataAcquisition instance:
    #
    #    - use 'device_resolution_bits=12' for 12-bit mode.
    #    - **timebase=43** => sample interval = (43 - 3) / 62.5e6 = ~640 ns => ~1.56 MS/s.
    #    - For a 10 ms capture, that yields ~15,600 samples (much less than timebase=4).
    #    - Coupling = "AC" if your photodiode/measurement chain needs it.
    #    - voltage_range="10MV" is presumably ±10 mV range—adjust 
    
    # keep 'capture_duration_s=duration' so the code can calculate how many samples it
    # expects. The final count will be near 1.56 MS/s * 0.01 s ≈ 15,600.
    data_acquisition = DataAcquisition(
        device_resolution_bits=12,
        timebase=43,                  # <-- KEY CHANGE: slower sampling for smaller traces
        sampling_rate_hz=1.56e6,      # nominal, but overridden by timebase
        capture_duration_s=duration,
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,     # trigger level in mV
        trigger_delay_samples=0,
        auto_trigger_ms=1000,         # auto-trigger after 1 second if no real trigger
        coupling_mode="AC",
        voltage_range="10MV",
        output_dir=output_dir,
        filename_prefix=filename_prefix
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a CSV file to log plaintext, ciphertext, and the corresponding trace filename
    csv_filename = "trace_overview_arduino.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    try:
        # 3) Initialize and configure the PicoScope hardware
        data_acquisition.setup_picoscope()

        # 4) Open CSV file for logging
        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write header
            csv_writer.writerow(["TraceIndex", "PlaintextHex", "CiphertextHex", "TraceFilePath"])

            # 5) Acquire N traces
            for trace_idx in range(num_rounds):
                print(f"\n--- Round {trace_idx+1} / {num_rounds} ---")

                # a) Generate random 16-byte plaintext
                plaintext = secrets.token_bytes(16)
                plaintext_hex = "".join(f"{b:02X}" for b in plaintext)
                print(f"Plaintext = {plaintext_hex}")

                # b) Send this plaintext to the Arduino
                encipher.set_state(plaintext)

                # c) Prepare scope in block mode (arms the scope, waiting for a trigger)
                data_acquisition.prepare_block_mode()

                # d) Trigger AES encryption on Arduino (toggles A0 -> triggers scope)
                encipher.encrypt()

                # e) Capture the triggered waveform
                waveform = data_acquisition.capture_block()
                print(f"Captured {len(waveform)} samples from the scope.")

                # f) Read the ciphertext from the Arduino
                #    The AES result is left in `state` after encryption.
                ciphertext = encipher.get_state()
                ciphertext_hex = "".join(f"{b:02X}" for b in ciphertext)
                print(f"Ciphertext = {ciphertext_hex}")

                # g) Save the waveform as a .npy file
                trace_filename = f"{filename_prefix}_{trace_idx:05d}.npy"
                full_trace_path = os.path.join(output_dir, trace_filename)
                np.save(full_trace_path, waveform)
                print(f"Waveform saved to {trace_filename}")

                # h) Write this acquisition's data to CSV
                csv_writer.writerow([
                    trace_idx,
                    plaintext_hex,
                    ciphertext_hex,
                    trace_filename
                ])

        print(f"\nAll {num_rounds} traces captured successfully.")
        print(f"CSV summary saved to: {csv_path}")

    finally:
        # 6) Clean up: close the scope and Arduino connection
        data_acquisition.close()
        encipher.close()


if __name__ == "__main__":
    # Example usage: acquire 10 traces with ~1.56 MS/s for ~10 ms each (timebase=43).
    batch_with_encryption()
