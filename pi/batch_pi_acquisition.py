import time
import threading
import numpy as np

from picoscope_acquisition import DataAcquisition
from pi.pi_bare_metal_api import PiBareMetalAPI

def continuous_read(pi_api):
    """
    Continuously read from the Pi's serial port and print each incoming line.
    Runs in a background thread so Pi output is visible in real time.
    """
    while pi_api.ser.is_open:
        try:
            line = pi_api.ser.readline()
            if line:
                print("PI>", line.decode(errors='ignore').rstrip())
        except Exception as e:
            print("Error reading from Pi:", e)
            break

def interactive_loop(pi_api):
    """
    Let the user type commands to the Pi in an interactive loop.
    """
    print("\nEntering interactive mode. Type commands or 'exit' to quit.\n")
    while True:
        user_input = input(">> ")
        if user_input.strip().lower() == "exit":
            break
        pi_api.send_command(user_input)
    print("Exiting interactive mode...")

def batch_pi_acquisition(
    pi_api,
    num_rounds=100,
    sampling_rate=62.5e6,      # ~16ns per sample if timebase=4 in 12-bit
    duration=0.00023,         # 230 microseconds
    filename_prefix="pi_encrypt_test"
):
    """
    Performs multiple rounds of data acquisition on the Raspberry Pi running
    the bare-metal AES kernel. For each round:
    - We do NOT set any plaintext (the Pi uses a default).
    - Tells the Pi to run 'aes 100' (i.e., 100 AES ops).
    - The Pi toggles a trigger pin for the scope (channel B).
    - DataAcquisition captures the waveform from channel A.

    :param pi_api: Instance of PiBareMetalAPI
    :param num_rounds: How many acquisitions to perform
    :param sampling_rate: e.g. 62.5e6 => 16ns sample interval (timebase=4)
    :param duration: total capture time in seconds (0.00023 => 230us)
    :param filename_prefix: name for the saved .npy files
    """

    data_acquisition = DataAcquisition(
        device_resolution_bits=12,   # 12-bit resolution
        timebase=4,                 # timebase=4 => ~16 ns sampling
        sampling_rate_hz=sampling_rate,
        capture_duration_s=duration,
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,   # 100 mV
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="20MV",       # 20 mV range for small signals
        output_dir="data_pi_encryption",
        filename_prefix=filename_prefix
    )

    try:
        data_acquisition.setup_picoscope()

        for round_idx in range(1, num_rounds + 1):
            print(f"\n--- Round {round_idx} / {num_rounds} ---")

            # 1) Run AES 100 times on Pi.
            #    (No plaintext is set, so the Pi uses a default.)
            pi_api.run_aes(1)

            # 2) Start scope acquisition (blocking until trigger or timeout)
            success = data_acquisition.start_acquisition(round_number=round_idx)
            if not success:
                print(f"Trigger timed out or no trigger for round {round_idx}.")
                # Decide if you want to continue or break.
                # break

        print(f"\nCompleted {num_rounds} rounds of Pi-based AES acquisitions.")

    finally:
        data_acquisition.close()

def main():
    # Create PiBareMetalAPI instance
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)

    # Start the background thread for continuous printing of Pi's output
    reader_thread = threading.Thread(target=continuous_read, args=(pi_api,), daemon=True)
    reader_thread.start()
    time.sleep(0.1)  # Give the thread a moment to start

    try:
        # Run batch acquisition
        batch_pi_acquisition(
            pi_api,
            num_rounds=100,         # Example: 100 acquisitions
            sampling_rate=62.5e6, # ~16ns per sample (timebase=4)
            duration=0.00023,     # 230 microseconds
            filename_prefix="pi_encrypt_test"
        )

        # After acquisition, optionally go interactive
        interactive_loop(pi_api)

    finally:
        # Close the Pi's serial connection
        pi_api.close()

if __name__ == "__main__":
    main()
