import numpy as np
from encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition

def batch_no_encryption(num_rounds=1000, sampling_rate=1e6, duration=0.01):
    """
    Perform multiple rounds of data acquisition without cryptographic operations.
    
    :param num_rounds: Number of rounds to repeat the acquisition process.
    :param sampling_rate: Sampling rate for Picoscope.
    :param duration: Duration for each data capture.
    """
    encipher = EncipherAPI(port="COM3")
    data_acquisition = DataAcquisition(
        sampling_rate=sampling_rate,
        duration=duration,
        channel="A",
        trigger_channel="EXT",
        output_dir="data_no_encryption"
    )

    try:
        # Setup Picoscope
        data_acquisition.setup_picoscope()

        # Repeat acquisition process
        for round_number in range(1, num_rounds + 1):
            print(f"Starting round {round_number}/{num_rounds} without encryption...")

            # Set a dummy state (not used, but ensures sync)
            dummy_state = b'\x00' * 16
            encipher.set_state(dummy_state)

            # Trigger and capture without performing encryption
            data_acquisition.start_acquisition(round_number)

        print(f"Completed {num_rounds} rounds of acquisition without encryption.")

    finally:
        # Clean up
        encipher.close()
        data_acquisition.close()


if __name__ == "__main__":
    # Configuration
    NUM_ROUNDS = 1000       # Number of acquisition rounds

    # Run batch acquisition without encryption
    batch_no_encryption(num_rounds=NUM_ROUNDS)
