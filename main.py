import numpy as np
from encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition


def batch_acquisition(num_rounds=1000, sampling_rate=1e6, duration=0.01):
    """
    Perform multiple rounds of data acquisition for cryptographic analysis.
    
    :param num_rounds: Number of rounds to repeat the acquisition process.
    :param sampling_rate: Sampling rate for Picoscope.
    :param duration: Duration for each data capture.
    """
    rng = np.random.default_rng()  # Random number generator
    encipher = EncipherAPI(port="COM3")
    data_acquisition = DataAcquisition(
        sampling_rate=sampling_rate,
        duration=duration,
        channel="A",
        trigger_channel="EXT",
        output_dir="data"
    )

    try:
        # Setup Picoscope
        data_acquisition.setup_picoscope()

        # Repeat acquisition process
        for round_number in range(1, num_rounds + 1):
            print(f"Starting round {round_number}/{num_rounds}...")

            # Generate random 16-byte plaintext
            plaintext = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
            print(f"Generated plaintext: {plaintext.hex()}")

            # Set random plaintext state
            encipher.set_state(plaintext)

            # Start encryption and data acquisition
            encipher.encrypt()
            data_acquisition.start_acquisition(round_number)

        print(f"Completed {num_rounds} rounds of acquisition.")

    finally:
        # Clean up
        encipher.close()
        data_acquisition.close()


if __name__ == "__main__":
    # Configuration
    NUM_ROUNDS = 1000       # Number of acquisition rounds
    SAMPLING_RATE = 1e6     # Sampling rate in Hz
    DURATION = 0.01         # Acquisition duration in seconds

    # Run batch acquisition
    batch_acquisition(num_rounds=NUM_ROUNDS, sampling_rate=SAMPLING_RATE, duration=DURATION)
