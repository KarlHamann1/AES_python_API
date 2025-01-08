import numpy as np
from encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition

def batch_acquisition(num_rounds=100, sampling_rate=1e9, duration=0.01, filename_prefix= "encrypt"):
    """
    Performs multiple rounds of data acquisition for cryptographic analysis.
    """
    # Sets up a random number generator
    rng = np.random.default_rng()
    
    # Creates an instance of the EncipherAPI to communicate with the device
    encipher = EncipherAPI(port="COM5")

    # Creates an instance of the DataAcquisition class with chosen settings
    data_acquisition = DataAcquisition(
        sampling_rate=sampling_rate,
        duration=duration,
        channel="A",
        trigger_channel="EXT",
        output_dir="data"
    )

    try:
        # Initializes PicoScope with configured channel, trigger, and timebase
        data_acquisition.setup_picoscope()

        # Repeats the data acquisition process for the specified number of rounds
        for round_number in range(1, num_rounds + 1):
            print(f"Starting round {round_number}/{num_rounds}...")

            # Generates a random 16-byte plaintext
            plaintext = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
            print(f"Generated plaintext: {plaintext.hex()}")

            # Sets the plaintext on the device
            encipher.set_state(plaintext)

            # Triggers the cryptographic operation
            encipher.encrypt()

            # Captures one trace of data from the PicoScope
            data_acquisition.start_acquisition(round_number)

        print(f"Completed {num_rounds} rounds of acquisition.")

    finally:
        # Closes the EncipherAPI connection
        encipher.close()

        # Closes the PicoScope connection
        data_acquisition.close()


if __name__ == "__main__":
    # Configuration
    NUM_ROUNDS = 1000       # Number of acquisition rounds
    SAMPLING_RATE = 1e9     # Sampling rate in Hz
    DURATION = 0.01         # Acquisition duration in seconds

    # Run batch acquisition
    batch_acquisition(num_rounds=NUM_ROUNDS, 
    sampling_rate=SAMPLING_RATE, duration=DURATION
    )
