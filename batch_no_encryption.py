import numpy as np
from encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition

def batch_no_encryption(num_rounds=1000, sampling_rate=1e9, duration=0.012, filename_prefix= "no_encrypt"):
    """
    Performs multiple rounds of data acquisition without cryptographic operations.
    """
    # Creates an instance of the EncipherAPI to maintain a consistent interface
    encipher = EncipherAPI(port="COM5")

    # Creates an instance of the DataAcquisition class with chosen settings
    data_acquisition = DataAcquisition(
        sampling_rate=sampling_rate,
        duration=duration,
        channel="A",
        trigger_channel="B",
        output_dir="data_no_encryption"
    )

    try:
        # Initializes PicoScope with configured channel, trigger, and timebase
        data_acquisition.setup_picoscope()

        # Repeats the data acquisition process for the specified number of rounds
        for round_number in range(1, num_rounds + 1):
            print(f"Starting round {round_number}/{num_rounds} without encryption...")

            # Sets a dummy state (no encryption is performed)
            dummy_state = b'\x00' * 16
            encipher.set_state(dummy_state)

            # Triggers the scope and captures one trace
            data_acquisition.start_acquisition(round_number)

        print(f"Completed {num_rounds} rounds of acquisition without encryption.")

    finally:
        # Closes the EncipherAPI connection
        encipher.close()

        # Closes the PicoScope connection
        data_acquisition.close()


if __name__ == "__main__":
    #  configuration for 1000 acquisitions at 1 GS/s, each lasting 12 ms
    batch_no_encryption(
        num_rounds=100,
        sampling_rate=1e9,
        duration=0.012,       # 12 ms total capture
    )
