from encipher import EncipherAPI
from picoscope_acquisition import DataAcquisition

if __name__ == "__main__":
    try:
        # Initialize Arduino and Picoscope APIs
        encipher = EncipherAPI(port="COM3")
        data_acquisition = DataAcquisition(
            sampling_rate=1e6, duration=0.01, channel="A", trigger_channel="EXT", output_dir="data"
        )

        # Configure Picoscope
        data_acquisition.setup_picoscope()

        # Perform multiple encryption rounds with data acquisition
        for round_number in range(1, 6):  # Example: 5 rounds
            print(f"Starting round {round_number}...")
            
            # Set initial plaintext block
            plaintext = b"1234567890ABCDEF"  # Example block, can vary per round
            encipher.set_state(plaintext)

            # Start encryption, which will trigger the Picoscope
            encipher.encrypt()

            # Start acquisition (Picoscope waits for trigger)
            data_acquisition.start_acquisition(round_number)

    finally:
        encipher.close()
        data_acquisition.close()
