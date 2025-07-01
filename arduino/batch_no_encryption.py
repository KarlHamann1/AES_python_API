import numpy as np
from arduino.encipher import EncipherAPI
# Import your revised DataAcquisition class from the new file
# e.g., from data_acquisition import DataAcquisition
from picoscope_acquisition import DataAcquisition

def batch_no_encryption(
    num_rounds=100, 
    sampling_rate=62.5e6, 
    duration=0.012, 
    filename_prefix="no_encrypt"
):
    """
    Performs multiple rounds of data acquisition WITHOUT cryptographic operations
    on an Arduino (1 MHz clock). The recorded traces serve as a baseline to compare
    against encryption-based captures.
    
    :param num_rounds: how many acquisitions to perform
    :param sampling_rate: nominal sampling rate in Hz (62.5e6 => 16 ns => timebase=4 at 12-bit)
    :param duration: total capture duration in seconds (e.g., 0.012 => 12 ms)
    :param filename_prefix: prefix for naming the saved .npy data files
    """
    # Create an instance of the EncipherAPI to keep consistency
    encipher = EncipherAPI(port="COM5")

    # Create an instance of the DataAcquisition class 
    # with the 12-bit resolution, timebase=4, etc.
    data_acquisition = DataAcquisition(
        device_resolution_bits=12,     # 12-bit mode
        timebase=4,                   # fixed timebase=4 for 16 ns sampling in 12-bit mode
        sampling_rate_hz=sampling_rate,  
        capture_duration_s=duration,
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,     # typical trigger threshold
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="100MV",
        output_dir="data_no_encryption",
        filename_prefix=filename_prefix
    )

    try:
        # Initialize the PicoScope with the above config
        data_acquisition.setup_picoscope()

        # Repeat the acquisition 'num_rounds' times
        for round_number in range(1, num_rounds + 1):
            print(f"Starting round {round_number}/{num_rounds} without encryption...")

            # Sets a dummy state (16 null bytes), but no encryption is performed on the Arduino
            dummy_state = b'\x00' * 16
            encipher.set_state(dummy_state)  # no real effect, just for consistency of interface

            # Perform one acquisition/trace
            success = data_acquisition.start_acquisition(round_number)
            if not success:
                print(f"Trigger timed out on round {round_number}.")
                # Decide whether to break or keep going. We'll keep going here.

        print(f"Completed {num_rounds} rounds of acquisition WITHOUT encryption.")

    finally:
        # Close the EncipherAPI
        encipher.close()
        # Close the PicoScope
        data_acquisition.close()


if __name__ == "__main__":
    batch_no_encryption()
