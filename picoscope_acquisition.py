import ctypes
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, mV2adc, adc2mV
import numpy as np
import time
import os

class DataAcquisition:
    def __init__(self, sampling_rate=1e6, duration=0.01, channel="A", trigger_channel="EXT", output_dir="data"):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.channel = channel
        self.trigger_channel = trigger_channel
        self.output_dir = output_dir
        self.scope = None
        self.chandle = ctypes.c_int16()
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_adc = None

    def setup_picoscope(self):
        # Open the PicoScope
        status_open = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, 1)
        try:
            assert_pico_ok(status_open)
            print("PicoScope connected successfully.")
        except:
            if status_open in [286, 282]:  # Handle power source issues
                ps.ps5000aChangePowerSource(self.chandle, status_open)
                print("Power source updated. Please retry.")
            else:
                raise

        # Set up main channel
        channel = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.channel}"]
        coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
        ch_range_enum = ps.PS5000A_RANGE["PS5000A_5V"]  # This is an index (e.g. 8)
        status = ps.ps5000aSetChannel(self.chandle, channel, 1, coupling, ch_range_enum, 0)
        assert_pico_ok(status)
        print(f"Channel {self.channel} configured for data acquisition.")

        # Set max ADC value
        self.max_adc = ctypes.c_int16(32512)

        # Configure trigger
        trigger_source = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.trigger_channel}"]
        threshold = 0  # 0 mV threshold
        # Pass the range enum (not millivolts) to mV2adc
        threshold_adc = int(mV2adc(threshold, ch_range_enum, self.max_adc))
        direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
        status = ps.ps5000aSetSimpleTrigger(self.chandle, 1, trigger_source, threshold_adc, direction, 0, 1000)
        assert_pico_ok(status)
        print(f"Trigger set on {self.trigger_channel} (Rising edge, threshold = 0 mV).")

        # Configure sampling and buffer
        pre_trigger_samples = int(self.sampling_rate * self.duration / 2)
        post_trigger_samples = pre_trigger_samples
        self.max_samples = pre_trigger_samples + post_trigger_samples

        timebase = int(1e9 / self.sampling_rate)
        time_interval_ns = ctypes.c_float()
        returned_max_samples = ctypes.c_int32()
        status = ps.ps5000aGetTimebase2(self.chandle, timebase, self.max_samples, ctypes.byref(time_interval_ns), ctypes.byref(returned_max_samples), 0)
        assert_pico_ok(status)
        print(f"Sampling configured: {self.sampling_rate} Hz for {self.duration}s.")

        self.ch_range_enum = ch_range_enum  # Store for later use in conversion

    def start_acquisition(self, round_number=None):
        # Start acquisition and wait for trigger
        print("Waiting for trigger...")
        status = ps.ps5000aRunBlock(self.chandle, int(self.max_samples / 2), int(self.max_samples / 2), 2, None, 0, None, None)
        assert_pico_ok(status)
        ready = ctypes.c_int16(0)
        timeout = time.time() + 5  # 5-second timeout

        while not ready.value:
            ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
            if time.time() > timeout:
                print("Trigger not detected within timeout period.")
                return False

        print("Trigger detected. Collecting data...")

        # Create buffer
        buffer = (ctypes.c_int16 * self.max_samples)()
        channel = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.channel}"]
        status = ps.ps5000aSetDataBuffer(self.chandle, channel, ctypes.byref(buffer), self.max_samples, 0, 0)
        assert_pico_ok(status)

        # Retrieve data
        overflow = ctypes.c_int16()
        num_samples = ctypes.c_uint32(self.max_samples)
        status = ps.ps5000aGetValues(self.chandle, 0, ctypes.byref(num_samples), 1, 0, 0, ctypes.byref(overflow))
        assert_pico_ok(status)

        # Convert data using the stored range enum
        voltage_buffer = adc2mV(buffer, self.ch_range_enum, self.max_adc)
        print(f"Captured data: {voltage_buffer[:5]}...")

        # Save data
        self.save_data(voltage_buffer, round_number)
        return True

    def save_data(self, buffer, round_number=None):
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.npy"
        if round_number is not None:
            filename = f"data_round_{round_number}_{timestamp}.npy"
        filepath = os.path.join(self.output_dir, filename)

        # Save buffer
        np.save(filepath, buffer)
        print(f"Data saved to {filepath}.")

    def stop_acquisition(self):
        # Stop the PicoScope
        status = ps.ps5000aStop(self.chandle)
        assert_pico_ok(status)
        print("PicoScope stopped.")

    def close(self):
        # Close the PicoScope
        if self.chandle:
            status = ps.ps5000aCloseUnit(self.chandle)
            assert_pico_ok(status)
            print("PicoScope connection closed.")


# Example Usage
def main():
    daq = DataAcquisition(sampling_rate=1e6, duration=0.01, channel="A", trigger_channel="A", output_dir="data")
    try:
        daq.setup_picoscope()
        if not daq.start_acquisition():
            print("No trigger detected. Exiting...")
    finally:
        daq.stop_acquisition()
        daq.close()

if __name__ == "__main__":
    main()
