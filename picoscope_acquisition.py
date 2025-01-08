import ctypes
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, mV2adc, adc2mV
import numpy as np
import time
import os

class DataAcquisition:
    def __init__(
        self,
        sampling_rate=1e9,   # 1 GS/s
        duration=20e-6,      # 20 microseconds total capture
        channel="A",
        trigger_channel="EXT",
        output_dir="data",
        filename_prefix=None
    ):
        # Sets basic parameters
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.channel = channel
        self.trigger_channel = trigger_channel
        self.output_dir = output_dir
        
        # Optional filename prefix for saved data
        self.filename_prefix = filename_prefix
        
        # Creates handle for the scope
        self.chandle = ctypes.c_int16()
        
        # Ensures the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Will hold the PicoScope's maximum ADC value
        self.max_adc = None

    def setup_picoscope(self):
        # Opens the PicoScope device
        status_open = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, 1)
        try:
            assert_pico_ok(status_open)
            print("PicoScope connected successfully.")
        except:
            if status_open in [286, 282]:  # Power source issues
                ps.ps5000aChangePowerSource(self.chandle, status_open)
                print("Power source updated. Please retry.")
            else:
                raise

        # Sets channel to DC mode with ±5 V range
        channel = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.channel}"]
        coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
        ch_range_enum = ps.PS5000A_RANGE["PS5000A_5V"]
        status = ps.ps5000aSetChannel(self.chandle, channel, 1, coupling, ch_range_enum, 0)
        assert_pico_ok(status)
        print(f"Channel {self.channel} configured for data acquisition.")

        # Sets max ADC value (device-dependent)
        self.max_adc = ctypes.c_int16(32512)

        # Configures trigger on the EXT input
        trigger_source = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.trigger_channel}"]
        threshold = 100  # 100 mV threshold
        threshold_adc = int(mV2adc(threshold, ch_range_enum, self.max_adc))
        direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
        status = ps.ps5000aSetSimpleTrigger(
            self.chandle,
            1,                 # Enable trigger
            trigger_source,
            threshold_adc,
            direction,
            0,                 # No trigger delay
            1000               # AutoTrigger in ms if no trigger event
        )
        assert_pico_ok(status)
        print(f"Trigger set on {self.trigger_channel} (Rising edge, threshold = {threshold} mV).")

        # Calculates total samples for the chosen duration and sampling rate
        total_samples = int(self.sampling_rate * self.duration)

        # Captures all data after the trigger
        pre_trigger_samples = 0
        post_trigger_samples = total_samples
        self.max_samples = total_samples

        # Calculates timebase for the chosen sampling rate
        timebase = int(1e9 / self.sampling_rate)
        
        # Prepares variables for the ps5000aGetTimebase2 call
        time_interval_ns = ctypes.c_float()
        returned_max_samples = ctypes.c_int32()
        
        # Configures the timebase in the driver
        status = ps.ps5000aGetTimebase2(
            self.chandle,
            timebase,
            self.max_samples,
            ctypes.byref(time_interval_ns),
            ctypes.byref(returned_max_samples),
            0
        )
        assert_pico_ok(status)
        
        # Prints sampling summary
        print(
            f"Sampling at {self.sampling_rate/1e6:.1f} MS/s "
            f"({self.sampling_rate/1e9:.1f} GS/s) for {self.duration*1e6:.1f} µs.\n"
            f"Collecting {self.max_samples} samples with timebase={timebase}."
        )

        # Stores channel range enum for later use
        self.ch_range_enum = ch_range_enum

    def start_acquisition(self, round_number=None):
        print("Waiting for trigger...")

        # Starts block capture with all samples after the trigger event
        status = ps.ps5000aRunBlock(
            self.chandle,
            0,                # 0 pre-trigger samples
            self.max_samples, # all samples post-trigger
            2,                # segment index
            None,             # timeIndisposedMs
            0,                # lpReady callback
            None,             # pParameter
            None
        )
        assert_pico_ok(status)

        # Waits until data is ready or timeout
        ready = ctypes.c_int16(0)
        timeout = time.time() + 2  # 2-second timeout
        while not ready.value:
            ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
            if time.time() > timeout:
                print("Trigger not detected within timeout period.")
                return False

        print("Trigger detected. Collecting data...")

        # Allocates buffer for data
        buffer = (ctypes.c_int16 * self.max_samples)()
        channel = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.channel}"]
        
        # Assigns buffer to the chosen channel
        status = ps.ps5000aSetDataBuffer(
            self.chandle,
            channel,
            ctypes.byref(buffer),
            self.max_samples,
            0,  # segment index
            0   # downSampleRatioMode
        )
        assert_pico_ok(status)

        # Retrieves captured data from the scope
        overflow = ctypes.c_int16()
        num_samples = ctypes.c_uint32(self.max_samples)
        status = ps.ps5000aGetValues(
            self.chandle,
            0,
            ctypes.byref(num_samples),
            1,  # downSampleRatio
            0,  # downSampleRatioMode
            0,  # segment index
            ctypes.byref(overflow)
        )
        assert_pico_ok(status)

        # Converts data to millivolts
        voltage_buffer = adc2mV(buffer, self.ch_range_enum, self.max_adc)
        print(f"Captured data: {voltage_buffer[:10]}... (first 10 samples)")

        # Saves the data to disk
        self.save_data(voltage_buffer, round_number)
        return True

    def save_data(self, buffer, round_number=None):
        # Generates a timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Builds the optional prefix string
        prefix_str = f"{self.filename_prefix}_" if self.filename_prefix else ""

        # Chooses filename based on round_number
        if round_number is not None:
            filename = f"{prefix_str}data_round_{round_number}_{timestamp}.npy"
        else:
            filename = f"{prefix_str}data_{timestamp}.npy"

        # Constructs full path and saves
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, buffer)
        print(f"Data saved to {filepath}.")

    def stop_acquisition(self):
        # Stops the PicoScope
        status = ps.ps5000aStop(self.chandle)
        assert_pico_ok(status)
        print("PicoScope stopped.")

    def close(self):
        # Closes the PicoScope connection
        if self.chandle:
            status = ps.ps5000aCloseUnit(self.chandle)
            assert_pico_ok(status)
            print("PicoScope connection closed.")
