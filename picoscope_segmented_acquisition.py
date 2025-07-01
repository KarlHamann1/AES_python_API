import ctypes
import time
import os
import numpy as np

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, adc2mV, mV2adc

class SegmentedPicoScopeAcquisition:
    """
    Demonstrates segmented captures with a PicoScope 5000A.
    - Allows multiple triggers in a single runBlock call (one segment per trigger).
    - Then retrieves each segment individually.
    """

    def __init__(
        self,
        device_resolution_bits=12,
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        coupling_mode="AC",
        voltage_range="20MV",
        capture_duration_s=0.00023,   # e.g. 230 us
        sampling_rate_hz=62.5e6,      # ~16 ns per sample if timebase=4, in many 5k models
        timebase=None,                # if None, we choose a default (often 4)
        output_dir="data_pi_encryption",
        filename_prefix="segmented_test",
    ):
        """
        :param device_resolution_bits: e.g. 8,12,14..  Must be supported by your 5k model
        :param capture_channel: e.g. "A"
        :param trigger_channel: e.g. "B"
        :param trigger_threshold_mV: e.g. 100 mV
        :param coupling_mode: "AC" or "DC"
        :param voltage_range: e.g. "20MV", "200MV", "2V", etc.
        :param capture_duration_s: total time per segment capture
        :param sampling_rate_hz: approximate sample rate (for memory calc)
        :param timebase: driver timebase override (integer). If None, use a default (like 4).
        :param output_dir: folder for saving waveforms
        :param filename_prefix: prefix for saved files
        """
        self.device_resolution_bits = device_resolution_bits
        self.capture_channel = capture_channel
        self.trigger_channel = trigger_channel
        self.trigger_threshold_mV = trigger_threshold_mV
        self.coupling_mode = coupling_mode
        self.voltage_range = voltage_range
        self.capture_duration_s = capture_duration_s
        self.sampling_rate_hz = sampling_rate_hz
        self.timebase_user = timebase

        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        os.makedirs(self.output_dir, exist_ok=True)

        self.scope_handle = ctypes.c_int16(0)
        # For 8-bit mode on a 5k, 32512 is typical. The driver auto-scales for higher resolutions.
        self.max_adc = ctypes.c_int16(32512)

        self._timebase_in_use = None
        self.max_samples_per_segment = 0
        self.ch_range_enum = None

        # We'll store how many segments we set up in "setup_segmented_mode()"
        self.num_segments = 1

    def open_unit_and_setup(self):
        """Open the device, set resolution, channels, timebase, etc."""
        # 1) Open
        status_open = ps.ps5000aOpenUnit(ctypes.byref(self.scope_handle), None, 1)
        try:
            assert_pico_ok(status_open)
            print("PicoScope 5000A opened successfully.")
        except:
            if status_open in [286, 282]:
                # Possibly need to change power source
                ps.ps5000aChangePowerSource(self.scope_handle, status_open)
                print("Power source changed, re-try if needed.")
            else:
                raise

        # 2) Resolution
        resolution_enum = ps.PS5000A_DEVICE_RESOLUTION.get(
            f"PS5000A_DR_{self.device_resolution_bits}BIT", None
        )
        if resolution_enum is None:
            raise ValueError(f"Unsupported resolution: {self.device_resolution_bits} bits")

        status_res = ps.ps5000aSetDeviceResolution(self.scope_handle, resolution_enum)
        assert_pico_ok(status_res)
        print(f"Resolution set to {self.device_resolution_bits} bits.")

        # 3) Channels
        ch_index = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        coupling_enum = ps.PS5000A_COUPLING[f"PS5000A_{self.coupling_mode}"]
        range_enum = ps.PS5000A_RANGE[f"PS5000A_{self.voltage_range.upper()}"]

        status_ch = ps.ps5000aSetChannel(
            self.scope_handle,
            ch_index,
            1,  # enabled
            coupling_enum,
            range_enum,
            0.0  # offset
        )
        assert_pico_ok(status_ch)
        self.ch_range_enum = range_enum
        print(f"Channel {self.capture_channel} configured: {self.coupling_mode}, ±{self.voltage_range}.")

        # 4) Simple Trigger
        trig_index = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.trigger_channel}"]
        threshold_adc = int(mV2adc(self.trigger_threshold_mV, range_enum, self.max_adc))
        direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]

        status_trig = ps.ps5000aSetSimpleTrigger(
            self.scope_handle,
            1,  # enable
            trig_index,
            threshold_adc,
            direction,
            0,   # delay (samples)
            1000 # auto trigger ms (1s)
        )
        assert_pico_ok(status_trig)
        print(f"Trigger on {self.trigger_channel} at {self.trigger_threshold_mV} mV, 1s auto trigger fallback.")

        # 5) Decide on timebase
        if self.timebase_user is not None:
            self._timebase_in_use = self.timebase_user
        else:
            # For 5k in 12-bit mode, timebase=4 often ~16ns sample.
            self._timebase_in_use = 4
        print(f"Using timebase={self._timebase_in_use}.")

        # 6) Estimate how many samples per segment
        #    If we want ~230us at 62.5MHz => ~14,375 samples
        self.max_samples_per_segment = int(self.sampling_rate_hz * self.capture_duration_s)

        # Check with ps5000aGetTimebase2
        time_interval_ns = ctypes.c_float()
        returned_max_samples = ctypes.c_int32()
        status_tb = ps.ps5000aGetTimebase2(
            self.scope_handle,
            self._timebase_in_use,
            self.max_samples_per_segment,
            ctypes.byref(time_interval_ns),
            ctypes.byref(returned_max_samples),
            0
        )
        assert_pico_ok(status_tb)
        real_interval_ns = time_interval_ns.value
        print(f"Timebase={self._timebase_in_use} => ~{real_interval_ns:.2f} ns/sample. "
            f"maxSamplesPossible={returned_max_samples.value}")

    def setup_segmented_mode(self, num_segments):
        """
        Allocate memory for 'num_segments' captures, one per trigger event.
        """
        self.num_segments = num_segments

        returnedMaxSamples = ctypes.c_int32(0)
        # This splits the scope memory into 'num_segments' segments
        status_mem = ps.ps5000aMemorySegments(
            self.scope_handle, num_segments, ctypes.byref(returnedMaxSamples)
        )
        assert_pico_ok(status_mem)

        # Then specify how many captures we want
        status_capt = ps.ps5000aSetNoOfCaptures(self.scope_handle, num_segments)
        assert_pico_ok(status_capt)

        print(f"Segmented mode: {num_segments} segments. Each segment can store up to {returnedMaxSamples.value} samples.")

    def run_segmented_capture(self):
        """
        Start a single block capture that can accommodate 'num_segments' triggers.
        Wait until all are triggered or auto-trigger hits.
        Returns True if successful, False on timeout or error.
        """
        post_trigger_samples = self.max_samples_per_segment
        pre_trigger_samples = 0

        print(f"Starting runBlock for {self.num_segments} triggers, up to {post_trigger_samples} samples/segment.")

        # runBlock
        status_run = ps.ps5000aRunBlock(
            self.scope_handle,
            pre_trigger_samples,
            post_trigger_samples,
            self._timebase_in_use,
            None,  # timeIndisposed
            0,     # lpReady callback
            None,
            None
        )
        assert_pico_ok(status_run)

        # Wait for readiness or time out
        is_ready = ctypes.c_int16(0)
        t0 = time.time()
        timeout_sec = 3 + self.num_segments*0.05  # e.g. ~3 + 0.05/segment
        while not is_ready.value:
            ps.ps5000aIsReady(self.scope_handle, ctypes.byref(is_ready))
            if (time.time() - t0) > timeout_sec:
                print("Timeout waiting for segmented capture to finish.")
                return False
            time.sleep(0.01)

        print("All segments captured (or auto-triggered).")
        return True

    def retrieve_segment(self, segment_index):
        """
        Retrieve data for a single segment as a NumPy array of mV.
        """
        # Prepare a buffer
        buffer_array = (ctypes.c_int16 * self.max_samples_per_segment)()

        ch_idx = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        # Map the buffer to the segment
        status_buf = ps.ps5000aSetDataBuffer(
            self.scope_handle,
            ch_idx,
            ctypes.byref(buffer_array),
            self.max_samples_per_segment,
            segment_index,
            0  # downSampleRatioMode = none
        )
        assert_pico_ok(status_buf)

        overflow = ctypes.c_int16(0)
        collected_count = ctypes.c_uint32(self.max_samples_per_segment)

        status_get = ps.ps5000aGetValues(
            self.scope_handle,
            0,  # start index
            ctypes.byref(collected_count),
            1,  # downSampleRatio
            0,  # ratioMode=NONE
            segment_index,
            ctypes.byref(overflow)
        )
        assert_pico_ok(status_get)

        # Convert to mV
        voltages = adc2mV(buffer_array, self.ch_range_enum, self.max_adc)
        return np.array(voltages[:collected_count.value], dtype=np.float32)

    # In picoscope_segmented_acquisition.py

    def save_segment_data(self, waveform, seg_idx, run_idx=None):
        """
        Saves one segment's waveform to disk as a .npy file.
        This version omits any timestamp and uses a shorter filename.
        
        Example filenames:
        - With run_idx:  "prefix_round_003_segment_005.npy"
        - Without run_idx: "prefix_segment_005.npy"
        """
        if run_idx is not None:
            # Wenn eine Runden-Nummer übergeben wurde
            fname = f"{self.filename_prefix}_round_{run_idx:03d}_segment_{seg_idx:03d}.npy"
        else:
            # Ohne Rundennummer
            fname = f"{self.filename_prefix}_segment_{seg_idx:03d}.npy"

        path = os.path.join(self.output_dir, fname)
        np.save(path, waveform)
        print(f"Segment {seg_idx}, run {run_idx}, saved to {path}")



    def stop(self):
        """Stop the scope (if needed)."""
        ps.ps5000aStop(self.scope_handle)

    def close(self):
        """Close the scope."""
        if self.scope_handle.value != 0:
            ps.ps5000aCloseUnit(self.scope_handle)
            self.scope_handle = ctypes.c_int16(0)
            print("PicoScope closed.")
