import ctypes
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, mV2adc, adc2mV
import numpy as np
import time
import os

class DataAcquisition:
    """
    DataAcquisition for PicoScope 5000A:
    - Allows user to select device resolution (8, 12, 14, 15, 16 bits)
    - Uses a fixed default timebase=4 unless overridden
    - Configurable trigger, capture channel, voltage range, etc.
    - Provides prepare_block_mode() + capture_block() for block captures.
    - Saves data to .npy (optional) or returns the waveform to the caller.
    """

    def __init__(
        self,
        device_resolution_bits=8,
        timebase=None,
        sampling_rate_hz=1e9,
        capture_duration_s=0.012,
        capture_channel="A",
        trigger_channel="EXT",
        trigger_threshold_mV=100,
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="100MV",
        output_dir="data",
        filename_prefix=None,
        # --- NEW: bandwidth limiter + analogue offset ---
        bandwidth_limit_20mhz=False,     # enable HW 20 MHz bandwidth filter on capture channel
        analogue_offset_mv=0.0           # DC offset to apply on capture channel (requires DC coupling)
    ):
        """
        :param device_resolution_bits: ADC resolution in bits (8..16). Must be supported by your 5000A model.
        :param timebase: Override for the scope's timebase (0..something). If None, default=4 is used.
        :param sampling_rate_hz: A nominal sampling rate (not used if we pin timebase=4).
        :param capture_duration_s: How long to capture, in seconds, for computing total samples.
        :param capture_channel: e.g. "A"
        :param trigger_channel: e.g. "B" or "EXT"
        :param trigger_threshold_mV: trigger level in mV
        :param trigger_delay_samples: how many post-trigger samples to offset
        :param auto_trigger_ms: auto-trigger if no real trigger event in X ms
        :param coupling_mode: "AC" or "DC"
        :param voltage_range: "100MV","200MV","2V","5V", etc.
        :param output_dir: folder to save .npy waveforms if desired
        :param filename_prefix: optional prefix for saved files
        :param bandwidth_limit_20mhz: enable 20 MHz bandwidth limiter on the capture channel
        :param analogue_offset_mv: analogue DC offset (in mV) to apply on the capture channel (DC coupling!)
        """
        self.device_resolution_bits = device_resolution_bits
        self.timebase_user = timebase
        self.sampling_rate_hz = sampling_rate_hz
        self.capture_duration_s = capture_duration_s
        self.capture_channel = capture_channel
        self.trigger_channel = trigger_channel
        self.trigger_threshold_mV = trigger_threshold_mV
        self.trigger_delay_samples = trigger_delay_samples
        self.auto_trigger_ms = auto_trigger_ms
        self.coupling_mode = coupling_mode
        self.voltage_range = voltage_range

        self.output_dir = output_dir
        self.filename_prefix = filename_prefix

        # NEW
        self.bandwidth_limit_20mhz = bool(bandwidth_limit_20mhz)
        self.analogue_offset_mv = float(analogue_offset_mv)

        # PicoScope handle
        self.scope_handle = ctypes.c_int16()
        os.makedirs(self.output_dir, exist_ok=True)

        # Max ADC value for 5000A (used by helper conversions)
        self.max_adc = ctypes.c_int16(32512)

        self.max_samples = 0
        self.ch_range_enum = None
        self.timebase_final = 4          # default fallback
        self.time_interval_ns = 0.0      # store actual Δt from driver

    def setup_picoscope(self):
        """
        Opens the device, sets resolution, configures channel and trigger.
        Sets a default timebase=4 if user didn't specify one.
        """
        # (1) Open the scope
        status_open = ps.ps5000aOpenUnit(ctypes.byref(self.scope_handle), None, 1)

        # numeric Pico status codes (not exported as attributes in ps5000a Python module)
        PICO_POWER_SUPPLY_NOT_CONNECTED = 286
        PICO_POWER_SUPPLY_CONNECTED     = 282

        if status_open in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_POWER_SUPPLY_CONNECTED):
            status_change = ps.ps5000aChangePowerSource(self.scope_handle, status_open)
            assert_pico_ok(status_change)
            print("Power source issue corrected.")
        else:
            assert_pico_ok(status_open)

        print("PicoScope connected successfully.")

        # (2) Set device resolution
        resolution_enum = ps.PS5000A_DEVICE_RESOLUTION.get(f"PS5000A_DR_{self.device_resolution_bits}BIT", None)
        if resolution_enum is None:
            raise ValueError(f"Unsupported resolution: {self.device_resolution_bits} bits")

        status_res = ps.ps5000aSetDeviceResolution(self.scope_handle, resolution_enum)
        assert_pico_ok(status_res)
        print(f"Device resolution set to {self.device_resolution_bits} bits.")

        # (3) Configure the capture channel
        channel_index = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        coupling_enum = ps.PS5000A_COUPLING[f"PS5000A_{self.coupling_mode}"]
        channel_range_enum = ps.PS5000A_RANGE[f"PS5000A_{self.voltage_range.upper()}"]

        # Query allowed analogue offset range for this range/coupling and clamp requested offset
        max_off = ctypes.c_float()
        min_off = ctypes.c_float()
        status_off = ps.ps5000aGetAnalogueOffset(
            self.scope_handle, channel_range_enum, coupling_enum,
            ctypes.byref(max_off), ctypes.byref(min_off)
        )
        assert_pico_ok(status_off)
        analogue_offset_v = max(min(self.analogue_offset_mv/1000.0, max_off.value), min_off.value)

        status_ch = ps.ps5000aSetChannel(
            self.scope_handle,
            channel_index,
            1,                # enabled
            coupling_enum,
            channel_range_enum,
            analogue_offset_v # offset in V (DC coupling required to have effect)
        )
        assert_pico_ok(status_ch)
        self.ch_range_enum = channel_range_enum
        print(
            f"Channel {self.capture_channel} configured: {self.coupling_mode} coupling, "
            f"±{self.voltage_range} range, offset={analogue_offset_v*1e3:.1f} mV."
        )

        # (3b) Optional 20 MHz bandwidth limiter on capture channel
        try:
            bw_enum = (ps.PS5000A_BANDWIDTH_LIMITER["PS5000A_BW_20MHZ"]
                    if self.bandwidth_limit_20mhz
                    else ps.PS5000A_BANDWIDTH_LIMITER["PS5000A_BW_FULL"])
            status_bw = ps.ps5000aSetBandwidthFilter(self.scope_handle, channel_index, bw_enum)
            assert_pico_ok(status_bw)
            print(f"Bandwidth limiter: {'20 MHz' if self.bandwidth_limit_20mhz else 'Full'}")
        except Exception as e:
            print(f"[Warn] Could not set bandwidth filter (falling back to default): {e}")

        # (4) Configure trigger – External via numeric enum + proper scaling
        trig_name = str(self.trigger_channel).upper()
        if trig_name in ("EXT", "EXTERNAL"):
            trig_src = 4  # PS5000A_EXTERNAL
            # EXT is ±5 V input range; mV2adc with 5V range maps to ±max_adc correctly
            ext_range_enum = ps.PS5000A_RANGE["PS5000A_5V"]
            threshold_adc = int(mV2adc(self.trigger_threshold_mV, ext_range_enum, self.max_adc))
        else:
            trig_src = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.trigger_channel}"]
            threshold_adc = int(mV2adc(self.trigger_threshold_mV, self.ch_range_enum, self.max_adc))

        direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
        status_trig = ps.ps5000aSetSimpleTrigger(
            self.scope_handle, 1, trig_src, threshold_adc, direction,
            self.trigger_delay_samples, self.auto_trigger_ms
        )
        assert_pico_ok(status_trig)
        print(
            f"Trigger on {self.trigger_channel}, threshold={self.trigger_threshold_mV} mV, "
            f"delaySamples={self.trigger_delay_samples}, autoTrigger={self.auto_trigger_ms} ms."
        )

        # (5) Determine timebase  – unverändert bis zur Wahl von self.timebase_final
        if self.timebase_user is not None:
            self.timebase_final = self.timebase_user
            print(f"Using user-specified timebase={self.timebase_final}.")
        else:
            self.timebase_final = 4
            print(f"No user timebase. Defaulting to timebase={self.timebase_final}.")

        # (6) tatsächliche Δt ermitteln und daraus max_samples berechnen – ersetzt bisherigen Block
        time_interval_ns = ctypes.c_float()
        returned_max_samples = ctypes.c_int32()

        # Erst mit minimaler Samplezahl Δt abfragen
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.scope_handle, self.timebase_final, 1,
            ctypes.byref(time_interval_ns), ctypes.byref(returned_max_samples), 0
        ))
        self.time_interval_ns = float(time_interval_ns.value)

        # gewünschte Sampleanzahl aus Dauer / Δt ableiten
        self.max_samples = int(max(1, round(self.capture_duration_s / (self.time_interval_ns * 1e-9))))
        print(f"Driver reports ~{self.time_interval_ns:.2f} ns/sample -> requesting {self.max_samples} samples.")

        # Mit realer Samplezahl validieren / ggf. begrenzen
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.scope_handle, self.timebase_final, self.max_samples,
            ctypes.byref(time_interval_ns), ctypes.byref(returned_max_samples), 0
        ))
        self.max_samples = min(self.max_samples, int(returned_max_samples.value))
        self.time_interval_ns = float(time_interval_ns.value)
        print(f"Timebase={self.timebase_final} confirmed. Δt ≈ {self.time_interval_ns:.2f} ns, max {self.max_samples} samples.")

    def prepare_block_mode(self):
        """
        Arms the scope for block capture (ps5000aRunBlock).
        This starts the acquisition and waits for a trigger event.
        We do NOT wait for 'IsReady' here - that happens in capture_block().
        """
        print("Arming scope with ps5000aRunBlock (Block Mode).")
        # Pre-trigger samples = 0, post-trigger = self.max_samples
        status_run = ps.ps5000aRunBlock(
            self.scope_handle,
            0,                 # preTriggerSamples
            self.max_samples,  # postTriggerSamples
            self.timebase_final,
            None,              # timeIndisposedMs
            0,                 # lpReady callback
            None,
            None
        )
        assert_pico_ok(status_run)
        print("Scope is now waiting for a trigger...")

    def capture_block(self, timeout_s=2.0):
        """
        Waits up to 'timeout_s' seconds for the scope to trigger (or auto-trigger).
        Then retrieves data into a NumPy array of mV. Returns that array.
        """
        is_ready = ctypes.c_int16(0)
        t_deadline = time.time() + timeout_s

        # Wait for the block capture to complete
        while not is_ready.value:
            ps.ps5000aIsReady(self.scope_handle, ctypes.byref(is_ready))
            if time.time() >= t_deadline:
                print(f"Trigger timeout: no trigger within {timeout_s} seconds.")
                return np.array([])  # or None

        print("Trigger event detected, retrieving data...")

        # Prepare a buffer in Python
        buffer_array = (ctypes.c_int16 * self.max_samples)()
        ch_idx = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        status_buf = ps.ps5000aSetDataBuffer(
            self.scope_handle,
            ch_idx,
            ctypes.byref(buffer_array),
            self.max_samples,
            0,  # segmentIndex
            0   # downSampleRatioMode
        )
        assert_pico_ok(status_buf)

        overflow = ctypes.c_int16()
        collected_count = ctypes.c_uint32(self.max_samples)

        status_get = ps.ps5000aGetValues(
            self.scope_handle,
            0,                           # start index
            ctypes.byref(collected_count),
            1,                           # downSampleRatio
            0,                           # ratioMode (PS5000A_RATIO_MODE_NONE)
            0,                           # segmentIndex
            ctypes.byref(overflow)
        )
        assert_pico_ok(status_get)

        # Convert raw ADC counts to millivolts
        voltage_mv = adc2mV(buffer_array, self.ch_range_enum, self.max_adc)
        print(
            f"Acquired {collected_count.value} samples. "
            f"First 10 samples (mV): {voltage_mv[:10]} ..."
        )

        # Optional: if you want to stop acquisition for each trace
        # self.stop_acquisition()

        # Return as a NumPy array
        waveform = np.array(voltage_mv[:collected_count.value], dtype=np.float32)
        return waveform

    def stop_acquisition(self):
        """
        Optional call to stop the scope if running repeated captures.
        """
        status = ps.ps5000aStop(self.scope_handle)
        assert_pico_ok(status)
        print("Scope acquisition stopped.")

    def close(self):
        if self.scope_handle.value != 0:
            try:
                status = ps.ps5000aCloseUnit(self.scope_handle)
                assert_pico_ok(status)
            except Exception:
                # Ignoriere INVALID_HANDLE o. Ä., wenn Open fehlgeschlagen war
                pass
            finally:
                self.scope_handle = ctypes.c_int16(0)
                print("PicoScope connection closed.")
