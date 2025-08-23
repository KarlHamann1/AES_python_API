"""
PicoScope 5000A block-mode helper.

- I pick a device resolution (8/12/14/15/16 bit).
- Timebase defaults to 4 unless I pass one.
- Configure channel, trigger, optional 20 MHz BW limit, and analogue offset.
- Use prepare_block_mode() then capture_block() to get one waveform (mV) as np.float32.
"""

import ctypes
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, mV2adc, adc2mV
import numpy as np
import time
import os


class DataAcquisition:
    """
    DataAcquisition for PicoScope 5000A.
    Core flow: setup_picoscope() → prepare_block_mode() → capture_block().

    I keep the logic simple and print short status lines so it’s easy to debug runs.
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
        # Optional analog tweaks
        bandwidth_limit_20mhz=False,  # 20 MHz HW filter on the capture channel
        analogue_offset_mv=0.0        # requires DC coupling to take effect
    ):
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

        self.bandwidth_limit_20mhz = bool(bandwidth_limit_20mhz)
        self.analogue_offset_mv = float(analogue_offset_mv)

        # PicoScope handle etc.
        self.scope_handle = ctypes.c_int16()
        os.makedirs(self.output_dir, exist_ok=True)

        # Max ADC value used by helpers
        self.max_adc = ctypes.c_int16(32512)

        self.max_samples = 0
        self.ch_range_enum = None
        self.timebase_final = 4         # default fallback
        self.time_interval_ns = 0.0     # actual Δt from driver

    def setup_picoscope(self):
        """Open device, set resolution, channel, trigger, and timebase."""
        # (1) Open unit (handle power-source edge cases)
        status_open = ps.ps5000aOpenUnit(ctypes.byref(self.scope_handle), None, 1)
        PICO_POWER_SUPPLY_NOT_CONNECTED = 286
        PICO_POWER_SUPPLY_CONNECTED     = 282
        if status_open in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_POWER_SUPPLY_CONNECTED):
            status_change = ps.ps5000aChangePowerSource(self.scope_handle, status_open)
            assert_pico_ok(status_change)
            print("Power source issue corrected.")
        else:
            assert_pico_ok(status_open)
        print("PicoScope connected.")

        # (2) Resolution
        resolution_enum = ps.PS5000A_DEVICE_RESOLUTION.get(f"PS5000A_DR_{self.device_resolution_bits}BIT", None)
        if resolution_enum is None:
            raise ValueError(f"Unsupported resolution: {self.device_resolution_bits} bits")
        assert_pico_ok(ps.ps5000aSetDeviceResolution(self.scope_handle, resolution_enum))
        print(f"Resolution set: {self.device_resolution_bits} bit.")

        # (3) Channel config
        channel_index = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        coupling_enum = ps.PS5000A_COUPLING[f"PS5000A_{self.coupling_mode}"]
        channel_range_enum = ps.PS5000A_RANGE[f"PS5000A_{self.voltage_range.upper()}"]

        # analogue offset: clamp to allowed range for this range/coupling
        max_off = ctypes.c_float()
        min_off = ctypes.c_float()
        assert_pico_ok(ps.ps5000aGetAnalogueOffset(
            self.scope_handle, channel_range_enum, coupling_enum,
            ctypes.byref(max_off), ctypes.byref(min_off)
        ))
        analogue_offset_v = max(min(self.analogue_offset_mv/1000.0, max_off.value), min_off.value)

        assert_pico_ok(ps.ps5000aSetChannel(
            self.scope_handle,
            channel_index,
            1,  # enabled
            coupling_enum,
            channel_range_enum,
            analogue_offset_v
        ))
        self.ch_range_enum = channel_range_enum
        print(
            f"Channel {self.capture_channel}: {self.coupling_mode} coupling, "
            f"±{self.voltage_range}, offset={analogue_offset_v*1e3:.1f} mV."
        )

        # (3b) Optional 20 MHz BW limiter
        try:
            bw_enum = (ps.PS5000A_BANDWIDTH_LIMITER["PS5000A_BW_20MHZ"]
                    if self.bandwidth_limit_20mhz
                    else ps.PS5000A_BANDWIDTH_LIMITER["PS5000A_BW_FULL"])
            assert_pico_ok(ps.ps5000aSetBandwidthFilter(self.scope_handle, channel_index, bw_enum))
            print(f"Bandwidth limiter: {'20 MHz' if self.bandwidth_limit_20mhz else 'Full'}")
        except Exception as e:
            print(f"[Warn] Could not set bandwidth filter; leaving default. ({e})")

        # (4) Trigger (EXT or a channel)
        trig_name = str(self.trigger_channel).upper()
        if trig_name in ("EXT", "EXTERNAL"):
            trig_src = 4  # PS5000A_EXTERNAL
            # EXT input is a fixed range (±5 V), so map mV to ADC using 5 V enum
            ext_range_enum = ps.PS5000A_RANGE["PS5000A_5V"]
            threshold_adc = int(mV2adc(self.trigger_threshold_mV, ext_range_enum, self.max_adc))
        else:
            trig_src = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.trigger_channel}"]
            threshold_adc = int(mV2adc(self.trigger_threshold_mV, self.ch_range_enum, self.max_adc))

        direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self.scope_handle, 1, trig_src, threshold_adc, direction,
            self.trigger_delay_samples, self.auto_trigger_ms
        ))
        print(
            f"Trigger: {self.trigger_channel}, {self.trigger_threshold_mV} mV, "
            f"delaySamples={self.trigger_delay_samples}, auto={self.auto_trigger_ms} ms."
        )

        # (5) Timebase selection
        self.timebase_final = self.timebase_user if self.timebase_user is not None else 4
        print(f"Timebase set: {self.timebase_final}")

        # (6) Query actual dt and max samples for this timebase
        time_interval_ns = ctypes.c_float()
        returned_max_samples = ctypes.c_int32()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.scope_handle, self.timebase_final, 1,
            ctypes.byref(time_interval_ns), ctypes.byref(returned_max_samples), 0
        ))
        self.time_interval_ns = float(time_interval_ns.value)

        # compute desired sample count from duration / dt
        self.max_samples = int(max(1, round(self.capture_duration_s / (self.time_interval_ns * 1e-9))))
        print(f"Driver reports ~{self.time_interval_ns:.2f} ns/sample → requesting {self.max_samples} samples.")

        # validate with the driver and clamp to allowed max
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.scope_handle, self.timebase_final, self.max_samples,
            ctypes.byref(time_interval_ns), ctypes.byref(returned_max_samples), 0
        ))
        self.max_samples = min(self.max_samples, int(returned_max_samples.value))
        self.time_interval_ns = float(time_interval_ns.value)
        print(f"Confirmed: Δt ≈ {self.time_interval_ns:.2f} ns, max {self.max_samples} samples (timebase={self.timebase_final}).")

    def prepare_block_mode(self):
        """Arm the scope for a single block capture (waits for a trigger afterwards)."""
        print("Arming scope (ps5000aRunBlock).")
        status_run = ps.ps5000aRunBlock(
            self.scope_handle,
            0,                 # preTriggerSamples
            self.max_samples,  # postTriggerSamples
            self.timebase_final,
            None,              # timeIndisposedMs
            0,                 # lpReady
            None,
            None
        )
        assert_pico_ok(status_run)
        print("Waiting for trigger...")

    def capture_block(self, timeout_s=2.0):
        """
        Wait for trigger (or auto-trigger) up to timeout_s, then fetch samples.
        Returns a 1-D np.float32 array (mV). Empty array on timeout.
        """
        is_ready = ctypes.c_int16(0)
        deadline = time.time() + timeout_s

        # wait until the device says it's ready
        while not is_ready.value:
            ps.ps5000aIsReady(self.scope_handle, ctypes.byref(is_ready))
            if time.time() >= deadline:
                print(f"Trigger timeout after {timeout_s} s.")
                return np.array([], dtype=np.float32)

        print("Triggered, retrieving data...")

        # set buffer and read back values
        buffer_array = (ctypes.c_int16 * self.max_samples)()
        ch_idx = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        assert_pico_ok(ps.ps5000aSetDataBuffer(
            self.scope_handle, ch_idx,
            ctypes.byref(buffer_array), self.max_samples,
            0,  # segmentIndex
            0   # PS5000A_RATIO_MODE_NONE
        ))

        overflow = ctypes.c_int16()
        fetched = ctypes.c_uint32(self.max_samples)
        assert_pico_ok(ps.ps5000aGetValues(
            self.scope_handle,
            0, ctypes.byref(fetched),
            1,  # downSampleRatio
            0,  # ratioMode (NONE)
            0,  # segmentIndex
            ctypes.byref(overflow)
        ))

        voltage_mv = adc2mV(buffer_array, self.ch_range_enum, self.max_adc)
        print(f"Acquired {fetched.value} samples. First 10 mV: {voltage_mv[:10]} ...")

        return np.array(voltage_mv[:fetched.value], dtype=np.float32)

    def stop_acquisition(self):
        """Optional: stop the scope explicitly."""
        assert_pico_ok(ps.ps5000aStop(self.scope_handle))
        print("Acquisition stopped.")

    def close(self):
        """Close the unit (safe to call multiple times)."""
        if self.scope_handle.value != 0:
            try:
                assert_pico_ok(ps.ps5000aCloseUnit(self.scope_handle))
            except Exception:
                pass  # ignore invalid-handle on failed
            finally:
                self.scope_handle = ctypes.c_int16(0)
                print("PicoScope connection closed.")
