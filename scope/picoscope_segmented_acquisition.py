#!/usr/bin/env python3
"""
Segmented captures with a PicoScope 5000A.

- One runBlock can record multiple triggers (one segment per trigger).
- Afterwards I pull each segment individually via retrieve_segment().
- Added bits I rely on elsewhere: resolution setup, channel config (incl. optional
20 MHz BW limit + analogue offset), proper trigger scaling (EXT vs. channel),
and a timebase check to compute a sane per-segment sample count.
"""

import ctypes
import time
import os
import numpy as np

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, adc2mV, mV2adc


class SegmentedPicoScopeAcquisition:
    """
    Segmented (rapid) block-mode helper for the PS5000A driver.
    Flow:
    open_unit_and_setup() → setup_segmented_mode(N) → run_segmented_capture()
    → retrieve_segment(i) for i in [0..N-1] → stop() / close()
    """

    def __init__(
        self,
        device_resolution_bits=12,
        capture_channel="A",
        trigger_channel="B",         # use "EXT" to trigger on the external input
        trigger_threshold_mV=100,
        coupling_mode="AC",
        voltage_range="20MV",
        capture_duration_s=0.00023,  # e.g. 230 µs
        sampling_rate_hz=62.5e6,     # nominal; I confirm actual dt via GetTimebase2
        timebase=None,               # if None, fallback to a sensible default (often 4)
        output_dir="data_pi_encryption",
        filename_prefix="segmented_test",
        # extras to match the non-segmented helper
        auto_trigger_ms=1000,
        trigger_delay_samples=0,
        bandwidth_limit_20mhz=False, # HW 20 MHz filter on the capture channel
        analogue_offset_mv=0.0       # requires DC coupling to take effect
    ):
        self.device_resolution_bits = device_resolution_bits
        self.capture_channel = capture_channel
        self.trigger_channel = trigger_channel
        self.trigger_threshold_mV = trigger_threshold_mV
        self.coupling_mode = coupling_mode
        self.voltage_range = voltage_range
        self.capture_duration_s = capture_duration_s
        self.sampling_rate_hz = sampling_rate_hz
        self.timebase_user = timebase

        self.auto_trigger_ms = auto_trigger_ms
        self.trigger_delay_samples = trigger_delay_samples
        self.bandwidth_limit_20mhz = bool(bandwidth_limit_20mhz)
        self.analogue_offset_mv = float(analogue_offset_mv)

        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        os.makedirs(self.output_dir, exist_ok=True)

        self.scope_handle = ctypes.c_int16(0)
        # typical max ADC for 5kA series; driver handles scaling by resolution
        self.max_adc = ctypes.c_int16(32512)

        self._timebase_in_use = None
        self.time_interval_ns = 0.0
        self.max_samples_per_segment = 0

        self.ch_range_enum = None        # range for capture channel
        self.trig_range_enum = None      # range used for trigger ADC scaling (EXT or trig channel)
        self.num_segments = 1

    # ─────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────
    def open_unit_and_setup(self):
        """Open the device, set resolution, channels, trigger, and timebase."""
        # 1) Open
        status_open = ps.ps5000aOpenUnit(ctypes.byref(self.scope_handle), None, 1)
        PICO_POWER_SUPPLY_NOT_CONNECTED = 286
        PICO_POWER_SUPPLY_CONNECTED     = 282
        if status_open in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_POWER_SUPPLY_CONNECTED):
            assert_pico_ok(ps.ps5000aChangePowerSource(self.scope_handle, status_open))
            print("Power source issue corrected.")
        else:
            assert_pico_ok(status_open)
        print("PicoScope 5000A opened.")

        # 2) Resolution
        resolution_enum = ps.PS5000A_DEVICE_RESOLUTION.get(
            f"PS5000A_DR_{self.device_resolution_bits}BIT", None
        )
        if resolution_enum is None:
            raise ValueError(f"Unsupported resolution: {self.device_resolution_bits} bits")
        assert_pico_ok(ps.ps5000aSetDeviceResolution(self.scope_handle, resolution_enum))
        print(f"Resolution set to {self.device_resolution_bits} bit.")

        # 3) Capture channel (enable + range/coupling + optional offset)
        ch_idx = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]
        coupling_enum = ps.PS5000A_COUPLING[f"PS5000A_{self.coupling_mode}"]
        range_enum = ps.PS5000A_RANGE[f"PS5000A_{self.voltage_range.upper()}"]

        # clamp analogue offset to supported limits for this range/coupling
        max_off = ctypes.c_float()
        min_off = ctypes.c_float()
        assert_pico_ok(ps.ps5000aGetAnalogueOffset(
            self.scope_handle, range_enum, coupling_enum, ctypes.byref(max_off), ctypes.byref(min_off)
        ))
        analogue_offset_v = max(min(self.analogue_offset_mv/1000.0, max_off.value), min_off.value)

        assert_pico_ok(ps.ps5000aSetChannel(
            self.scope_handle, ch_idx, 1, coupling_enum, range_enum, analogue_offset_v
        ))
        self.ch_range_enum = range_enum
        print(f"Channel {self.capture_channel}: {self.coupling_mode}, ±{self.voltage_range}, offset={analogue_offset_v*1e3:.1f} mV.")

        # optional 20 MHz BW limiter on the capture channel
        try:
            bw_enum = (ps.PS5000A_BANDWIDTH_LIMITER["PS5000A_BW_20MHZ"]
                    if self.bandwidth_limit_20mhz
                    else ps.PS5000A_BANDWIDTH_LIMITER["PS5000A_BW_FULL"])
            assert_pico_ok(ps.ps5000aSetBandwidthFilter(self.scope_handle, ch_idx, bw_enum))
            print(f"Bandwidth limiter: {'20 MHz' if self.bandwidth_limit_20mhz else 'Full'}")
        except Exception as e:
            print(f"[Warn] Could not set bandwidth filter; leaving default. ({e})")

        # 4) Trigger setup
        trig_name = str(self.trigger_channel).upper()
        if trig_name in ("EXT", "EXTERNAL"):
            trig_src = 4  # PS5000A_EXTERNAL
            self.trig_range_enum = ps.PS5000A_RANGE["PS5000A_5V"]  # EXT uses ±5 V scale for ADC mapping
        else:
            trig_src = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.trigger_channel}"]
            # ensure the trigger channel is configured too (use same coupling/range, zero offset)
            trig_ch_idx = trig_src
            assert_pico_ok(ps.ps5000aSetChannel(
                self.scope_handle, trig_ch_idx, 1, coupling_enum, range_enum, 0.0
            ))
            self.trig_range_enum = range_enum

        threshold_adc = int(mV2adc(self.trigger_threshold_mV, self.trig_range_enum, self.max_adc))
        direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self.scope_handle, 1, trig_src, threshold_adc, direction,
            self.trigger_delay_samples, self.auto_trigger_ms
        ))
        print(f"Trigger: {self.trigger_channel}, {self.trigger_threshold_mV} mV, delay={self.trigger_delay_samples}, auto={self.auto_trigger_ms} ms.")

        # 5) Timebase choice
        self._timebase_in_use = self.timebase_user if self.timebase_user is not None else 4
        print(f"Timebase={self._timebase_in_use} (user override{' on' if self.timebase_user is not None else ' off'}).")

        # 6) Confirm dt and compute per-segment sample count
        t_ns = ctypes.c_float()
        max_samps = ctypes.c_int32()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.scope_handle, self._timebase_in_use, 1, ctypes.byref(t_ns), ctypes.byref(max_samps), 0
        ))
        self.time_interval_ns = float(t_ns.value)

        # pick samples from desired duration and actual dt
        desired = int(max(1, round(self.capture_duration_s / (self.time_interval_ns * 1e-9))))
        # validate against driver limits for this timebase
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.scope_handle, self._timebase_in_use, desired, ctypes.byref(t_ns), ctypes.byref(max_samps), 0
        ))
        self.time_interval_ns = float(t_ns.value)
        self.max_samples_per_segment = min(desired, int(max_samps.value))
        print(f"Δt ≈ {self.time_interval_ns:.2f} ns → {self.max_samples_per_segment} samples/segment (max {int(max_samps.value)}).")

    def setup_segmented_mode(self, num_segments: int):
        """Split memory and request this many captures (one per segment)."""
        self.num_segments = int(num_segments)
        returned_max = ctypes.c_int32(0)

        assert_pico_ok(ps.ps5000aMemorySegments(self.scope_handle, self.num_segments, ctypes.byref(returned_max)))
        assert_pico_ok(ps.ps5000aSetNoOfCaptures(self.scope_handle, self.num_segments))

        # clamp per-segment samples if the segment memory is tighter than I thought
        if self.max_samples_per_segment > int(returned_max.value):
            self.max_samples_per_segment = int(returned_max.value)
            print(f"[Info] Clamped samples/segment to {self.max_samples_per_segment} due to segment memory.")

        print(f"Segmented mode ready: {self.num_segments} segments, up to {returned_max.value} samples/segment.")

    # ─────────────────────────────────────────────────────────────────────
    # Acquisition
    # ─────────────────────────────────────────────────────────────────────
    def run_segmented_capture(self) -> bool:
        """
        Start a single runBlock able to record num_segments triggers.
        Return True when acquisition completed (all segments armed + finished),
        False on timeout.
        """
        pre_trigger = 0
        post_trigger = self.max_samples_per_segment

        print(f"runBlock: {self.num_segments} segments, {post_trigger} samples/segment.")
        assert_pico_ok(ps.ps5000aRunBlock(
            self.scope_handle, pre_trigger, post_trigger,
            self._timebase_in_use, None, 0, None, None
        ))

        is_ready = ctypes.c_int16(0)
        t0 = time.time()
        timeout_s = 3 + self.num_segments * 0.05
        while not is_ready.value:
            ps.ps5000aIsReady(self.scope_handle, ctypes.byref(is_ready))
            if time.time() - t0 > timeout_s:
                print("Timeout waiting for segmented capture to finish.")
                return False
            time.sleep(0.01)

        print("Segmented capture complete (or auto-triggered).")
        return True

    def retrieve_segment(self, segment_index: int) -> np.ndarray:
        """Fetch one segment as a float32 array in mV."""
        # buffer for this segment
        buffer_array = (ctypes.c_int16 * self.max_samples_per_segment)()
        ch_idx = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{self.capture_channel}"]

        assert_pico_ok(ps.ps5000aSetDataBuffer(
            self.scope_handle, ch_idx, ctypes.byref(buffer_array),
            self.max_samples_per_segment, segment_index, 0  # ratioMode NONE
        ))

        overflow = ctypes.c_int16(0)
        fetched = ctypes.c_uint32(self.max_samples_per_segment)

        assert_pico_ok(ps.ps5000aGetValues(
            self.scope_handle, 0, ctypes.byref(fetched),
            1, 0, segment_index, ctypes.byref(overflow)
        ))

        mv = adc2mV(buffer_array, self.ch_range_enum, self.max_adc)
        return np.array(mv[:fetched.value], dtype=np.float32)

    def save_segment_data(self, waveform: np.ndarray, seg_idx: int, run_idx: int | None = None):
        """Save one segment to disk as .npy; filenames stay short and readable."""
        if run_idx is not None:
            fname = f"{self.filename_prefix}_round_{run_idx:03d}_segment_{seg_idx:03d}.npy"
        else:
            fname = f"{self.filename_prefix}_segment_{seg_idx:03d}.npy"
        path = os.path.join(self.output_dir, fname)
        np.save(path, waveform)
        print(f"Segment {seg_idx}, run {run_idx}, saved to {path}")

    # ─────────────────────────────────────────────────────────────────────
    # Housekeeping
    # ─────────────────────────────────────────────────────────────────────
    def stop(self):
        """Stop the scope if I want to abort between runs."""
        ps.ps5000aStop(self.scope_handle)

    def close(self):
        """Close the scope (idempotent)."""
        if self.scope_handle.value != 0:
            try:
                assert_pico_ok(ps.ps5000aCloseUnit(self.scope_handle))
            except Exception:
                pass
            finally:
                self.scope_handle = ctypes.c_int16(0)
                print("PicoScope closed.")
