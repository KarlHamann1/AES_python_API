from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok
import numpy as np
import time
import os

class DataAcquisition:
    """
    Handles data acquisition using a Picoscope with metadata and unique filenames.
    """
    def __init__(self, sampling_rate=1e6, duration=0.01, channel="A", trigger_channel="EXT", output_dir="data"):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.channel = channel
        self.trigger_channel = trigger_channel
        self.output_dir = output_dir
        self.scope = None
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

    def setup_picoscope(self):
        # Initialize and configure the Picoscope
        self.scope = ps.PS5000a()
        assert_pico_ok(self.scope.handle)

        # Set up main channel for signal recording
        self.scope.setChannel(self.channel, enabled=True, coupling="DC", range=5)
        print(f"Channel {self.channel} configured for data.")

        # Configure the trigger channel (EXT input)
        self.scope.setSimpleTrigger(self.trigger_channel, threshold=0, direction="RISING")
        print(f"Trigger set on {self.trigger_channel} (Rising edge, threshold = 0V).")

        # Configure sampling and buffer size
        self.scope.setSamplingInterval(1 / self.sampling_rate, int(self.sampling_rate * self.duration))
        print(f"Sampling configured: {self.sampling_rate} Hz for {self.duration}s.")

    def start_acquisition(self, round_number=None):
        # Start acquisition on hardware trigger
        print("Waiting for trigger...")
        self.scope.runBlock()
        self.scope.waitReady()

        # Record the start time
        start_time = time.time()

        # Capture data once triggered
        buffer = self.scope.getDataV(self.channel, int(self.sampling_rate * self.duration))
        end_time = time.time()

        print(f"Data captured: {buffer[:5]}...")  # Show first 5 data points

        # Save captured data with metadata
        self.save_data(buffer, start_time, end_time, round_number)

    def save_data(self, buffer, start_time, end_time, round_number=None):
        # Generate a unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.npy"
        if round_number is not None:
            filename = f"data_round_{round_number}_{timestamp}.npy"
        filepath = os.path.join(self.output_dir, filename)

        # Save data and metadata
        metadata = {
            "sampling_rate": self.sampling_rate,
            "duration": self.duration,
            "start_time": start_time,
            "end_time": end_time,
            "channel": self.channel,
            "trigger_channel": self.trigger_channel,
        }
        np.save(filepath, {"data": buffer, "metadata": metadata})
        print(f"Data saved to {filepath}.")

    def stop_acquisition(self):
        print("Stopping acquisition.")
        self.scope.stop()

    def close(self):
        if self.scope:
            self.scope.close()
            print("Picoscope connection closed.")
