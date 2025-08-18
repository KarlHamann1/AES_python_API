import time
import threading
from pi.pi_bare_metal_api import PiBareMetalAPI
from picoscope_segmented_acquisition import SegmentedPicoScopeAcquisition

def continuous_read(pi_api):
    """
    Continuously read from the Pi's UART and print lines.
    Runs in a background thread so Pi output is visible in real time.
    """
    while pi_api.ser.is_open:
        try:
            line = pi_api.ser.readline()
            if line:
                print("PI>", line.decode(errors='ignore').rstrip())
        except Exception as e:
            print("Error reading from Pi:", e)
            break

def main():
    # 1) Connect to Pi
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)
    # Start background thread to show Pi output
    reader_thread = threading.Thread(target=continuous_read, args=(pi_api,), daemon=True)
    reader_thread.start()
    time.sleep(0.2)

    # 2) Create the segmented acquisition object
    seg_acq = SegmentedPicoScopeAcquisition(
        device_resolution_bits=12,
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        coupling_mode="AC",
        voltage_range="10MV",
        capture_duration_s=0.00023,   # 230 us
        sampling_rate_hz=62.5e6,      # ~16 ns sample interval
        timebase=4,                   # if your device supports timebase=4 at 12 bits
        output_dir="data_pi_dummy",
        filename_prefix="segmented_dummy_test"
    )

    try:
        # 3) Open the scope and set it up
        seg_acq.open_unit_and_setup()

        # 4) Configure 100 segments
        seg_acq.setup_segmented_mode(num_segments=100)

        # Adjust as desired.
        num_runs = 10

        for run_idx in range(num_runs):
            print(f"\n=== Starting segmented acquisition run {run_idx+1}/{num_runs} ===")

            # 5) Arm the scope
            success = seg_acq.run_segmented_capture()
            if not success:
                print("Scope timed out waiting for triggers. Skipping this run.")
                seg_acq.stop()
                continue

            # 6) Call the dummy program on the Pi
            #    e.g. 'dummy 100' => Pi toggles the trigger line once per iteration
            print(f"Sending 'dummy 100' command to Pi for run {run_idx+1}...")
            pi_api.run_dummy(100)

            # 7) Retrieve and save all 100 segments
            for seg_idx in range(100):
                waveform = seg_acq.retrieve_segment(seg_idx)
                seg_acq.save_segment_data(waveform, seg_idx, run_idx=run_idx)

            # 8) Stop between runs
            seg_acq.stop()
            print(f"Completed run {run_idx+1}. Captured and saved 100 segments.")

        print(f"\nAll {num_runs} runs completed. Total {num_runs*100} dummy traces collected.")

    finally:
        seg_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
