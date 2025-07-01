import time
import threading
from pi.pi_bare_metal_api import PiBareMetalAPI

def continuous_read(pi_api):
    """
    Continuously read from the Pi's serial port and print each incoming line.
    This runs in a background thread.
    """
    while pi_api.ser.is_open:
        try:
            line = pi_api.ser.readline()
            if line:
                print("PI>", line.decode(errors='ignore').rstrip())
        except Exception as e:
            print("Error reading from Pi:", e)
            break

def pi_no_acquisition():
    """
    Simple script to interact with the Pi's bare-metal AES kernel over serial,
    without using any PicoScope acquisition.
    We set a plaintext and run 'aes <num>' to test the command interface.
    """
    # Adjust port/baud as needed. For example: port="COM7", baud=115200 
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)

    # Start a background thread for continuous reading from the Pi.
    reader_thread = threading.Thread(target=continuous_read, args=(pi_api,), daemon=True)
    reader_thread.start()

    try:
        print("Testing basic commands on the Pi...")

        # a 16-byte plaintext (as 32 hex chars)
        # e.g. 00112233445566778899AABBCCDDEEFF
        plaintext_hex = "11112233445566778899AABBCCDDEEFF"
        # Uncomment if you wish to set the plaintext:
        # pi_api.set_plaintext(plaintext_hex)
        
        # Read and print some initial lines from the Pi.
        response_lines = pi_api.read_lines(max_lines=5)
        for line in response_lines:
            print("PI>", line.rstrip())

        # Example: run 'aes 5000' => Pi will do 5000 AES encryptions.
        pi_api.run_aes(5000)
        response_lines = pi_api.read_lines(max_lines=10)
        for line in response_lines:
            print("PI>", line.rstrip())

        print("Done sending commands to Pi. You can try different steps below.")

        # Interactive command loop.
        while True:
            user_input = input("\nType a command for Pi (or 'exit'): ")
            if user_input.strip().lower() == "exit":
                break
            pi_api.send_command(user_input)
            # Optionally, if you still want to read some lines immediately after sending,
            # you can use read_lines() here. Otherwise, the background thread is handling it.
            # lines = pi_api.read_lines(max_lines=10)
            # for l in lines:
            #     print("PI>", l.rstrip())

    finally:
        pi_api.close()

if __name__ == "__main__":
    pi_no_acquisition()
