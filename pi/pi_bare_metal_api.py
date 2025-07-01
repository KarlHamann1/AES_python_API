import serial
import time

class PiBareMetalAPI:
    """
    Interface for sending commands to the Raspberry Pi's bare-metal AES kernel over UART.
    Commands:
    - plaintext <32-hex>
    - aes <num>
    - help
    """

    def __init__(self, port="COM7", baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.75)  # small delay to let Pi boot or get ready
        print(f"Connected to Pi on {port} at {baudrate} baud.")

        # Optionally flush any initial output from the Pi
        self.flush_input()

    def flush_input(self):
        """Reads/clears any pending output from the Pi so subsequent reads are fresh."""
        while self.ser.in_waiting:
            line = self.ser.readline()
            print("Flushed:", line.decode(errors='ignore'), end='')

    def send_command(self, cmd_str: str):
        """
        Sends a string command (e.g. 'plaintext 00112233445566778899AABBCCDDEEFF') 
        plus a newline to the Pi.
        """
        command_with_newline = cmd_str.strip() + "\r\n"
        self.ser.write(command_with_newline.encode('ascii'))
        time.sleep(0.1)

    def read_lines(self, max_lines=10):
        """
        Reads up to 'max_lines' lines from Pi. 
        Returns them as a list of decoded strings. Non-blocking if timeout is set.
        """
        lines = []
        for _ in range(max_lines):
            if self.ser.in_waiting == 0:
                break
            line = self.ser.readline().decode(errors='ignore').rstrip("\r\n")
            lines.append(line)
        return lines

    def set_plaintext(self, plaintext_hex: str):
        """
        plaintext_hex: exactly 32 hex chars, e.g. '00112233445566778899AABBCCDDEEFF'
        Sends 'plaintext <32-hex>'.
        """
        if len(plaintext_hex) != 32:
            raise ValueError("Plaintext must be exactly 32 hex digits.")
        cmd = f"plaintext {plaintext_hex}"
        self.send_command(cmd)

    def run_aes(self, iterations: int):
        """
        Tells the Pi to run 'aes <num>' where <num> is the number of times to encrypt.
        """
        cmd = f"aes {iterations}"
        self.send_command(cmd)
        
    def run_dummy(self, iterations: int):
        cmd = f"dummy {iterations}"
        self.send_command(cmd)

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("Closed connection to Pi.")
