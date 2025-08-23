#!/usr/bin/env python3
"""
Unified UART client for your bare-metal Pi kernel.

- Binary protocol (quiet; used by acquisition):
    set_state(pt16), encrypt(), encrypt_n(n), get_state() -> 16B
- ASCII helpers (handy for quick manual tests):
    set_plaintext(hex32), run_aes(n), run_dummy(n)

Notes:
- Mini-UART baud depends on the Pi core clock, so keep it steady in config.txt.
- pySerial's read() can return fewer bytes when a timeout is set; we check for 16.
"""

import serial
import time
import struct

#  Binary command bytes (match the hybrid kernel) 
CMD_SET_STATE = 0x01    # +16 bytes
CMD_GET_STATE = 0x02    # -> 16 bytes
CMD_ENCRYPT   = 0x03    # run AES once (GPIO16 toggled in firmware)
CMD_ENCRYPT_N = 0x04    # +uint16 (big-endian): run AES N times


class PiBareMetalAPI:
    """
    UART interface to the Pi bare-metal AES kernel.

    Typical flow (binary):
        set_state(pt16); encrypt(); ct = get_state()
    """

    def __init__(self, port: str = "COM7", baudrate: int = 115200, timeout: float = 1.0):
        # Tip: pin the core clock in config.txt to keep mini-UART baud stable.
        # (e.g., set core_freq/core_freq_min as needed; see docs)
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.20)                 # small settle after open
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    #  Binary protocol (used by acquisition) 
    def set_state(self, pt16: bytes) -> None:
        if len(pt16) != 16:
            raise ValueError("Plaintext must be 16 bytes.")
        self.ser.write(bytes([CMD_SET_STATE]) + pt16)
        self.ser.flush()

    def encrypt(self) -> None:
        self.ser.write(bytes([CMD_ENCRYPT]))
        self.ser.flush()

    def encrypt_n(self, n: int) -> None:
        if not (0 <= n <= 0xFFFF):
            raise ValueError("n must be in 0..65535")
        self.ser.write(bytes([CMD_ENCRYPT_N]) + struct.pack(">H", n))
        self.ser.flush()

    def get_state(self) -> bytes:
        self.ser.write(bytes([CMD_GET_STATE]))
        self.ser.flush()
        ct = self.ser.read(16)           # may return <16 if timeout hit
        if len(ct) != 16:
            raise TimeoutError("Timed out reading ciphertext (16 bytes).")
        return ct

    #  ASCII helpers (optional) 
    def flush_input(self) -> None:
        """Dump any pending ASCII text (useful right after connect)."""
        out = b""
        while self.ser.in_waiting:
            out += self.ser.read(self.ser.in_waiting)
        if out:
            try:
                print(out.decode(errors="ignore"), end="")
            except Exception:
                pass

    def send_command(self, cmd_str: str) -> None:
        """Send an ASCII command (e.g., 'plaintext 0011..EEFF')."""
        self.ser.write((cmd_str.strip() + "\r\n").encode("ascii"))
        self.ser.flush()
        time.sleep(0.05)

    def set_plaintext(self, plaintext_hex: str) -> None:
        if len(plaintext_hex) != 32:
            raise ValueError("Plaintext must be exactly 32 hex digits.")
        self.send_command(f"plaintext {plaintext_hex}")

    def run_aes(self, iterations: int) -> None:
        self.send_command(f"aes {iterations}")

    def run_dummy(self, iterations: int) -> None:
        self.send_command(f"dummy {iterations}")

    #  Cleanup 
    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass
