#!/usr/bin/env python3
"""
Minimal UART helper for the Arduino AES target.

Binary protocol (1-byte opcodes):
'0' (=0x30): SET_STATE  + 16 data bytes
'1' (=0x31): GET_STATE  -> 16 data bytes
'2' (=0x32): ENCRYPT    (no payload/response)

Notes
- We clear the input buffer on connect to avoid stale bytes from the Arduino reset.
- Reads loop until the exact number of bytes arrive or a total timeout elapses.
"""

import time
import serial
from typing import Optional


def _read_exact(ser: serial.Serial, n: int, total_timeout_s: float, err_msg: str) -> bytes:
    """Read exactly n bytes or raise TimeoutError after total_timeout_s."""
    buf = bytearray()
    deadline = time.time() + total_timeout_s
    while len(buf) < n:
        if time.time() > deadline:
            raise TimeoutError(err_msg)
        # pySerial read() may return fewer than requested when timeout is set.
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        # else: no sleep needed; Serial.read() already honors per-call timeout
    return bytes(buf)


class EncipherAPI:
    """Tiny client for the Arduino-side AES firmware."""

    CMD_SET_STATE = b"0"
    CMD_GET_STATE = b"1"
    CMD_ENCRYPT   = b"2"

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0, *, verbose: bool = True):
        """
        Open the serial port.
        - timeout: per-read timeout used by pySerial (seconds)
        - verbose: print short connection status messages
        """
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.5)                 # Arduino typically auto-resets on open
        self.ser.reset_input_buffer()   # drop banner/boot noise
        if verbose:
            print(f"[EncipherAPI] Connected on {port} @ {baudrate} Bd")

    # Context manager sugar:  with EncipherAPI(...) as enc: ...
    def __enter__(self) -> "EncipherAPI":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- Commands -------------------------------------------------
    def set_state(self, state: bytes) -> None:
        """Send 16-byte plaintext/state."""
        if len(state) != 16:
            raise ValueError("state must be exactly 16 bytes")
        self.ser.write(self.CMD_SET_STATE + state)

    def get_state(self, *, total_timeout_s: float = 2.0) -> bytes:
        """Request the 16-byte state/ciphertext and return it."""
        self.ser.write(self.CMD_GET_STATE)
        data = _read_exact(self.ser, 16, total_timeout_s, "ciphertext timeout")
        self.ser.reset_input_buffer()  # keep line clean for next round
        return data

    def encrypt(self) -> None:
        """Trigger one AES encryption (GPIO/trigger handled on device)."""
        self.ser.write(self.CMD_ENCRYPT)

    # --- Housekeeping --------------------------------------------
    def flush_input(self) -> None:
        """Drop any unread bytes (useful after errors)."""
        self.ser.reset_input_buffer()

    def close(self) -> None:
        """Close the serial port (idempotent)."""
        if getattr(self.ser, "is_open", False):
            self.ser.close()
