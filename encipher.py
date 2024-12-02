import serial
import time
#from aes import *
import threading

class EncipherAPI:
    """
    Interface for controlling Arduino for AES encryption.
    """
    CMD_SET_STATE = b'0'
    CMD_GET_STATE = b'1'
    CMD_ENCRYPT = b'2'

    def __init__(self, port, baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(2)
        print(f"Connected to Arduino on {port}")

    def set_state(self, state: bytes):
        assert len(state) == 16, "State must be exactly 16 bytes."
        self.ser.write(self.CMD_SET_STATE)
        self.ser.write(state)
        print("State set.")

    def get_state(self) -> bytes:
        self.ser.write(self.CMD_GET_STATE)
        state = self.ser.read(16)
        print("State received:", state)
        return state

    def encrypt(self):
        self.ser.write(self.CMD_ENCRYPT)
        print("Encryption started.")

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("Connection to Arduino closed.")