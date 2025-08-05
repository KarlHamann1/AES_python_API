"""
import serial
import time

class EncipherAPI:

    CMD_SET_STATE = b'0'
    CMD_GET_STATE = b'1'
    CMD_ENCRYPT = b'2'

    def __init__(self, port, baudrate=9600, timeout=1):
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
            
"""        
import serial
import time

# --- helper -------------------------------------------------
def _read_exact(ser, n, err):
    buf = bytearray()
    deadline = time.time() + 2        # 2‑s Gesamt‑Timeout
    while len(buf) < n:
        if time.time() > deadline:
            raise TimeoutError(err)
        buf.extend(ser.read(n - len(buf)))
    return bytes(buf)


# --- API ----------------------------------------------------
class EncipherAPI:
    
    CMD_SET_STATE = b'0'
    CMD_GET_STATE = b'1'
    CMD_ENCRYPT   = b'2'

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.5)                       # settle after auto‑reset
        print(f"Connected to Arduino on {port}")

    # --------------------------------------------------------
    def set_state(self, state: bytes) -> None:
        if len(state) != 16:
            raise ValueError("state must be 16 bytes")
        self.ser.write(self.CMD_SET_STATE + state)
        print("State set.")

    # --------------------------------------------------------
    def get_state(self) -> bytes:
        self.ser.write(self.CMD_GET_STATE)
        data = _read_exact(self.ser, 16, "ciphertext timeout")
        self.ser.reset_input_buffer()      # Rest garantiert leer
        return data


    # --------------------------------------------------------
    def encrypt(self) -> None:
        self.ser.write(self.CMD_ENCRYPT)
        print("Encryption started.")

    # --------------------------------------------------------
    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()
            print("Connection to Arduino closed.")
