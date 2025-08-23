
"""
Interactive UART tester for the ATmega AES demo (binary protocol).

Keys:
0  → 0x30 (CMD_SET_STATE)
1  → 0x31 (CMD_GET_STATE)
2  → 0x32 (CMD_ENCRYPT)
p  → 0x30 + 16-byte PLAINTEXT (hard-coded)
anything like '0xAA 23' → raw bytes

Empty line or Ctrl-C exits.
"""

import serial, textwrap, threading, sys, time

PORT = "COM7"      # change to your port
BAUD = 115200      # or 57600 etc.

PLAINTEXT = bytes.fromhex(  # 16 bytes, change if you like
    "11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00"
)

def hexline(data: bytes) -> str:
    return " ".join(textwrap.wrap(data.hex().upper(), 2))

def reader_loop(ser: serial.Serial):
    """Continuously read and dump received bytes (hex)."""
    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                ts = time.time()
                print(f"\n{ts:.3f}s  ({len(data)} B)  {hexline(data)}")
    except serial.SerialException:
        pass

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1, dsrdtr=False, rtscts=False)
    except serial.SerialException as e:
        print("Port error:", e); sys.exit(1)

    print(f"[{PORT} @ {BAUD} Bd] – keys 0/1/2/p, empty = quit.")
    ser.reset_input_buffer()

    threading.Thread(target=reader_loop, args=(ser,), daemon=True).start()

    try:
        while True:
            line = input("> ").strip().lower()
            if not line:
                break

            if line == "0":
                ser.write(b"\x30")
                print("  CMD_SET_STATE (command only)")
            elif line == "1":
                ser.write(b"\x31")
                print("  CMD_GET_STATE")
            elif line == "2":
                ser.write(b"\x32")
                print("  CMD_ENCRYPT")
            elif line == "p":
                ser.write(b"\x30" + PLAINTEXT)
                print("  plaintext sent:", hexline(PLAINTEXT))
            else:
                # send  byte list
                for token in line.split():
                    try:
                        val = int(token, 0) & 0xFF
                        ser.write(bytes([val]))
                        print(f"  byte 0x{val:02X} sent")
                    except ValueError:
                        print(f"  ignored: {token}")

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nPort closed — bye.")

if __name__ == "__main__":
    main()
