#!/usr/bin/env python3
"""
Interaktiver UART‑Tester für den ATmega‑AES‑Demo.
 Tasten:
   0  → 0x30  (CMD_SET_STATE)
   1  → 0x31  (CMD_GET_STATE)
   2  → 0x32  (CMD_ENCRYPT)
   p  → 0x30 + 16‑Byte‑PLAINTEXT   (hart codiert)
   sonst Zahl(en) wie ‚0xAA 23‘   → direkte Byte‑Ausgabe
Leere Eingabe oder Ctrl‑C beendet das Programm.
"""

import serial, textwrap, threading, sys, time

PORT = "COM7"     # anpassen
BAUD = 115200     # oder 4800 etc.

PLAINTEXT = bytes.fromhex(  # 16 Byte hart codiert
    "11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00"
)

# ---------- Hilfsfunktionen -----------------------------------------
def hexline(data: bytes) -> str:
    return " ".join(textwrap.wrap(data.hex().upper(), 2))

def reader_loop(ser: serial.Serial):
    """Liest fortlaufend und zeigt empfangene Bytes."""
    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                ts = time.time()
                print(f"\n{ts:.3f}s  ({len(data)} B)  {hexline(data)}")
    except serial.SerialException:
        pass

# ---------- Hauptprogramm -------------------------------------------
def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1,
                            dsrdtr=False, rtscts=False)
    except serial.SerialException as e:
        print("Port‑Fehler:", e); sys.exit(1)

    print(f"[{PORT} @ {BAUD} Baud] – Tasten 0/1/2/p, leer = Ende.")
    ser.reset_input_buffer()

    threading.Thread(target=reader_loop, args=(ser,), daemon=True).start()

    try:
        while True:
            line = input("> ").strip().lower()
            if not line:
                break

            if line == "0":
                ser.write(b"\x30")
                print("  CMD_SET_STATE (nur Kommando)")
            elif line == "1":
                ser.write(b"\x31")
                print("  CMD_GET_STATE")
            elif line == "2":
                ser.write(b"\x32")
                print("  CMD_ENCRYPT")
            elif line == "p":
                ser.write(b"\x30" + PLAINTEXT)
                print("  PLAINTEXT gesendet:", hexline(PLAINTEXT))
            else:
                # beliebige Byte‑Liste wie "0xAA 23 5" senden
                for token in line.split():
                    try:
                        val = int(token, 0) & 0xFF
                        ser.write(bytes([val]))
                        print(f"  byte 0x{val:02X} gesendet")
                    except ValueError:
                        print(f"  ignoriert: {token}")

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nPort geschlossen – Programm Ende.")

if __name__ == "__main__":
    main()
