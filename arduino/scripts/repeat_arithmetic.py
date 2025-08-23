#!/usr/bin/env python3
"""
Trigger loop for the ATmega328P (AES).
"""

import time, signal
import numpy as np

from pathlib import Path
import sys
REPO_ROOT   = Path(__file__).resolve().parents[2] 
ARDUINO_DIR = REPO_ROOT / "arduino"
sys.path.append(str(ARDUINO_DIR / "scripts"))       # ← make local helpers importable

from encipher import EncipherAPI   # tiny binary-protocol client

#  Config 
PORT          = "COM7"
BAUD          = 115200             # 115200 Bd
ROUNDS        = 10_00000
IDLE_GAP_S    = 0.0                # add a pause between ops if you want
FIXED_PT      = None               # or e.g. bytes.fromhex("00112233445566778899AABBCCDDEEFF")
REPORT_EVERY  = 1000
POST_ENC_WAIT = 0.00050              # give AVR time to TX; 0.00050 s = 0.5 ms

def repeat_arithmetic():
    rng = np.random.default_rng()
    enc = EncipherAPI(port=PORT, baudrate=BAUD, timeout=1)

    t0 = time.time()
    done = 0

    # Allow Ctrl+C to break out cleanly
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    try:
        for _ in range(ROUNDS):
            # fixed PT if provided, else fresh 16 random bytes
            pt = FIXED_PT or rng.integers(0, 256, 16, np.uint8).tobytes()

            enc.set_state(pt)
            enc.encrypt()

            # small wait so device can push bytes into its UART buffer
            time.sleep(POST_ENC_WAIT)

            try:
                # will raise TimeoutError if we didn't get 16 bytes in time
                ct = enc.get_state()
            except TimeoutError:
                # read whatever did arrive (if anything) and warn; keep going
                leftover = enc.ser.read_all()
                print(f"[Warn] only {len(leftover)} / 16 bytes received")

            done += 1

            if IDLE_GAP_S:
                time.sleep(IDLE_GAP_S)

            if done % REPORT_EVERY == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                print(f"[{done}/{ROUNDS}] {rate:.1f} encrypt/s")

    except KeyboardInterrupt:
        print(f"\nInterrupted after {done} rounds.")
    finally:
        enc.close()
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0.0
        print(f"Total {elapsed:.1f}s, average {rate:.1f} encrypt/s.")

#  Entry point 
if __name__ == "__main__":
    repeat_arithmetic()
