#!/usr/bin/env python3
"""
Trigger‑Loop für den ATmega328P (AES), robust gegen UART‑Verluste.
"""

import time, signal, secrets, numpy as np
from encipher import EncipherAPI           # gepatchte API mit Hex‑Print

# ---- Konfiguration -------------------------------------------------
PORT          = "COM7"
BAUD          = 57_600         # 57 600 Baud
ROUNDS        = 10_000
IDLE_GAP_S    = 0.0          # 0.1 s für sichtbare Pausen
FIXED_PT      = None         # bytes.fromhex("00112233445566778899AABBCCDDEEFF")
REPORT_EVERY  = 1000
POST_ENC_WAIT = 0.050        # 5 ms, damit der AVR Zeit hat, die TX‑FIFO zu füllen

# ---- Hauptschleife -------------------------------------------------
def repeat_arithmetic():
    rng = np.random.default_rng()
    enc = EncipherAPI(port=PORT, baudrate=BAUD, timeout=1)

    t0          = time.time()
    last_report = t0
    done        = 0

    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    try:
        for _ in range(ROUNDS):
            pt = FIXED_PT or rng.integers(0, 256, 16, np.uint8).tobytes()

            enc.set_state(pt)
            enc.encrypt()

            time.sleep(POST_ENC_WAIT)   # gibt dem AVR Zeit, TX‑FIFO zu füllen

            try:
                ct = enc.get_state()    # liest 16 Bytes, wirft TimeoutError falls <16
            except TimeoutError:
                # lese alles, was evtl. doch angekommen ist und zeige Warnung
                leftover = enc.ser.read_all()
                print(f"[Warn] only {len(leftover)} / 16 bytes received")
                # trotzdem weitermachen
            done += 1

            if IDLE_GAP_S:
                time.sleep(IDLE_GAP_S)

            if done % REPORT_EVERY == 0:
                now = time.time()
                print(f"[{done}/{ROUNDS}] {(done/(now-t0)):.1f} encrypt/s")

    except KeyboardInterrupt:
        print(f"\nInterrupted after {done} rounds.")
    finally:
        enc.close()
        elapsed = time.time() - t0
        print(f"Total {elapsed:.1f}s, average {(done/elapsed):.1f} encrypt/s.")

# ---- Start ---------------------------------------------------------
if __name__ == "__main__":
    repeat_arithmetic()
