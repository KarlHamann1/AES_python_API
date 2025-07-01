import time
import threading
import secrets  # Für Zufalls-Plaintexts
import csv
import os
import sys
import numpy as np

from pi_bare_metal_api import PiBareMetalAPI

# Pfadkonfiguration: Wir nehmen an, dass picoscope_acquisition.py im parent_dir liegt.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from picoscope_acquisition import DataAcquisition

def main():
    """
    Führt eine Block-Mode-Akquisition für AES-Seitenkanal-Messungen durch:
    - Schickt einen Plaintext an den Pi.
    - Rüstet das Picoscope im Block Mode.
    - Startet 'run_aes(1)' auf dem Pi (ein Trigger).
    - Liest Wellenform via Block Mode aus und speichert sie als .npy
    - Liest den Ciphertext vom Pi und schreibt ihn zusammen mit dem Plaintext in eine CSV.
    """

    # 1) Verbindung zum Pi (UART)
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)

    # (Optional) Hintergrund-Thread für Live-UART-Ausgaben:
    # def continuous_read(pi):
    #     ...
    # reader_thread = threading.Thread(target=continuous_read, args=(pi_api,), daemon=True)
    # reader_thread.start()

    # Warte kurz, damit Pi bereit ist
    time.sleep(0.2)

    # 2) Erstelle das DataAcquisition-Objekt für **Block Mode**
    data_acq = DataAcquisition(
        device_resolution_bits=12,
        timebase=4,                 # ~16 ns Sampleintervall bei 12 Bit
        sampling_rate_hz=62.5e6,    # ~16 ns/Abtastung
        capture_duration_s=0.00023, # 230 us
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        trigger_delay_samples=0,
        auto_trigger_ms=1000,
        coupling_mode="AC",
        voltage_range="20MV",
        output_dir="data_pi_40dB_block_encryption",  # Neuer Ordner für Block-Mode-Daten
        filename_prefix="aes_trace_block"
    )

    try:
        # 3) Picoscope einrichten
        data_acq.setup_picoscope()

        # 4) CSV-Datei für die gesamte Messreihe
        csv_filename = "trace_overview_block.csv"
        csv_path = os.path.join(data_acq.output_dir, csv_filename)
        os.makedirs(data_acq.output_dir, exist_ok=True)  # Ordner anlegen, falls nötig

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # Kopfzeile
            csv_writer.writerow(["TraceIndex", "PlaintextHex", "CiphertextHex", "TraceFilePath"])

            # 5) Anzahl Traces (z. B. 10.000)
            num_traces = 10
            for trace_idx in range(num_traces):
                # a) Zufalls-Plaintext erzeugen
                plaintext_bytes = secrets.token_bytes(16)
                plaintext_hex = "".join(f"{b:02X}" for b in plaintext_bytes)

                # b) Plaintext an Pi
                pi_api.set_plaintext(plaintext_hex)

                # c) Scope-Blockaufnahme vorbereiten
                #    Bei jedem Durchlauf neu, damit wir pro AES-Lauf eine Blockmessung haben.
                data_acq.prepare_block_mode()

                # d) Pi führt AES durch => Trigger für das Picoscope
                pi_api.run_aes(1)

                # e) Block-Capture abholen (blocking bis getValues fertig)
                waveform = data_acq.capture_block()

                # f) Speichern der Wellenform in .npy
                trace_filename = f"aes_trace_block_{trace_idx}.npy"
                full_path = os.path.join(data_acq.output_dir, trace_filename)
                np.save(full_path, waveform)
                print(f"[Trace {trace_idx}] Wellenform gespeichert als {trace_filename}")

                # g) Ciphertext vom Pi über UART lesen
                time.sleep(0.05)  # Kurz warten, damit Pi Zeit zum Senden hat
                lines = pi_api.read_lines(max_lines=20)
                ciphertext_hex = None
                for line in lines:
                    if "Final Ciphertext: " in line:
                        parts = line.split(": ")
                        if len(parts) == 2:
                            ciphertext_hex = parts[1].strip()
                            break

                # h) In CSV schreiben
                csv_writer.writerow([
                    trace_idx,
                    plaintext_hex,
                    ciphertext_hex if ciphertext_hex else "UNKNOWN",
                    trace_filename
                ])

        print(f"\nAlle {num_traces} Block-Mode-Traces wurden aufgezeichnet. CSV in {csv_path}.")

    finally:
        data_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
