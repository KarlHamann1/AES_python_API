import time
import threading
import secrets  # für Zufalls-Plaintexts, wenn gewünscht
import csv
import os
import sys
import numpy as np

from pi_bare_metal_api import PiBareMetalAPI

# Pfadkonfiguration für den picoscope_segmented_acquisition-Import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from picoscope_segmented_acquisition import SegmentedPicoScopeAcquisition

"""
def continuous_read(pi_api):
    while pi_api.ser.is_open:
        try:
            line = pi_api.ser.readline()
            if line:
                print("PI>", line.decode(errors='ignore').rstrip())
        except Exception as e:
            print("Error reading from Pi:", e)
            break
"""
def main():
    # 1) Verbindung zum Pi (UART)
    pi_api = PiBareMetalAPI(port="COM7", baudrate=115200)
    
    # Hintergrund-Thread, damit Pi-Ausgaben (UART) sichtbar sind
    #reader_thread = threading.Thread(target=continuous_read, args=(pi_api,), daemon=True)
    #reader_thread.start()
    time.sleep(0.075)  # kleine Pause für UART-Setup

    # 2) Picoscope-Instrument erstellen und konfigurieren
    seg_acq = SegmentedPicoScopeAcquisition(
        device_resolution_bits=12,
        capture_channel="A",
        trigger_channel="B",
        trigger_threshold_mV=100,
        coupling_mode="AC",
        voltage_range="20MV",
        capture_duration_s=0.00023,   # z.B. 230 us
        sampling_rate_hz=62.5e6,      # ~16 ns/sample
        timebase=4,                   # Beispielhafter Timebase-Wert für 12-Bit
        output_dir="data_pi_30dB_single_encryption",   # Neuer Ordner
        filename_prefix="aes_trace"   # Kürzerer Prefix
    )

    try:
        # 3) PicoScope öffnen und Grundkonfiguration
        seg_acq.open_unit_and_setup()

        # Wir erstellen EINE CSV für die gesamte Messreihe
        csv_filename = "trace_overview.csv"
        csv_path = os.path.join(seg_acq.output_dir, csv_filename)
        os.makedirs(seg_acq.output_dir, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # Kopfzeile in der CSV
            csv_writer.writerow(["TraceIndex", "PlaintextHex", "CiphertextHex", "TraceFilePath"])

            # 4) Anzahl der gewünschten Traces
            num_traces = 10000
            for trace_idx in range(num_traces):
                # a) Plaintext wählen/generieren (16 Byte)
                #    - entweder zufällig oder fest
                plaintext_bytes = secrets.token_bytes(16)  # zufällig
                plaintext_hex = "".join(f"{b:02X}" for b in plaintext_bytes)

                # b) Plaintext an den Pi senden
                pi_api.set_plaintext(plaintext_hex)

                # c) Auf 1 Segment konfigurieren (1 Trigger pro Verschlüsselung)
                seg_acq.setup_segmented_mode(num_segments=1)

                # d) Scope „armen“ => wartet auf 1 Trigger
                success = seg_acq.run_segmented_capture()
                if not success:
                    print(f"[!] Timeout beim Warten auf Trigger in Iteration {trace_idx}.")
                    seg_acq.stop()
                    continue

                # e) AES 1x auf dem Pi laufen lassen => Pin-Trigger => Scope nimmt 1 Segment auf
                pi_api.run_aes(1)

                # f) Segment 0 aus dem Picoscope auslesen
                waveform = seg_acq.retrieve_segment(segment_index=0)

                # g) Waveform speichern (ohne Timestamp)
                #    Wir ändern hier save_segment_data, damit Dateiname kürzer wird:
                trace_filename = f"aes_trace_round_{trace_idx}.npy"
                full_path = os.path.join(seg_acq.output_dir, trace_filename)
                # numpy-Array speichern
                np.save(full_path, waveform)
                print(f"Trace {trace_idx} gespeichert als {trace_filename}.")

                # h) Pi hat nach aes_encrypt den Ciphertext per UART gesendet => auslesen/parsen
                time.sleep(0.075)  # kurze Pause, damit Pi Zeit zum Senden hat
                lines = pi_api.read_lines(max_lines=20)
                ciphertext_hex = None
                for line in lines:
                    if "Final Ciphertext: " in line:
                        # z. B. "Final Ciphertext: AABBCCDDEEFF..."
                        parts = line.split(": ")
                        if len(parts) == 2:
                            ciphertext_hex = parts[1].strip()
                            break

                # i) In CSV schreiben
                csv_writer.writerow([
                    trace_idx,
                    plaintext_hex,
                    ciphertext_hex if ciphertext_hex else "UNKNOWN",
                    trace_filename
                ])

                # j) Aufnahme stoppen, bevor nächster Trace
                seg_acq.stop()

            print(f"\nAlle {num_traces} Traces wurden aufgezeichnet und in {csv_path} gelistet.")

    finally:
        # Cleanup: Picoscope schließen, Pi schließen
        seg_acq.close()
        pi_api.close()

if __name__ == "__main__":
    main()
