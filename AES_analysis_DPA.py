import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from Crypto.Cipher import AES

# --- AES S-box ---
SBOX = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

# --- AES Rcon table (first element 0x8D is mostly used for advanced generation, 
#     but we simply index from 1 to 10 for AES-128) ---
RCON = [0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

def forward_key_schedule(key, n_rounds=10):
    """
    key: 16 bytes for AES-128
    n_rounds: usually 10 for AES-128
    Returns a list of 4*(n_rounds+1)*4 bytes = 176 bytes (the expanded key)
    """
    round_keys = list(key)  # initial 16 bytes
    # We need 4*(n_rounds+1) 'words'. Each word is 4 bytes.
    # For AES-128, that means 44 words total → 176 bytes.

    for i in range(4, 4*(n_rounds+1)):
        # Extract the previous word
        t0, t1, t2, t3 = round_keys[(i-1)*4 : i*4]

        # Every 4th word, we rotate, S-box, and apply Rcon
        if i % 4 == 0:
            # Rotate left by 1 byte
            t0, t1, t2, t3 = SBOX[t1], SBOX[t2], SBOX[t3], SBOX[t0]
            # Rcon for round i//4
            # e.g. for i=4 => i//4=1 => RCON[1] = 0x01
            rcon_index = i // 4
            t0 ^= RCON[rcon_index]

        # XOR with the word from (i-4)
        s0, s1, s2, s3 = round_keys[(i-4)*4 : (i-3)*4]
        round_keys.extend([t0 ^ s0, t1 ^ s1, t2 ^ s2, t3 ^ s3])

    return round_keys

def verify_key(round_key_1, plaintext, ciphertext):
    """
    Given a candidate 16-byte key (round_key_1),
    check if AES(plaintext) == ciphertext using PyCryptodome.
    """
    if len(round_key_1) != 16:
        return False
    aes128 = AES.new(round_key_1, AES.MODE_ECB)
    cipher_test = aes128.encrypt(plaintext)
    return cipher_test == ciphertext

def main():
    # Path to your data directory
    data_dir = "pi/data_pi_40dB_single_encryption_diff"
    #data_dir = "arduino/data_arduino_40dB_block_encryption"
    csv_file = os.path.join(data_dir, "trace_overview.csv")
    #csv_file = os.path.join(data_dir, "trace_overview_arduino.csv")
    
    plaintexts = []
    ciphertexts = []
    traces = []  # List of power/LED traces (NumPy arrays)
    
    # --- Read the CSV, load each trace ---
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Each row has:
            #   TraceIndex,PlaintextHex,CiphertextHex,TraceFilePath
            pt_hex = row["PlaintextHex"].strip()
            ct_hex = row["CiphertextHex"].strip()
            trace_filename = row["TraceFilePath"].strip()

            pt = bytes.fromhex(pt_hex)
            ct = bytes.fromhex(ct_hex)

            plaintexts.append(pt)
            ciphertexts.append(ct)

            trace_path = os.path.join(data_dir, trace_filename)
            trace = np.load(trace_path)
            traces.append(trace)

            # OPTIONAL choose a subset of traces
            if len(traces) >= 6000:
                break

    num_traces = len(traces)
    if num_traces == 0:
        print("Keine Traces gefunden.")
        return

    samples_per_trace = len(traces[0])
    print(f"Anzahl der Traces: {num_traces}")
    print(f"Samples pro Trace: {samples_per_trace}")

    # --- Perform a (very) simple DPA for each byte ---
    recovered_round_key = None
    mean_delta_accu_visualization = [[] for _ in range(16)]

    # We'll loop over bit positions l = 0..7 in the S-box output
    for l in range(8):
        print(f"Analyse für Bitposition {l}:")
        candidate_key = []

        for byte_index in range(16):
            delta = np.zeros(256)
            # For each possible byte guess (0..255)
            for guess in range(256):
                count0 = 0
                count1 = 0
                accu0 = np.zeros(samples_per_trace)
                accu1 = np.zeros(samples_per_trace)

                for i in range(num_traces):
                    # Hypothetical intermediate = SBOX[ P[i][byte_index] XOR guess ]
                    v = SBOX[ plaintexts[i][byte_index] ^ guess ]
                    target_bit = (v >> l) & 1

                    if target_bit == 0:
                        accu0 += traces[i]
                        count0 += 1
                    else:
                        accu1 += traces[i]
                        count1 += 1

                # If we never got any traces in group0 or group1, skip
                if count0 == 0 or count1 == 0:
                    continue

                mean0 = accu0 / count0
                mean1 = accu1 / count1
                diff = np.abs(mean1 - mean0)

                # we take the max difference as our metric
                delta[guess] = np.max(diff)

                # (Optional) store diff for visualization
                # If you wanted to store or plot them, you could append
                # but this can become huge. We'll just keep one example:
                if guess == 0:
                    mean_delta_accu_visualization[byte_index] = diff

            # Pick the best guess for this byte
            predicted_byte = int(np.argmax(delta))
            candidate_key.append(predicted_byte)
            print(f"Byte {byte_index}: {predicted_byte:02X}", end=" ")
        print("")

        candidate_key = bytes(candidate_key)

        # Check if this candidate key is correct by encrypting one known plaintext
        if verify_key(candidate_key, plaintexts[0], ciphertexts[0]):
            print("Verifizierter Schlüssel gefunden!")
            recovered_round_key = candidate_key
            break
        else:
            print("Schlüsselkandidat passt nicht. Weiter mit anderer Bitposition...")

    if recovered_round_key is None:
        print("Kein gültiger Schlüssel extrahiert.")
    else:
        print("Rekonstruierter Schlüssel:", recovered_round_key.hex().upper())
        # Save the key in a text file
        key_file = os.path.join(data_dir, "recovered_key.txt")
        with open(key_file, "w") as f:
            f.write(recovered_round_key.hex().upper())
        print(f"Schlüssel in {key_file} gespeichert.")

    # (Optional) Example of saving a quick plot for each byte
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for byte_index in range(16):
        plt.figure()
        # Just an example: we plot the diff for guess=0 that we stored above
        if len(mean_delta_accu_visualization[byte_index]) > 0:
            plt.plot(mean_delta_accu_visualization[byte_index], label="Diff (guess=0)")
        plt.title(f"Delta für Schlüsselbyte {byte_index}")
        plt.xlabel("Sample Index")
        plt.ylabel("Delta")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"DPA_byte_{byte_index:02d}.png"))
        plt.close()

if __name__ == "__main__":
    main()
