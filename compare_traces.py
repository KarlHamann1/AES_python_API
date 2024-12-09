import numpy as np
import matplotlib.pyplot as plt

def compare_traces(encrypted_path, no_encryption_path):
    """
    Compare traces from cryptographic and non-cryptographic operations.

    :param encrypted_path: Path to the .npy file with encrypted traces.
    :param no_encryption_path: Path to the .npy file with non-encrypted traces.
    """
    # Load traces
    encrypted_data = np.load(encrypted_path, allow_pickle=True).item()
    no_encryption_data = np.load(no_encryption_path, allow_pickle=True).item()

    encrypted_traces = encrypted_data['data']
    no_encryption_traces = no_encryption_data['data']

    # Calculate the average traces
    avg_encrypted = np.mean(encrypted_traces, axis=0)
    avg_no_encryption = np.mean(no_encryption_traces, axis=0)

    # Calculate the difference
    difference = avg_encrypted - avg_no_encryption

    # Plot the average traces
    plt.figure(figsize=(12, 6))
    plt.plot(avg_encrypted, label="With Encryption", alpha=0.8)
    plt.plot(avg_no_encryption, label="Without Encryption", alpha=0.8)
    plt.plot(difference, label="Difference", linestyle='--', alpha=0.8)
    plt.title("Comparison of Traces (With and Without Encryption)")
    plt.xlabel("Sample Points")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print statistics
    print("=== Statistics ===")
    print(f"Mean of encrypted trace: {np.mean(avg_encrypted):.5f}")
    print(f"Mean of non-encrypted trace: {np.mean(avg_no_encryption):.5f}")
    print(f"Variance of difference: {np.var(difference):.5f}")
    print(f"Max difference: {np.max(difference):.5f}")
    print(f"Min difference: {np.min(difference):.5f}")


if __name__ == "__main__":
    # Paths to encrypted and non-encrypted data
    ENCRYPTED_PATH = "data/data_round_1_20231122_143045.npy"  # Example
    NO_ENCRYPTION_PATH = "data_no_encryption/data_round_1_20231122_143045.npy"  # Example

    compare_traces(ENCRYPTED_PATH, NO_ENCRYPTION_PATH)
