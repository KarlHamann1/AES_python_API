import numpy as np
from encipher import EncipherAPI

def repeat_arithmetic(num_rounds=1000):
    """
    Repeatedly send random plaintext to the Arduino, trigger encryption, 
    and retrieve the resulting state without recording any data.
    
    :param num_rounds: Number of iterations to repeat the process.
    """
    rng = np.random.default_rng()  # Random number generator
    encipher = EncipherAPI(port="COM5")  # Update port as needed

    try:
        for round_number in range(1, num_rounds + 1):
            print(f"\n=== Round {round_number}/{num_rounds} ===")

            # Generate random 16-byte plaintext
            plaintext = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
            print(f"Generated plaintext: {plaintext.hex()}")

            # Set the state on the Arduino
            encipher.set_state(plaintext)

            # Trigger encryption
            encipher.encrypt()

            # Optionally retrieve the ciphertext
            ciphertext = encipher.get_state()
            print(f"Retrieved ciphertext: {ciphertext.hex()}")

    finally:
        encipher.close()
        print("\nFinished repeating arithmetic.")

if __name__ == "__main__":
    # Configuration
    NUM_ROUNDS = 10000  # Number of repetitions

    # Start the process
    repeat_arithmetic(num_rounds=NUM_ROUNDS)
