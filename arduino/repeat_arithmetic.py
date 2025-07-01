import time
import numpy as np
from encipher import EncipherAPI

def repeat_arithmetic(num_rounds=10000):
    """
    Repeatedly send random plaintext to the Arduino, trigger encryption, 
    and retrieve the resulting state without recording any data.
    
    :param num_rounds: Number of iterations to repeat the process.
    """
    rng = np.random.default_rng()  # Random number generator
    encipher = EncipherAPI(port="COM5")  # Update port as needed

    # Record the overall start time
    overall_start = time.time()

    try:
        for round_number in range(1, num_rounds + 1):
            round_start = time.time()  # Time at the start of each round

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

            round_end = time.time()  # Time at the end of each round
            elapsed_round = round_end - round_start
            print(f"Round {round_number} took {elapsed_round:.4f} seconds.")

    finally:
        encipher.close()

        overall_end = time.time()
        elapsed_total = overall_end - overall_start
        print(f"\nFinished repeating arithmetic. Total time: {elapsed_total:.4f} seconds.")

if __name__ == "__main__":
    # Configuration
    NUM_ROUNDS = 10000  # Number of repetitions

    # Start the process
    repeat_arithmetic(num_rounds=NUM_ROUNDS)
