import numpy as np
import pandas as pd

def decode_circuit_output(filename="ltspice_output.csv", scale=0.25):
    # Expecting: single column called 'voltage'
    df = pd.read_csv(filename)
    measured = df["voltage"].values

    # Reverse the scale factor
    reconstructed_weights = measured / scale

    # Identify which digit was activated
    # Simplest approach: pick the highest-weight channel
    predicted_digit = np.argmax(reconstructed_weights)

    print("Reconstructed weights:", reconstructed_weights)
    print("Predicted digit:", predicted_digit)

    return predicted_digit

if __name__ == "__main__":
    decode_circuit_output()
