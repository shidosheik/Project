import numpy as np
import pandas as pd

def decode_circuit_output(filename="ltspice_output.csv", scale=0.25):
    # LTspice export: tab/whitespace-separated
    df = pd.read_csv(filename, delim_whitespace=True)

    # 🔧 Drop any rows that are completely empty / NaN (LTspice often has one at the end)
    df = df.dropna(how="all")

    # Keep only voltage columns like V(d), V(nc0), V(y), etc.
    voltage_cols = [c for c in df.columns if c.startswith("V(")]

    if not voltage_cols:
        raise ValueError(
            f"No LTspice voltage columns found. Columns I see are: {list(df.columns)}"
        )

    # ✅ Take the last *valid* sample now
    last_row = df.iloc[-1][voltage_cols]
    measured = last_row.values.astype(float)

    reconstructed_weights = measured / scale
    predicted_digit = int(np.argmax(reconstructed_weights))

    print("Voltage columns used:", voltage_cols)
    print("Measured voltages:", measured)
    print("Reconstructed weights:", reconstructed_weights)
    print("Predicted digit:", predicted_digit)

    return predicted_digit

if __name__ == "__main__":
    decode_circuit_output()
