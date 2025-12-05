import numpy as np
import pandas as pd

# -----------------------------
# 1. LOAD CALIBRATION DATA
# -----------------------------
def load_calibration(csv_path: str) -> pd.DataFrame:
    """
    Load the LTspice weight sweep table exported from Excel.
    Expected columns (case-insensitive):
      - Vw0, Vw1, Vw2, Vw3 (0 or 1.8)
      - Effective C (fF)  or something similar
      - t_inv             (in seconds; you can keep it as LTspice prints)
      - weight_code       (0..15).
    """
    df = pd.read_csv(csv_path)

    # Normalize column names a bit
    df.columns = [c.strip() for c in df.columns]

    # If there's no weight_code column, derive it from Vw bits
    if "weight_code" not in df.columns:
        required = ["Vw0", "Vw1", "Vw2", "Vw3"]
        for r in required:
            if r not in df.columns:
                raise ValueError(
                    f"Missing column '{r}' in CSV; have columns: {df.columns}"
                )

        def row_to_code(row):
            code = 0
            if row["Vw0"] > 0.9:
                code |= 1  # bit 0
            if row["Vw1"] > 0.9:
                code |= 2  # bit 1
            if row["Vw2"] > 0.9:
                code |= 4  # bit 2
            if row["Vw3"] > 0.9:
                code |= 8  # bit 3
            return code

        df["weight_code"] = df.apply(row_to_code, axis=1)

    return df


# -----------------------------
# 2. BUILD DELAY → WEIGHT LUT
# -----------------------------
def build_delay_lut(df: pd.DataFrame):
    """
    From the calibration data, build a mapping:
      weight_code -> mean t_inv
    Returns:
      codes  : np.ndarray of shape [N_codes]
      delays : np.ndarray of shape [N_codes] (seconds)
    """

    if "weight_code" not in df.columns:
        raise ValueError("DataFrame must have 'weight_code' column.")

    # Select the t_inv (s) column
    if "t_inv (s)" in df.columns:
        df = df.rename(columns={"t_inv (s)": "t_inv"})
    elif "t_inv" in df.columns:
        pass
    else:
        raise ValueError(f"No usable t_inv column found in: {df.columns}")

    # Drop any rows without a valid measured delay
    df = df.dropna(subset=["t_inv"])

    grouped = df.groupby("weight_code")["t_inv"].mean().sort_index()

    codes = grouped.index.to_numpy()
    delays = grouped.values

    return codes, delays


def decode_delay(t_meas: float, codes: np.ndarray, delays: np.ndarray) -> int:
    """
    Given a measured delay t_meas (in seconds), return the closest weight_code.
    """
    idx = int(np.argmin(np.abs(delays - t_meas)))
    return int(codes[idx])


# -----------------------------
# 3. EXAMPLE USAGE / CLI
# -----------------------------
def example_usage():
    # 1) Load calibration table
    df = load_calibration("ltspice_weight_sweep.csv")

    # 2) Build LUT
    codes, delays = build_delay_lut(df)

    print("Calibrated codes:", codes)
    print("Mean delays (s):", delays)

    # 3) print effective capacitance per code
    cap_by_code = (
        df.sort_values("weight_code")
          .drop_duplicates("weight_code")
          .set_index("weight_code")
    )
    if "Effective C (fF)" in cap_by_code.columns:
        print("Effective capacitance per code (fF):")
        for c in codes:
            print(f"  code {c:2d}: {cap_by_code.loc[c, 'Effective C (fF)']}")
    else:
        print("No 'Effective C (fF)' column found; skipping C lookup.")


if __name__ == "__main__":
    example_usage()
