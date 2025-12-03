import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd


# -------------------------
# 1. SIMPLE 16-WEIGHT MODEL
# -------------------------
class Tiny16Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Single fully-connected layer: 784 -> 16
        self.fc1 = nn.Linear(28 * 28, 16)

    def forward(self, x):
        # Flatten input image and apply linear layer
        x = x.view(-1, 28 * 28)
        return self.fc1(x)


# -------------------------
# 2. TRAIN MODEL ON MNIST
# -------------------------
def train_model(model_path="model.pt"):
    """
    Train Tiny16Net on MNIST (very simple loss) OR load an existing model
    from model_path if it already exists and is non-empty.
    """
    model = Tiny16Net()

    # If a valid model file exists, load it and skip training
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded existing model from {model_path}")
        return model

    # Otherwise, train a fresh model
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # We define a simple loss that just encourages stable outputs
    model.train()
    for epoch in range(2):
        for images, _ in loader:
            outputs = model(images)

            # Loss encourages the mean output to hover near 0
            loss = ((outputs.mean(dim=1) - 0) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {epoch + 1}/2, Loss={loss.item():.4f}")

    # Save trained weights so we can reuse them next run
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

    return model


# -------------------------
# 3. EXTRACT WEIGHTS
# -------------------------
def extract_16_weights(model):
    """
    Extract 16 real-valued weights from the trained model.
    Here we take the mean across inputs for each of the 16 outputs.
    Result: 16-element NumPy array.
    """
    weights = model.fc1.weight.data.mean(dim=1).numpy()
    return weights


# -------------------------
# 4. CONVERT TO LTSPICE VOLTAGES
# -------------------------
def convert_weights_to_voltages(weights, scale=0.12):
    """
    Map integer weights w_i to programming voltages.
    For example: scale=1.8/15 so |w_i| = 15 -> 1.8 V.
    """
    return weights * scale


# -------------------------
# 5. WRITE VOLTAGES TO CSV
# -------------------------
def write_voltage_file(voltages, filename="voltages_for_ltspice.csv"):
    """
    Write one or more voltages to a CSV file with a single column 'voltage'.
    In the current project we collapse to ONE voltage (single cell).
    """
    df = pd.DataFrame({"voltage": voltages})
    df.to_csv(filename, index=False)
    print(f"Saved voltages to {filename}")


# -------------------------
# 6. NORMALIZATION / QUANTIZATION
# -------------------------
def normalize(weights, min_norm, max_norm):
    """
    Normalize weights into [min_norm, max_norm].
    (Not used in the current collapsed-flow, but kept for reference.)
    """
    norm_weights = []
    for weight in weights:
        norm_weights += [
            (weight - min(weights)) / (max(weights) - min(weights))
            * (max_norm - min_norm) + min_norm
        ]
    return norm_weights


def quantize_simple_rounding(weights, beta=15):
    """
    Normalize weights based on the maximum absolute value, then map into
    integer levels in [-beta, +beta].

    This prevents all weights from collapsing to zero when their magnitudes
    are very small.
    """
    # Ensure we are working with a NumPy array of floats
    w = np.array(weights, dtype=float)

    # Find the largest absolute value
    max_abs = float(np.max(np.abs(w)))
    if max_abs == 0.0:
        # All weights are exactly zero -> return 0 levels
        return np.zeros_like(w, dtype=int)

    w_quant = []
    for v in w:
        # Normalize into [-1, 1]
        norm = v / max_abs

        # Scale normalized value into [-beta, +beta] and round
        n = int(np.round(norm * beta))

        # Clip to ensure we stay in range
        n = max(-beta, min(beta, n))

        w_quant.append(n)

    return np.array(w_quant, dtype=int)


# -------------------------
# MAIN SCRIPT
# -------------------------
if __name__ == "__main__":
    # 1) Train or load model
    model = train_model()

    # 2) Get 16 real-valued weights from the model
    v = extract_16_weights(model)                 # shape: (16,)

    # 3) Quantize to integer levels [-15..+15]
    w_quant = quantize_simple_rounding(v)         # shape: (16,)

    # 4) Collapse the 16 levels into ONE effective level
    #    Option A: just use the first level
    effective_level = int(w_quant[0])

    #    (If you prefer averaging, replace the line above with:
    #     effective_level = int(np.round(w_quant.mean()))
    #    )

    # 5) Convert that single level to a single programming voltage
    single_voltage = convert_weights_to_voltages(
        np.array([effective_level]),  # make it a 1-element array
        scale=0.12
    )

    # 6) Write ONE voltage to CSV for LTspice (to drive Vw0)
    write_voltage_file(single_voltage)
