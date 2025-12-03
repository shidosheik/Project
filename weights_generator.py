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
        self.fc1 = nn.Linear(28*28, 16)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc1(x)

# -------------------------
# 2. TRAIN MODEL ON MNIST
# -------------------------
def train_model():
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Tiny16Net()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # We need fake labels because we aren’t doing classification here.
    # Instead, we train the model to “produce stable weights”
    dummy_labels = torch.zeros(64, dtype=torch.long)

    for epoch in range(2):
        for images, _ in loader:
            outputs = model(images)

            # Loss encourages outputs to stabilize
            loss = ((outputs.mean(dim=1) - 0)**2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1}/2, Loss={loss.item():.4f}")

    return model

# -------------------------
# 3. EXTRACT WEIGHTS
# -------------------------
def extract_16_weights(model):
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
    df = pd.DataFrame({"voltage": voltages})
    df.to_csv(filename, index=False)
    print(f"Saved voltages to {filename}")

# Normalize weights between a minimum and maximum range
# useage: weights - input to be normalize
# min_norm - minimum value of the normalized range
# max_norm - maximum value of the notmalized range
# this is dissimilar to the method propsed in the paper
def normalize(weights, min_norm, max_norm):
    norm_weights = []
    for weight in weights:
        norm_weights += [(weight - min(weights)) / (max(weights) - min(weights)) * (max_norm - min_norm) + min_norm]
    return norm_weights

def quantize_simple_rounding(weights, alpha=80.0, beta=15):
    """
    Oshio-style simple rounding:
    v_i (real) -> w_i in [-beta, ..., +beta] with 4-bit magnitude.

    alpha and beta correspond to Eq. (2) in Oshio et al.
    """
    w_quant = []
    for v in weights:
        s = 1 if v >= 0 else -1
        mag = abs(v)

        # scale
        scaled = alpha * mag

        # round and clip
        n = int(np.round(scaled))
        n = max(0, min(beta, n))

        w_quant.append(s * n)

    return np.array(w_quant, dtype=int)

# -------------------------
# MAIN SCRIPT
# -------------------------
if __name__ == "__main__":
    model = train_model()
    v = extract_16_weights(model)                 # real valued
    w_quant = quantize_simple_rounding(v)         # integers -15..+15
    voltages = convert_weights_to_voltages(w_quant, scale=0.12)  # see below
    write_voltage_file(voltages)

