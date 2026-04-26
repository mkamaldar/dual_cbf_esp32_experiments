import torch
import torch.nn as nn
from dual_cbf_compiler import parse_pytorch, emit_cpp_header

# Reconstruct the specific Softplus architecture for the Pendulum
model = nn.Sequential(
    nn.Linear(2, 32), nn.Softplus(),
    nn.Linear(32, 32), nn.Softplus(),
    nn.Linear(32, 1)
)

# Load the trained weights from the .pt file
pt_path = "pendulum_2_32_32_1/cbf.pt"
try:
    model.load_state_dict(torch.load(pt_path))
except RuntimeError:
    # Fallback in case 01_train_models.py saved the entire model object
    model = torch.load(pt_path)

print("Parsing PyTorch model directly...")
network = parse_pytorch(model)
header = emit_cpp_header(network, relative_degree=2)

with open("pendulum_2_32_32_1/main/dual_cbf.h", "w") as f:
    f.write(header)

print("SUCCESS: pendulum_2_32_32_1/main/dual_cbf.h generated via direct PyTorch ingestion!")