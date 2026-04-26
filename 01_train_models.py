"""Train neural CBFs for the three paper examples and export them to ONNX.

Run from the repository root:
    pip install torch numpy onnx
    python 01_train_models.py

This produces:
    bicycle_4_32_32_1/cbf.onnx
    bicycle_4_32_32_1/cbf.pt
    vdp_2_64_64_1/cbf.onnx
    vdp_2_64_64_1/cbf.pt
    pendulum_2_32_32_1/cbf.onnx        (softplus, for relative_degree=2)
    pendulum_2_32_32_1/cbf.pt

Notes
-----
The training objective here is to fit the *shape* of a sensible barrier
function so the timing experiment uses representative weights -- it is
*not* tuned for guaranteed safety in a closed loop. For real safety
guarantees, train with a CBF-specific loss as in [Dawson 2023; So 2023].
The timing and memory measurements reported in the paper depend only
on the network topology and are independent of training quality.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

def make_relu_net(widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def make_softplus_net(widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            layers.append(nn.Softplus(beta=1.0))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Target barrier shapes -- analytic functions of the state we want the
# network to imitate. Choosing simple, smooth h(x) keeps training stable.
# ---------------------------------------------------------------------------

def bicycle_target(x: torch.Tensor) -> torch.Tensor:
    """4-state bicycle: state = (px, py, psi, v).

    Safe set: stay within a radius-5 disk in the (px, py) plane and below
    a 25 m/s speed limit. The barrier is the minimum of those two margins
    (with smooth softplus replacing min for differentiability of the
    target).
    """
    px, py, _, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    margin_pos = 25.0 - (px ** 2 + py ** 2)  # >0 inside disk
    margin_v = 25.0 - v                      # >0 below limit
    # smoothed min = -log(exp(-a) + exp(-b))
    return -torch.logsumexp(torch.stack([-margin_pos, -margin_v], dim=-1), dim=-1)


def vdp_target(x: torch.Tensor) -> torch.Tensor:
    """Van der Pol: state = (x1, x2). Safe set = ellipse x1^2/9 + x2^2/4 <= 1."""
    x1, x2 = x[..., 0], x[..., 1]
    return 1.0 - (x1 ** 2 / 9.0 + x2 ** 2 / 4.0)


def pendulum_target(x: torch.Tensor) -> torch.Tensor:
    """Inverted pendulum: state = (theta, theta_dot). Safe = |theta| < pi/4."""
    theta, dtheta = x[..., 0], x[..., 1]
    # Smooth bound around theta with a velocity penalty
    return (math.pi / 4.0) ** 2 - theta ** 2 - 0.05 * dtheta ** 2


# ---------------------------------------------------------------------------
# Generic trainer
# ---------------------------------------------------------------------------

def train_cbf(
    model: nn.Sequential,
    target_fn,
    sampler,
    n_iters: int = 5000,
    batch_size: int = 1024,
    lr: float = 1e-3,
) -> nn.Sequential:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(n_iters):
        x = sampler(batch_size)
        y_target = target_fn(x).unsqueeze(-1)
        y_pred = model(x)
        loss = (y_pred - y_target).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 1000 == 0 or it == n_iters - 1:
            print(f"  iter {it:5d}  loss = {loss.item():.4e}")
    return model


def export_pt_and_onnx(model: nn.Sequential, in_dim: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "cbf.pt"
    onnx_path = out_dir / "cbf.onnx"
    torch.save(model.state_dict(), pt_path)

    # ONNX export: use a single-sample input so dimensions are literal,
    # not symbolic -- the compiler reads concrete (n_in, n_out) shapes.
    sample = torch.zeros(1, in_dim)
    torch.onnx.export(
        model,
        sample,
        onnx_path.as_posix(),
        input_names=["x"],
        output_names=["h"],
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  wrote {pt_path.name} and {onnx_path.name}")


# ---------------------------------------------------------------------------
# Samplers (uniform over each system's relevant state region)
# ---------------------------------------------------------------------------

def bicycle_sampler(n: int) -> torch.Tensor:
    px = torch.empty(n).uniform_(-7, 7)
    py = torch.empty(n).uniform_(-7, 7)
    psi = torch.empty(n).uniform_(-math.pi, math.pi)
    v = torch.empty(n).uniform_(0, 35)
    return torch.stack([px, py, psi, v], dim=-1)


def vdp_sampler(n: int) -> torch.Tensor:
    x1 = torch.empty(n).uniform_(-4, 4)
    x2 = torch.empty(n).uniform_(-3.5, 3.5)
    return torch.stack([x1, x2], dim=-1)


def pendulum_sampler(n: int) -> torch.Tensor:
    theta = torch.empty(n).uniform_(-math.pi / 2, math.pi / 2)
    dtheta = torch.empty(n).uniform_(-3, 3)
    return torch.stack([theta, dtheta], dim=-1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== [1/3] Bicycle 4-32-32-1 (ReLU) ===")
    bicycle = make_relu_net([4, 32, 32, 1])
    train_cbf(bicycle, bicycle_target, bicycle_sampler, n_iters=5000)
    export_pt_and_onnx(bicycle, in_dim=4, out_dir=ROOT / "bicycle_4_32_32_1")

    print("\n=== [2/3] Van der Pol 2-64-64-1 (ReLU) ===")
    vdp = make_relu_net([2, 64, 64, 1])
    train_cbf(vdp, vdp_target, vdp_sampler, n_iters=5000)
    export_pt_and_onnx(vdp, in_dim=2, out_dir=ROOT / "vdp_2_64_64_1")

    print("\n=== [3/3] Pendulum 2-32-32-1 (Softplus, for relative_degree=2) ===")
    pend = make_softplus_net([2, 32, 32, 1])
    train_cbf(pend, pendulum_target, pendulum_sampler, n_iters=5000)
    export_pt_and_onnx(pend, in_dim=2, out_dir=ROOT / "pendulum_2_32_32_1")

    print("\nAll three CBFs trained and exported.")


if __name__ == "__main__":
    main()
