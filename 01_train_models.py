"""Train neural CBFs for the three paper examples and export them to ONNX.

Run from the repository root:
    pip install torch numpy onnx
    python 01_train_models.py

This produces:
    bicycle_4_32_32_1/cbf.onnx
    bicycle_4_32_32_1/cbf.pt
    vdp_2_64_64_1/cbf.onnx
    vdp_2_64_64_1/cbf.pt
    pendulum_2_32_32_1/cbf.onnx        (softplus, function of the angle only, relative_degree=2)
    pendulum_2_32_32_1/cbf.pt
    depth_{2,4,8}x32/cbf.onnx          (ReLU, width 32, 2/4/8 hidden layers)
    depth_{2,4,8}x32/cbf.pt

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

# Lane half-width (m) and look-ahead time (s) of the bicycle target.
BICYCLE_HALF_WIDTH = 3.5
BICYCLE_LOOKAHEAD = 1.0


def bicycle_target(x: torch.Tensor) -> torch.Tensor:
    """4-state bicycle: state = (px, py, psi, v), lane along the px axis.

    Look-ahead lane-keeping barrier
        h(x) = w^2 - e(x)^2,   e(x) = py + tau * v * sin(psi),
    where e is the lateral position predicted tau seconds ahead. Both inputs
    appear in the derivative of h along the dynamics:
        L_G h = -2 e [tau sin(psi),  tau v^2 cos(psi) / l_w],
    so h has relative degree one wherever e != 0. (The previous target, a
    smoothed minimum of a position margin and a speed margin, had L_G h = 0
    for its position margin and was not a valid CBF.)
    """
    py, psi, v = x[..., 1], x[..., 2], x[..., 3]
    e = py + BICYCLE_LOOKAHEAD * v * torch.sin(psi)
    return BICYCLE_HALF_WIDTH ** 2 - e ** 2


def vdp_target(x: torch.Tensor) -> torch.Tensor:
    """Van der Pol: state = (x1, x2). Safe set = ellipse x1^2/9 + x2^2/4 <= 1."""
    x1, x2 = x[..., 0], x[..., 1]
    return 1.0 - (x1 ** 2 / 9.0 + x2 ** 2 / 4.0)


def pendulum_target(x: torch.Tensor) -> torch.Tensor:
    """Inverted pendulum: state = (phi, phi_dot). Safe set = |phi| <= pi/4.

    The target depends on the angle only, so that L_G h = 0 and the barrier
    has relative degree two (Example 3 of the paper).
    """
    phi = x[..., 0]
    return (math.pi / 4.0) ** 2 - phi ** 2


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
    frozen_zero_inputs: tuple[int, ...] = (),
    lr_drop_at: float | None = None,
) -> nn.Sequential:
    """Least-squares regression of the network onto target_fn.

    frozen_zero_inputs lists input indices whose first-layer weight columns
    are set to zero before training and kept at zero after each step, so that
    the trained network does not depend on those inputs. If lr_drop_at is
    given, the learning rate is divided by five after that fraction of the
    iterations.
    """
    def zero_columns() -> None:
        with torch.no_grad():
            for j in frozen_zero_inputs:
                model[0].weight[:, j].zero_()

    zero_columns()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(n_iters):
        if lr_drop_at is not None and it == int(lr_drop_at * n_iters):
            for group in optim.param_groups:
                group["lr"] = lr / 5.0
        x = sampler(batch_size)
        y_target = target_fn(x).unsqueeze(-1)
        y_pred = model(x)
        loss = (y_pred - y_target).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        zero_columns()
        if it % 1000 == 0 or it == n_iters - 1:
            print(f"  iter {it:5d}  loss = {loss.item():.4e}")
    with torch.no_grad():  # fit quality on fresh samples, reported in the log
        x = sampler(100_000)
        err = model(x).squeeze(-1) - target_fn(x)
        print(f"  test RMSE = {err.pow(2).mean().sqrt().item():.4e}"
              f"   max |error| = {err.abs().max().item():.4e}")
    return model


def export_pt_and_onnx(model: nn.Sequential, in_dim: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "cbf.pt"
    onnx_path = out_dir / "cbf.onnx"
    torch.save(model.state_dict(), pt_path)

    # ONNX export: use a single-sample input so dimensions are literal,
    # not symbolic -- the compiler reads concrete (n_in, n_out) shapes.
    sample = torch.zeros(1, in_dim)
    kwargs = dict(
        input_names=["x"],
        output_names=["h"],
        opset_version=14,
        do_constant_folding=True,
    )
    try:
        # TorchScript-based exporter: one Gemm node per layer, weights inline.
        # Recent PyTorch releases default to a different exporter.
        torch.onnx.export(model, sample, onnx_path.as_posix(), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch: no `dynamo` argument, same exporter
        torch.onnx.export(model, sample, onnx_path.as_posix(), **kwargs)
    print(f"  wrote {pt_path.name} and {onnx_path.name}")


# ---------------------------------------------------------------------------
# Samplers (uniform over each system's relevant state region)
# ---------------------------------------------------------------------------

def bicycle_sampler(n: int) -> torch.Tensor:
    px = torch.empty(n).uniform_(-7, 7)
    py = torch.empty(n).uniform_(-5, 5)
    psi = torch.empty(n).uniform_(-0.35, 0.35)
    v = torch.empty(n).uniform_(4, 12)
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
    train_cbf(bicycle, bicycle_target, bicycle_sampler, n_iters=20000, lr_drop_at=0.7)
    export_pt_and_onnx(bicycle, in_dim=4, out_dir=ROOT / "bicycle_4_32_32_1")

    print("\n=== [2/3] Van der Pol 2-64-64-1 (ReLU) ===")
    vdp = make_relu_net([2, 64, 64, 1])
    train_cbf(vdp, vdp_target, vdp_sampler, n_iters=5000)
    export_pt_and_onnx(vdp, in_dim=2, out_dir=ROOT / "vdp_2_64_64_1")

    print("\n=== [3/3] Pendulum 2-32-32-1 (Softplus, for relative_degree=2) ===")
    pend = make_softplus_net([2, 32, 32, 1])
    # Velocity column of the first layer is zero and frozen: h depends on phi only.
    train_cbf(pend, pendulum_target, pendulum_sampler, n_iters=5000,
              frozen_zero_inputs=(1,))
    export_pt_and_onnx(pend, in_dim=2, out_dir=ROOT / "pendulum_2_32_32_1")

    # Depth sweep: same target and sampling box as the Van der Pol example,
    # ReLU, width 32, with 2, 4, and 8 hidden layers.
    for hidden in (2, 4, 8):
        print(f"\n=== Depth sweep: 2-{'-'.join(['32'] * hidden)}-1 (ReLU) ===")
        net = make_relu_net([2] + [32] * hidden + [1])
        train_cbf(net, vdp_target, vdp_sampler, n_iters=5000)
        export_pt_and_onnx(net, in_dim=2, out_dir=ROOT / f"depth_{hidden}x32")

    print("\nAll CBFs trained and exported.")


if __name__ == "__main__":
    main()
