#!/usr/bin/env python3
"""Independent check of the C++ double-precision reference (ref64 in methods.h).

For each example, the trained network is rebuilt in PyTorch in float64 with
the float32 weights of cbf.pt, and h, grad h, and the Hessian-vector product
H v are obtained from torch.autograd at random states. The same states are
passed to ref64::evaluate (compiled with g++), and the two results are
compared. Run from the repository root after 02_compile_headers.py:

    python tools/check_reference.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = [
    ("bicycle_4_32_32_1", [4, 32, 32, 1], nn.ReLU,
     [(-7, 7), (-5, 5), (-0.35, 0.35), (4, 12)]),
    ("vdp_2_64_64_1", [2, 64, 64, 1], nn.ReLU, [(-4, 4), (-3.5, 3.5)]),
    ("pendulum_2_32_32_1", [2, 32, 32, 1], nn.Softplus, [(-1.5, 1.5), (-3, 3)]),
]

SHIM = r"""
#include <cstdio>
#include "dual_cbf.h"
#include "methods.h"
int main() {
    int n = dual_cbf::INPUT_DIM;
    double x[16], v[16], h, g[16], Hv[16], mp;
    for (;;) {
        for (int i = 0; i < n; ++i) if (scanf("%lf", &x[i]) != 1) return 0;
        for (int i = 0; i < n; ++i) if (scanf("%lf", &v[i]) != 1) return 0;
        ref64::evaluate(x, v, &h, g, Hv, &mp);
        printf("%.17g", h);
        for (int i = 0; i < n; ++i) printf(" %.17g", g[i]);
        for (int i = 0; i < n; ++i) printf(" %.17g", Hv[i]);
        printf("\n");
    }
}
"""


def main() -> None:
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, widths, act, box in EXAMPLES:
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(act())
        model = nn.Sequential(*layers)
        model.load_state_dict(torch.load(ROOT / name / "cbf.pt"))
        model = model.double()   # float32 weights, float64 arithmetic

        n = widths[0]
        N = 500
        X = np.stack([rng.uniform(lo, hi, N) for lo, hi in box], axis=1)
        V = rng.standard_normal((N, n))

        tmp = Path(tempfile.mkdtemp(prefix="ref64_"))
        (tmp / "shim.cpp").write_text(SHIM)
        exe = tmp / "shim"
        subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(ROOT / name / "main"),
                        str(tmp / "shim.cpp"), "-o", str(exe)], check=True)
        text = "\n".join(" ".join(f"{t:.17g}" for t in np.concatenate([X[k], V[k]]))
                         for k in range(N))
        out = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True)
        C = np.array([[float(t) for t in line.split()] for line in out.stdout.strip().splitlines()])

        err = 0.0
        for k in range(N):
            x = torch.tensor(X[k], dtype=torch.float64, requires_grad=True)
            v = torch.tensor(V[k], dtype=torch.float64)
            h = model(x).squeeze()
            (g,) = torch.autograd.grad(h, x, create_graph=True)
            (Hv,) = torch.autograd.grad(g @ v, x, allow_unused=True)
            Hv = torch.zeros(n, dtype=torch.float64) if Hv is None else Hv
            ref = np.concatenate([[h.item()], g.detach().numpy(), Hv.detach().numpy()])
            scale = np.maximum(1.0, np.abs(ref))
            err = max(err, float(np.max(np.abs(C[k] - ref) / scale)))
        worst = max(worst, err)
        print(f"{name}: max scaled difference between ref64 and torch.autograd (float64)"
              f" over {N} states = {err:.3e}")
    print("PASS" if worst < 1e-10 else "FAIL")


if __name__ == "__main__":
    main()
