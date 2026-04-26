"""Host-side closed-loop simulations for the bicycle and Van der Pol examples.

We compile each generated dual_cbf.h into a tiny shared library and call it
via ctypes from Python. The same .h file is used on the ESP32-S3 in
production -- the host simulation merely uses it to render the closed-loop
trajectories that appear in Fig. 4 and Fig. 6 of the paper. (The pendulum
closed-loop is more involved and is omitted here; the timing claim for the
pendulum already comes from the ESP32-S3 measurement.)

Run after step 02:
    pip install numpy matplotlib cvxpy
    python 03_run_closed_loop.py

Outputs:
    bicycle_closed_loop.pdf
    vdp_phase.pdf
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Compile a generated dual_cbf.h into a shared library exposing a single
# C-linkage entry point: cbf_eval(x, f, G, m, h_out, Lf_out, Lg_out).
# ---------------------------------------------------------------------------

_C_SHIM = r"""
#include "dual_cbf.h"

extern "C" void cbf_eval(
    const float* x, const float* f, const float* G, int m,
    float* h, float* Lf, float* Lg)
{
    dual_cbf::evaluate_cbf(x, f, G, m, h, Lf, Lg);
}
"""


def compile_shim(header_path: Path) -> ctypes.CDLL:
    """Compile dual_cbf.h + a thin shim into a shared library, return CDLL."""
    tmp = Path(tempfile.mkdtemp(prefix="dual_cbf_shim_"))
    shim = tmp / "shim.cpp"
    shim.write_text(_C_SHIM)
    shutil.copy(header_path, tmp / "dual_cbf.h")
    so = tmp / "libcbf.so"
    cmd = [
        "g++", "-O2", "-std=c++17", "-fPIC", "-shared",
        "-I", str(tmp),
        str(shim), "-o", str(so),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"g++ failed:\n{res.stderr}")
    lib = ctypes.CDLL(str(so))
    lib.cbf_eval.restype = None
    lib.cbf_eval.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # f
        ctypes.POINTER(ctypes.c_float),  # G
        ctypes.c_int,                    # m
        ctypes.POINTER(ctypes.c_float),  # h
        ctypes.POINTER(ctypes.c_float),  # Lf
        ctypes.POINTER(ctypes.c_float),  # Lg
    ]
    return lib


def make_evaluator(header_path: Path, n: int, m: int):
    """Return a callable (x, f, G) -> (h, Lf, Lg) using the compiled shim."""
    lib = compile_shim(header_path)
    h_buf = (ctypes.c_float * 1)()
    Lf_buf = (ctypes.c_float * 1)()
    Lg_buf = (ctypes.c_float * m)()

    def evaluate(x: np.ndarray, f: np.ndarray, G: np.ndarray):
        x_f = np.ascontiguousarray(x, dtype=np.float32)
        f_f = np.ascontiguousarray(f, dtype=np.float32)
        # Row-major flatten -- G shape is (n, m)
        G_f = np.ascontiguousarray(G, dtype=np.float32).reshape(-1)
        lib.cbf_eval(
            x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            f_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            G_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m, h_buf, Lf_buf, Lg_buf,
        )
        return float(h_buf[0]), float(Lf_buf[0]), np.array(list(Lg_buf), dtype=np.float64)

    return evaluate


# ---------------------------------------------------------------------------
# QP solver: 1/2 ||u - u_nom||^2 s.t. Lf + Lg u >= -alpha * h, with simple
# input-bound constraints. Closed-form solution since constraint is scalar.
# ---------------------------------------------------------------------------

def solve_cbf_qp(
    u_nom: np.ndarray, Lf: float, Lg: np.ndarray, h: float,
    alpha: float, u_lo: np.ndarray, u_hi: np.ndarray,
) -> np.ndarray:
    """Project u_nom onto {u : Lf + Lg @ u >= -alpha h, u_lo <= u <= u_hi}."""
    m = u_nom.size
    rhs = -alpha * h - Lf  # need Lg @ u >= rhs
    if Lg @ u_nom >= rhs - 1e-9:
        u = u_nom.copy()
    else:
        # Projection of u_nom onto Lg @ u = rhs (single linear constraint)
        denom = Lg @ Lg
        if denom < 1e-12:
            u = u_nom.copy()  # gracefully degrade
        else:
            slack = rhs - Lg @ u_nom
            u = u_nom + (slack / denom) * Lg
    return np.minimum(np.maximum(u, u_lo), u_hi)


# ---------------------------------------------------------------------------
# Bicycle closed loop
# ---------------------------------------------------------------------------

def bicycle_closed_loop(out_pdf: Path) -> None:
    print("=== Bicycle closed-loop simulation ===")
    header = ROOT / "bicycle_4_32_32_1" / "main" / "dual_cbf.h"
    if not header.exists():
        print(f"  SKIP: {header} not found. Run 02_compile_headers.py first.")
        return
    evaluate = make_evaluator(header, n=4, m=2)

    # Dynamics: kinematic bicycle.
    #   state = (px, py, psi, v),  input = (a, delta)
    L_wb = 2.5
    def dynamics(x, u):
        px, py, psi, v = x
        return np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            v / L_wb * np.tan(u[1]),
            u[0],
        ])

    def f_G(x):
        _, _, psi, v = x
        f = np.array([v*np.cos(psi), v*np.sin(psi), 0.0, 0.0])
        G = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, v / L_wb],  # linearization of v/L * tan(delta) at delta=0
            [1.0, 0.0],
        ])
        return f, G

    # Initial condition: well inside the safe disk, low speed.
    x = np.array([2.0, 0.5, 0.1, 8.0])
    dt = 0.02
    T = 5.0
    n_steps = int(T / dt)
    alpha = 1.0
    u_lo = np.array([-3.0, -0.5])
    u_hi = np.array([ 3.0,  0.5])

    traj = np.zeros((n_steps + 1, 4))
    traj[0] = x
    h_history = np.zeros(n_steps + 1)
    delta_nom_history = np.zeros(n_steps + 1)
    delta_safe_history = np.zeros(n_steps + 1)

    h_history[0], _, _ = evaluate(x, *f_G(x))

    for k in range(n_steps):
        f, G = f_G(x)
        h, Lf, Lg = evaluate(x, f, G)
        # Nominal: drive toward the boundary (steering toward outside)
        delta_nom = 0.1
        u_nom = np.array([0.5, delta_nom])
        u = solve_cbf_qp(u_nom, Lf, Lg, h, alpha, u_lo, u_hi)
        delta_nom_history[k] = u_nom[1]
        delta_safe_history[k] = u[1]

        # Forward Euler
        x = x + dt * dynamics(x, u)
        traj[k + 1] = x
        h_next, _, _ = evaluate(x, *f_G(x))
        h_history[k + 1] = h_next

    delta_nom_history[-1] = delta_nom_history[-2]
    delta_safe_history[-1] = delta_safe_history[-2]

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    t = np.linspace(0, T, n_steps + 1)
    axes[0].plot(t, h_history, color="C0", lw=2)
    axes[0].axhline(0, color="red", lw=2)
    axes[0].fill_between(t, -1, 0, color="red", alpha=0.15)
    axes[0].set_ylabel(r"$h(x(t))$")
    axes[0].set_ylim(-0.5, max(2.0, h_history.max() * 1.1))
    axes[0].grid(True, ls="--", alpha=0.4)
    axes[1].plot(t, delta_nom_history, color="black", ls="--", label=r"Nominal $\delta_\mathrm{nom}$")
    axes[1].plot(t, delta_safe_history, color="C0", lw=2, label=r"Safe filtered $\delta^*$")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"Steering $\delta$ (rad)")
    axes[1].grid(True, ls="--", alpha=0.4)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  wrote {out_pdf.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Van der Pol closed loop
# ---------------------------------------------------------------------------

def vdp_phase(out_pdf: Path) -> None:
    print("=== Van der Pol closed-loop simulation ===")
    header = ROOT / "vdp_2_64_64_1" / "main" / "dual_cbf.h"
    if not header.exists():
        print(f"  SKIP: {header} not found. Run 02_compile_headers.py first.")
        return
    evaluate = make_evaluator(header, n=2, m=1)

    mu = 1.0
    def dynamics(x, u):
        return np.array([x[1], -x[0] + mu*(1 - x[0]**2)*x[1] + u[0]])
    def f_G(x):
        f = np.array([x[1], -x[0] + mu*(1 - x[0]**2)*x[1]])
        G = np.array([[0.0], [1.0]])
        return f, G

    dt = 0.02
    T = 12.0
    n_steps = int(T / dt)
    alpha = 1.0
    u_lo = np.array([-5.0])
    u_hi = np.array([ 5.0])

    # Two trajectories from the same x0: nominal (no filter) and filtered.
    x0 = np.array([-0.5, 0.0])

    # Nominal: no input, watch the limit cycle escape the safe set.
    x = x0.copy()
    nominal = [x.copy()]
    for _ in range(n_steps):
        x = x + dt * dynamics(x, np.array([0.0]))
        nominal.append(x.copy())
    nominal = np.array(nominal)

    # Filtered: same x0, CBF-filtered zero-input nominal.
    x = x0.copy()
    filtered = [x.copy()]
    for _ in range(n_steps):
        f, G = f_G(x)
        h, Lf, Lg = evaluate(x, f, G)
        u = solve_cbf_qp(np.array([0.0]), Lf, Lg, h, alpha, u_lo, u_hi)
        x = x + dt * dynamics(x, u)
        filtered.append(x.copy())
    filtered = np.array(filtered)

    # Plot phase portrait + zero level set of h
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sample h on a grid to draw the zero level set
    g = np.linspace(-4, 4, 200)
    g2 = np.linspace(-3.5, 3.5, 200)
    GX, GY = np.meshgrid(g, g2)
    H = np.zeros_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            xs = np.array([GX[i, j], GY[i, j]])
            f, Gm = f_G(xs)
            h_val, _, _ = evaluate(xs, f, Gm)
            H[i, j] = h_val

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(GX, GY, H >= 0, levels=[-0.5, 0.5, 1.5], colors=["#fdd", "#ddf"])
    ax.contour(GX, GY, H, levels=[0.0], colors="red", linewidths=2)
    ax.plot(nominal[:, 0], nominal[:, 1], color="black", ls="--", lw=1.5,
            label="Nominal trajectory")
    ax.plot(filtered[:, 0], filtered[:, 1], color="C0", lw=2,
            label=r"Safe filtered $x(t)$")
    ax.plot(*x0, "ko", ms=5)
    ax.annotate(r"$x(0)$", x0, textcoords="offset points", xytext=(-12, -12))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  wrote {out_pdf.relative_to(ROOT)}")


def main() -> None:
    bicycle_closed_loop(ROOT / "bicycle_closed_loop.pdf")
    vdp_phase(ROOT / "vdp_phase.pdf")
    print("\nFigures complete. They go into Fig. 4 and Fig. 6 of the paper.")


if __name__ == "__main__":
    main()
