"""Closed-loop sampled-data simulations on the host with the compiled headers.

The header that is flashed on the ESP32-S3 is compiled into a shared library
with g++ and called through ctypes, so the constraint coefficients come from
the same generated code. The plant is integrated with a fourth-order
Runge-Kutta scheme (20 substeps per control period) under a zero-order hold.

Two cases per example:
  ideal      exact state, input applied at the sampling instant
  perturbed  additive measurement noise and a one-period input delay

These simulations illustrate the behavior of the filter. They are not a
verification of continuous-time or sampled-data forward invariance.

Outputs:
  closed_loop.pdf     three panels (Van der Pol, bicycle, pendulum)
  sim_results.txt     SIM,<key>,<value> lines read by tools/collect.py
  also: grid check of the high-order CBF inequality for the pendulum barrier,
        a comparison of forward Euler with Runge-Kutta for the Van der Pol case,
        and a step refinement of the filtered trajectories (10, 20, 40, and 80
        Runge-Kutta substeps per control period)

Run after 02_compile_headers.py:
    python 03_run_closed_loop.py
"""
from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CXX_FLAGS = ["-O2", "-ffast-math", "-funroll-loops", "-std=c++17"]
TS = 0.02          # control period (s)
SUBSTEPS = 20      # Runge-Kutta substeps per control period
RESULTS: list[tuple[str, str]] = []

# Figure style: one style per curve in all panels. The solid curve is drawn first,
# and the dash-dot and dashed curves are drawn over it (zorder 3, 4, 5).
# The hues (light greenish blue, dark gold, crimson) and the black nominal curve pass an all-pairs color-vision-deficiency
# separation check, and each curve also has its own line style.
PLOT_DATA = {}             # arrays of the three panels, written to closed_loop_data.mat for plot_closed_loop.m
SAFE_FILL = "#e6ecf5"     # set on which the barrier is nonnegative
BOUNDARY = "#c8102e"      # zero level set of the barrier (crimson)
STYLE = {
    "ideal":     dict(color="#3fb6d3", lw=3.0, ls="-", zorder=3, label="filtered, ideal"),
    "perturbed": dict(color="#c77700", lw=1.3, ls=(0, (4.0, 1.4, 1.2, 1.4)), zorder=4, label="filtered, perturbed"),
    "nominal":   dict(color="#000000", lw=1.2, ls=(0, (2.8, 1.6)), zorder=5, label="nominal"),
}


def record(key: str, value) -> None:
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    RESULTS.append((key, text))
    print(f"  SIM,{key},{text}")


# ---------------------------------------------------------------------------
# Compiled header -> callable
# ---------------------------------------------------------------------------
_SHIM1 = r"""
#include "dual_cbf.h"
extern "C" void cbf_eval(const float* x, const float* f, const float* G, int m,
                         float* h, float* Lf, float* Lg)
{ dual_cbf::evaluate_cbf(x, f, G, m, h, Lf, Lg); }
"""
_SHIM2 = r"""
#include "dual_cbf.h"
extern "C" void hocbf_eval(const float* x, const float* f, const float* G,
                           const float* Df_f, const float* Df_G, int m,
                           float* h, float* Lf, float* L2f, float* LgLf)
{ dual_cbf::evaluate_hocbf(x, f, G, Df_f, Df_G, m, h, Lf, L2f, LgLf); }
"""
FP = ctypes.POINTER(ctypes.c_float)


def _build(header: Path, shim: str) -> ctypes.CDLL:
    tmp = Path(tempfile.mkdtemp(prefix="dual_cbf_shim_"))
    (tmp / "shim.cpp").write_text(shim)
    shutil.copy(header, tmp / "dual_cbf.h")
    so = tmp / "libcbf.so"
    subprocess.run(["g++", *CXX_FLAGS, "-fPIC", "-shared", "-I", str(tmp),
                    str(tmp / "shim.cpp"), "-o", str(so)], check=True)
    return ctypes.CDLL(str(so))


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def make_first_order(example: str, m: int):
    lib = _build(ROOT / example / "main" / "dual_cbf.h", _SHIM1)
    h, Lf, Lg = ctypes.c_float(), ctypes.c_float(), (ctypes.c_float * m)()

    def evaluate(x, f, G):
        xs, fs, Gs = _f32(x), _f32(f), _f32(G).reshape(-1)
        lib.cbf_eval(xs.ctypes.data_as(FP), fs.ctypes.data_as(FP), Gs.ctypes.data_as(FP),
                     m, ctypes.byref(h), ctypes.byref(Lf), Lg)
        return float(h.value), float(Lf.value), np.array(list(Lg), dtype=float)

    return evaluate


def make_second_order(example: str, m: int):
    lib = _build(ROOT / example / "main" / "dual_cbf.h", _SHIM2)
    h, Lf, L2f = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    LgLf = (ctypes.c_float * m)()

    def evaluate(x, f, G, Df_f, Df_G):
        a = [_f32(v).reshape(-1) for v in (x, f, G, Df_f, Df_G)]
        lib.hocbf_eval(*[v.ctypes.data_as(FP) for v in a], m, ctypes.byref(h),
                       ctypes.byref(Lf), ctypes.byref(L2f), LgLf)
        return (float(h.value), float(Lf.value), float(L2f.value),
                np.array(list(LgLf), dtype=float))

    return evaluate


# ---------------------------------------------------------------------------
# QP (7) with a box: exact breakpoint search (same algorithm as bench_types.h)
# ---------------------------------------------------------------------------
def qp_halfspace_box(un, a, b, lo, hi):
    u = np.clip(un, lo, hi)
    phi0 = float(a @ u)
    if phi0 >= b:
        return u, True
    bp = []
    for i in range(len(un)):
        if a[i] > 0:
            if un[i] < lo[i]:
                bp.append((lo[i] - un[i]) / a[i])
            if un[i] < hi[i]:
                bp.append((hi[i] - un[i]) / a[i])
        elif a[i] < 0:
            if un[i] > hi[i]:
                bp.append((hi[i] - un[i]) / a[i])
            if un[i] > lo[i]:
                bp.append((lo[i] - un[i]) / a[i])
    lam0 = 0.0
    for lam1 in sorted(bp):
        phi1 = float(a @ np.clip(un + lam1 * a, lo, hi))
        if phi1 >= b:
            lam = lam0 + (b - phi0) * (lam1 - lam0) / (phi1 - phi0)
            return np.clip(un + lam * a, lo, hi), True
        lam0, phi0 = lam1, phi1
    return np.clip(un + lam0 * a, lo, hi), False


# Number of control periods at which the constraint cannot be met inside the box,
# so that the QP has no solution and the search returns the fallback input
# (the box point that maximizes a.u).  Reset before each filtered simulation.
QP_FALLBACK = [0]


def qp_counted(un, a, b, lo, hi):
    u, feasible = qp_halfspace_box(un, a, b, lo, hi)
    if not feasible:
        QP_FALLBACK[0] += 1
    return u


# ---------------------------------------------------------------------------
# Sampled-data loop
# ---------------------------------------------------------------------------
def rk4(rhs, x, u, dt):
    k1 = rhs(x, u)
    k2 = rhs(x + 0.5 * dt * k1, u)
    k3 = rhs(x + 0.5 * dt * k2, u)
    k4 = rhs(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(rhs, controller, x0, T, m, noise=None, delay=False, seed=0, euler=False, substeps=SUBSTEPS):
    """Zero-order hold at period TS. controller(t, x_measured) -> u."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / TS))
    x = np.array(x0, dtype=float)
    X = np.zeros((n_steps + 1, len(x)))
    U = np.zeros((n_steps, m))
    X[0] = x
    u_prev = np.zeros(m)
    for k in range(n_steps):
        xm = x + (rng.standard_normal(len(x)) * noise if noise is not None else 0.0)
        u_new = controller(k * TS, xm)
        u = u_prev if delay else u_new      # one-period input delay
        u_prev = u_new
        if euler:
            x = x + TS * rhs(x, u)
        else:
            for _ in range(substeps):
                x = rk4(rhs, x, u, TS / substeps)
        X[k + 1] = x
        U[k] = u
    return X, U


REFINE = (10, 20, 40, 80)   # Runge-Kutta substeps per control period of the refinement study


def refinement(tag, rhs, controller, x0, T, m, noise, seed):
    """Step refinement of the two filtered cases.

    Records, for 10, 20, and 40 substeps, the largest Euclidean distance over the
    sampling times and over the ideal and perturbed cases between the state and
    the state computed with 80 substeps.
    """
    runs = {}
    for s in REFINE:
        Xi, _ = simulate(rhs, controller, x0, T, m, substeps=s)
        Xp, _ = simulate(rhs, controller, x0, T, m, noise=noise, delay=True, seed=seed, substeps=s)
        runs[s] = (Xi, Xp)
    for s in REFINE[:-1]:
        err = max(float(np.max(np.linalg.norm(runs[s][c] - runs[REFINE[-1]][c], axis=1))) for c in (0, 1))
        record(f"{tag}.refine_{s}", err)


# ---------------------------------------------------------------------------
# Example 2: Van der Pol
# ---------------------------------------------------------------------------
def vdp(ax):
    print("=== Van der Pol ===")
    ev = make_first_order("vdp_2_64_64_1", 1)
    mu, alpha = 1.0, 1.0
    lo, hi = np.array([-5.0]), np.array([5.0])

    def rhs(x, u):
        return np.array([x[1], -x[0] + mu * (1 - x[0] ** 2) * x[1] + u[0]])

    def fG(x):
        return np.array([x[1], -x[0] + mu * (1 - x[0] ** 2) * x[1]]), np.array([[0.0], [1.0]])

    def h_of(x):
        return ev(x, *fG(x))[0]

    def filt(t, x):
        f, G = fG(x)
        h, Lf, Lg = ev(x, f, G)
        return qp_counted(np.zeros(1), Lg, -alpha * h - Lf, lo, hi)

    x0, T = [-0.5, 0.0], 12.0
    Xn, _ = simulate(rhs, lambda t, x: np.zeros(1), x0, T, 1)
    QP_FALLBACK[0] = 0
    Xi, _ = simulate(rhs, filt, x0, T, 1)
    record("vdp.fallback_ideal", QP_FALLBACK[0])
    QP_FALLBACK[0] = 0
    Xp, _ = simulate(rhs, filt, x0, T, 1, noise=np.array([0.01, 0.01]), delay=True, seed=1)
    record("vdp.fallback_perturbed", QP_FALLBACK[0])
    Xe, _ = simulate(rhs, filt, x0, T, 1, euler=True)
    record("vdp.hmin_nominal", min(h_of(x) for x in Xn))
    record("vdp.hmin_ideal", min(h_of(x) for x in Xi))
    record("vdp.hmin_perturbed", min(h_of(x) for x in Xp))
    record("vdp.hmin_euler", min(h_of(x) for x in Xe))
    record("vdp.euler_vs_rk4", float(np.max(np.linalg.norm(Xe - Xi, axis=1))))
    refinement("vdp", rhs, filt, x0, T, 1, np.array([0.01, 0.01]), 1)

    g1, g2 = np.linspace(-4, 4, 401), np.linspace(-3.5, 3.5, 351)
    H = np.array([[h_of(np.array([a, b])) for a in g1] for b in g2])
    PLOT_DATA.update(vdp_g1=g1, vdp_g2=g2, vdp_h=H, vdp_ideal=Xi, vdp_perturbed=Xp, vdp_nominal=Xn, vdp_x0=np.asarray(x0, dtype=float))
    ax.contourf(g1, g2, H, levels=[0.0, 1e9], colors=[SAFE_FILL])
    ax.contour(g1, g2, H, levels=[0.0], colors=BOUNDARY, linewidths=1.2)
    ax.plot(Xi[:, 0], Xi[:, 1], **STYLE["ideal"])
    ax.plot(Xp[:, 0], Xp[:, 1], **STYLE["perturbed"])
    ax.plot(Xn[:, 0], Xn[:, 1], **STYLE["nominal"])
    ax.plot(*x0, "o", color="#222222", ms=3, zorder=6)
    ax.set_xlim(-4, 4); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$x_{(1)}$"); ax.set_ylabel(r"$x_{(2)}$")
    ax.set_title("(a) Van der Pol", fontsize=9)


# ---------------------------------------------------------------------------
# Example 1: kinematic bicycle
# ---------------------------------------------------------------------------
def bicycle(ax):
    print("=== Bicycle ===")
    ev = make_first_order("bicycle_4_32_32_1", 2)
    lw, alpha = 2.5, 1.0
    lo, hi = np.array([-3.0, -math.tan(0.5)]), np.array([3.0, math.tan(0.5)])

    def rhs(x, u):          # u = [a, tan(delta)]
        return np.array([x[3] * math.cos(x[2]), x[3] * math.sin(x[2]), x[3] / lw * u[1], u[0]])

    def fG(x):
        f = np.array([x[3] * math.cos(x[2]), x[3] * math.sin(x[2]), 0.0, 0.0])
        G = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, x[3] / lw], [1.0, 0.0]])
        return f, G

    # The lane is invariant under translation in p_x, and the target barrier does not
    # depend on p_x. The vehicle leaves the sampling interval p_x in [-7, 7] m, and thus
    # the filter uses hbar(x) = h_theta(P x) with P = diag(0, 1, 1, 1). Its Lie
    # derivatives along f and G are the outputs of the header at P x with the seeds
    # P f(x) and P G(x) (Theorem 1, with grad hbar(x) = P grad h_theta(P x)).
    P = np.diag([0.0, 1.0, 1.0, 1.0])

    def hbar(x):
        f, G = fG(x)
        return ev(P @ x, P @ f, P @ G)

    def h_of(x):
        return hbar(x)[0]

    def nominal(t, x):      # slow weave that leaves the lane if it is not filtered
        return np.array([0.2, 0.03 * math.sin(0.5 * t)])

    def filt(t, x):
        h, Lf, Lg = hbar(x)
        return qp_counted(nominal(t, x), Lg, -alpha * h - Lf, lo, hi)

    x0, T = [0.0, 0.0, 0.0, 6.0], 12.0
    noise = np.array([0.02, 0.02, 0.002, 0.02])
    Xn, _ = simulate(rhs, nominal, x0, T, 2)
    QP_FALLBACK[0] = 0
    Xi, _ = simulate(rhs, filt, x0, T, 2)
    record("bic.fallback_ideal", QP_FALLBACK[0])
    QP_FALLBACK[0] = 0
    Xp, _ = simulate(rhs, filt, x0, T, 2, noise=noise, delay=True, seed=2)
    record("bic.fallback_perturbed", QP_FALLBACK[0])
    e = lambda X: X[:, 1] + 1.0 * X[:, 3] * np.sin(X[:, 2])   # look-ahead lateral position
    record("bic.hmin_nominal", min(h_of(x) for x in Xn))
    record("bic.hmin_ideal", min(h_of(x) for x in Xi))
    record("bic.hmin_perturbed", min(h_of(x) for x in Xp))
    record("bic.emax_nominal", float(np.max(np.abs(e(Xn)))))
    record("bic.emax_ideal", float(np.max(np.abs(e(Xi)))))
    record("bic.emax_perturbed", float(np.max(np.abs(e(Xp)))))
    refinement("bic", rhs, filt, x0, T, 2, noise, 2)
    t = np.arange(len(Xi)) * TS
    PLOT_DATA.update(bic_t=t, bic_ideal=e(Xi), bic_perturbed=e(Xp), bic_nominal=e(Xn), bic_bound=3.5, bic_T=float(T))
    ax.axhspan(-3.5, 3.5, color=SAFE_FILL, lw=0)
    ax.axhline(3.5, color=BOUNDARY, lw=1.2); ax.axhline(-3.5, color=BOUNDARY, lw=1.2)
    ax.plot(t, e(Xi), **STYLE["ideal"])
    ax.plot(t, e(Xp), **STYLE["perturbed"])
    ax.plot(t, e(Xn), **STYLE["nominal"])
    ax.set_xlim(0, T); ax.set_ylim(-9, 9)
    ax.set_xlabel(r"$t$ (s)"); ax.set_ylabel(r"$p_y+\tau_{\rm h}v\,\sin\psi$ (m)")
    ax.set_title("(b) Bicycle", fontsize=9)


# ---------------------------------------------------------------------------
# Example 3: inverted pendulum (relative degree two)
# ---------------------------------------------------------------------------
def pendulum(ax):
    print("=== Pendulum ===")
    ev = make_second_order("pendulum_2_32_32_1", 1)
    g, l, mass, k1, k2 = 9.81, 1.0, 1.0, 3.0, 6.0
    b = 1.0 / (mass * l * l)
    lo, hi = np.array([-20.0]), np.array([20.0])

    def rhs(x, u):
        return np.array([x[1], g / l * math.sin(x[0]) + b * u[0]])

    def terms(x):
        f = np.array([x[1], g / l * math.sin(x[0])])
        G = np.array([[0.0], [b]])
        c = g / l * math.cos(x[0])
        Df = np.array([[0.0, 1.0], [c, 0.0]])
        return ev(x, f, G, Df @ f, Df @ G)

    def constraint(x):      # a u >= rhs_b
        h, Lf, L2f, LgLf = terms(x)
        return LgLf, -(L2f + k1 * Lf + k2 * (Lf + k1 * h)), h, Lf

    def filt(t, x):
        a, rb, _, _ = constraint(x)
        return qp_counted(np.zeros(1), a, rb, lo, hi)

    x0, T = [0.2, 0.0], 5.0
    Xn, _ = simulate(rhs, lambda t, x: np.zeros(1), x0, 1.2, 1)
    QP_FALLBACK[0] = 0
    Xi, _ = simulate(rhs, filt, x0, T, 1)
    record("pen.fallback_ideal", QP_FALLBACK[0])
    QP_FALLBACK[0] = 0
    Xp, _ = simulate(rhs, filt, x0, T, 1, noise=np.array([0.002, 0.01]), delay=True, seed=3)
    record("pen.fallback_perturbed", QP_FALLBACK[0])
    h_of = lambda x: terms(x)[0]
    record("pen.hmin_ideal", min(h_of(x) for x in Xi))
    record("pen.hmin_perturbed", min(h_of(x) for x in Xp))
    record("pen.phimax_ideal", float(np.max(np.abs(Xi[:, 0]))))
    record("pen.phimax_perturbed", float(np.max(np.abs(Xp[:, 0]))))

    record("pen.dphimax_ideal", float(np.max(np.abs(Xi[:, 1]))))
    record("pen.dphimax_perturbed", float(np.max(np.abs(Xp[:, 1]))))
    refinement("pen", rhs, filt, x0, T, 1, np.array([0.002, 0.01]), 3)

    # Grid check of the high-order CBF inequality (numerical check, not a certificate):
    # at each grid point with h >= 0 and psi_1 = Lf + k1 h >= 0, test whether some
    # u in [lo, hi] satisfies  L2f + LgLf u + k1 Lf + k2 (Lf + k1 h) >= 0.
    def grid_check(tag: str, dphi_max: float) -> None:
        n_in, n_fail, worst = 0, 0, float("inf")
        for phi in np.linspace(-1.0, 1.0, 201):
            for dphi in np.linspace(-dphi_max, dphi_max, 201):
                a, rb, h, Lf = constraint(np.array([phi, dphi]))
                if h < 0.0 or Lf + k1 * h < 0.0:
                    continue
                n_in += 1
                margin = abs(a[0]) * hi[0] - rb       # sup_u (a u) - rb
                worst = min(worst, margin)
                if margin < 0.0:
                    n_fail += 1
        record(f"pen.grid{tag}_points", n_in)
        record(f"pen.grid{tag}_violations", n_fail)
        record(f"pen.grid{tag}_min_margin", float(worst))

    grid_check("A", 1.5)     # |phi_dot| <= 1.5 rad/s
    grid_check("B", 3.0)     # |phi_dot| <= 3 rad/s (benchmark sampling box)

    PLOT_DATA.update(pen_t_ideal=np.arange(len(Xi)) * TS, pen_ideal=Xi[:, 0], pen_t_perturbed=np.arange(len(Xp)) * TS, pen_perturbed=Xp[:, 0],
                     pen_t_nominal=np.arange(len(Xn)) * TS, pen_nominal=Xn[:, 0], pen_bound=math.pi / 4, pen_T=float(T))
    ax.axhspan(-math.pi / 4, math.pi / 4, color=SAFE_FILL, lw=0)
    ax.axhline(math.pi / 4, color=BOUNDARY, lw=1.2); ax.axhline(-math.pi / 4, color=BOUNDARY, lw=1.2)
    ax.plot(np.arange(len(Xi)) * TS, Xi[:, 0], **STYLE["ideal"])
    ax.plot(np.arange(len(Xp)) * TS, Xp[:, 0], **STYLE["perturbed"])
    ax.plot(np.arange(len(Xn)) * TS, Xn[:, 0], **STYLE["nominal"])
    ax.set_xlim(0, T); ax.set_ylim(-0.2, 1.3)
    ax.set_xlabel(r"$t$ (s)"); ax.set_ylabel(r"$\phi$ (rad)")
    ax.set_title("(c) Pendulum", fontsize=9)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    vdp(axes[0]); bicycle(axes[1]); pendulum(axes[2])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Patch(facecolor=SAFE_FILL, edgecolor="none"), Line2D([], [], color=BOUNDARY, lw=1.2)]
    labels += [r"barrier $\geq 0$", "boundary"]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), handlelength=3.0, columnspacing=1.6)
    for ax in axes:
        ax.grid(True, ls=":", lw=0.4)
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ROOT / "closed_loop.pdf", bbox_inches="tight", pad_inches=0.03)
    from scipy.io import savemat
    savemat(ROOT / "closed_loop_data.mat", PLOT_DATA)   # input of plot_closed_loop.m
    record("refine_max", max(float(v) for k, v in RESULTS if ".refine_" in k))
    record("period_s", TS)
    (ROOT / "sim_results.txt").write_text("".join(f"SIM,{k},{v}\n" for k, v in RESULTS))
    print("wrote closed_loop.pdf and sim_results.txt")


if __name__ == "__main__":
    main()
