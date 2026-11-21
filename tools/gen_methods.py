#!/usr/bin/env python3
"""Emit methods.h for one example from its compiled dual_cbf.h.

    python tools/gen_methods.py <example>/main/dual_cbf.h --m <inputs>

The emitted header is included AFTER dual_cbf.h and reuses the weight arrays
dual_cbf::W0, dual_cbf::b0, ... of the compiled header, so that all methods
evaluate the same network with bit-identical parameters.  Every method is
ahead-of-time code with compile-time-sized static buffers and no allocation.

Relative degree one (header exposes evaluate_cbf and evaluate_cbf_shared):
  rev_static   reverse mode, statically allocated cache and adjoint buffers
  ref64        double-precision reference (h, gradient)

Relative degree two (header exposes evaluate_cbf_2nd_order):
  jet3         three-component truncated algebra (1, eta1, eta2), eta1^2 = 2 eta2;
               mixed terms by polarization (two extra passes per input)
  rev2_static  exact second order by forward-over-reverse, statically allocated
  ref64        double-precision reference (h, gradient, Hessian-vector product)

The topology and the activation kinds are read from dual_cbf.h itself.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ACT_CODE = {"identity": 0, "relu": 1, "softplus": 2, "tanh": 3, "sigmoid": 4}


# --------------------------------------------------------------------------
# Activation expressions.  The float value/derivative expressions are the
# ones emitted by dual_cbf_compiler 0.1.1, so primal values agree bitwise.
# --------------------------------------------------------------------------
def act_f(kind: str, t: str) -> str:
    return {
        "relu": f"(({t}) > 0.0f ? ({t}) : 0.0f)",
        "softplus": f"((({t}) > 20.0f) ? ({t}) : logf(1.0f + expf({t})))",
        "tanh": f"tanhf({t})",
        "sigmoid": f"(1.0f / (1.0f + expf(-({t}))))",
        "identity": f"({t})",
    }[kind]


def dact_f(kind: str, x: str, s: str) -> str:
    return {
        "relu": f"(({x}) > 0.0f ? 1.0f : 0.0f)",
        "softplus": f"(1.0f / (1.0f + expf(-({x}))))",
        "tanh": f"(1.0f - ({s}) * ({s}))",
        "sigmoid": f"(({s}) * (1.0f - ({s})))",
        "identity": "1.0f",
    }[kind]


def ddact_f(kind: str, x: str, s: str, sp: str) -> str:
    return {
        "relu": "0.0f",
        "softplus": f"(({sp}) * (1.0f - ({sp})))",
        "tanh": f"(-2.0f * ({s}) * ({sp}))",
        "sigmoid": f"(({sp}) * (1.0f - 2.0f * ({s})))",
        "identity": "0.0f",
    }[kind]


def dact_from_post_f(kind: str, p: str) -> str:
    """sigma'(a) written in terms of the stored post-activation p = sigma(a)."""
    return {
        "relu": f"(({p}) > 0.0f ? 1.0f : 0.0f)",
        "softplus": f"(-expm1f(-({p})))",
        "tanh": f"(1.0f - ({p}) * ({p}))",
        "sigmoid": f"(({p}) * (1.0f - ({p})))",
        "identity": "1.0f",
    }[kind]


def act_d(kind: str, t: str) -> str:
    return {
        "relu": f"(({t}) > 0.0 ? ({t}) : 0.0)",
        "softplus": f"((({t}) > 0.0) ? (({t}) + log1p(exp(-({t})))) : log1p(exp({t})))",
        "tanh": f"tanh({t})",
        "sigmoid": f"(1.0 / (1.0 + exp(-({t}))))",
        "identity": f"({t})",
    }[kind]


def dact_d(kind: str, x: str, s: str) -> str:
    return {
        "relu": f"(({x}) > 0.0 ? 1.0 : 0.0)",
        "softplus": f"(1.0 / (1.0 + exp(-({x}))))",
        "tanh": f"(1.0 - ({s}) * ({s}))",
        "sigmoid": f"(({s}) * (1.0 - ({s})))",
        "identity": "1.0",
    }[kind]


def ddact_d(kind: str, x: str, s: str, sp: str) -> str:
    return {
        "relu": "0.0",
        "softplus": f"(({sp}) * (1.0 - ({sp})))",
        "tanh": f"(-2.0 * ({s}) * ({sp}))",
        "sigmoid": f"(({sp}) * (1.0 - 2.0 * ({s})))",
        "identity": "0.0",
    }[kind]


# --------------------------------------------------------------------------
def parse_header(text: str):
    m = re.search(r"LAYER_WIDTHS\[DEPTH \+ 1\]\s*=\s*\{([^}]*)\}", text)
    if not m:
        raise SystemExit("LAYER_WIDTHS not found: is this a dual_cbf_compiler header?")
    widths = [int(t) for t in m.group(1).split(",")]
    second = "evaluate_cbf_2nd_order" in text
    pat = r"// Hyper-dual activation: (\w+)" if second else r"// Dual activation: (\w+)"
    acts = re.findall(pat, text)
    if len(acts) != len(widths) - 2:
        raise SystemExit(f"expected {len(widths) - 2} hidden activations, found {len(acts)}")
    if widths[-1] != 1:
        raise SystemExit("scalar-output networks only")
    if len(widths) < 3:
        raise SystemExit("at least one hidden layer is required")
    return widths, acts, second


class Out:
    def __init__(self):
        self.lines = []
        self.ind = 0

    def w(self, s: str = ""):
        self.lines.append(("    " * self.ind + s) if s else "")

    def open(self, s: str):
        self.w(s)
        self.ind += 1

    def close(self, s: str = "}"):
        self.ind -= 1
        self.w(s)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# --------------------------------------------------------------------------
# Relative degree one
# --------------------------------------------------------------------------
def _mul_dpost(kind: str, acc: str, p: str) -> str:
    if kind == "relu":
        return f"(({p}) > 0.0f ? ({acc}) : 0.0f)"
    return f"(({acc}) * {dact_from_post_f(kind, p)})"


def emit_rev_static(o: Out, widths, acts, m: int):
    d = len(widths) - 1
    hidden = widths[1:-1]
    maxh = max(hidden)
    n0 = widths[0]
    o.w("// ---------------------------------------------------------------------")
    o.w("// rev_static: reverse mode compiled ahead of time.  The forward sweep")
    o.w("// stores the post-activations of all hidden layers in static arrays; the")
    o.w("// backward sweep reads them in reverse order.  No heap, no run-time graph.")
    o.w("// ---------------------------------------------------------------------")
    o.open("namespace rev_static {")
    o.w(f"static constexpr int CACHE_FLOATS = {sum(hidden)};")
    o.w(f"static constexpr int STATIC_FLOATS = {sum(hidden) + 2 * maxh};")
    for k, n in enumerate(hidden, start=1):
        o.w(f"static float P{k}[{n}];  // post-activations of hidden layer {k}")
    o.w(f"static float gA[{maxh}];")
    o.w(f"static float gB[{maxh}];")
    o.w()
    o.open("inline void evaluate_cbf(const float* x, const float* f, const float* G,"
           " float* h, float* Lf, float* Lg) {")
    o.w("// forward sweep")
    for li in range(d):
        n_in, n_out = widths[li], widths[li + 1]
        src = "x" if li == 0 else f"P{li}"
        o.open(f"for (int i = 0; i < {n_out}; ++i) {{  // layer {li}: {n_in} -> {n_out}")
        o.w(f"float acc = dual_cbf::b{li}[i];")
        o.w(f"for (int j = 0; j < {n_in}; ++j) acc += dual_cbf::W{li}[i * {n_in} + j] * {src}[j];")
        if li < d - 1:
            o.w(f"P{li + 1}[i] = {act_f(acts[li], 'acc')};")
        else:
            o.w("*h = acc;")
        o.close()
    o.w("// backward sweep")
    k = d - 1  # last hidden layer index
    o.w(f"for (int j = 0; j < {widths[k]}; ++j) "
        f"gA[j] = {_mul_dpost(acts[k - 1], f'dual_cbf::W{d - 1}[j]', f'P{k}[j]')};")
    cur, nxt = "gA", "gB"
    for li in range(d - 2, 0, -1):
        n_in, n_out = widths[li], widths[li + 1]
        o.w(f"for (int j = 0; j < {n_in}; ++j) {nxt}[j] = 0.0f;")
        o.open(f"for (int k = 0; k < {n_out}; ++k) {{  // W{li}^T")
        o.w(f"float gk = {cur}[k];")
        o.w(f"for (int j = 0; j < {n_in}; ++j) {nxt}[j] += dual_cbf::W{li}[k * {n_in} + j] * gk;")
        o.close()
        o.w(f"for (int j = 0; j < {n_in}; ++j) "
            f"{nxt}[j] = {_mul_dpost(acts[li - 1], f'{nxt}[j]', f'P{li}[j]')};")
        cur, nxt = nxt, cur
    o.w(f"float grad[{n0}];")
    o.w(f"for (int j = 0; j < {n0}; ++j) grad[j] = 0.0f;")
    o.open(f"for (int k = 0; k < {widths[1]}; ++k) {{  // W0^T")
    o.w(f"float gk = {cur}[k];")
    o.w(f"for (int j = 0; j < {n0}; ++j) grad[j] += dual_cbf::W0[k * {n0} + j] * gk;")
    o.close()
    o.w("float lf = 0.0f;")
    o.w(f"for (int k = 0; k < {n0}; ++k) lf += grad[k] * f[k];")
    o.w("*Lf = lf;")
    o.open(f"for (int j = 0; j < {m}; ++j) {{")
    o.w("float lg = 0.0f;")
    o.w(f"for (int k = 0; k < {n0}; ++k) lg += grad[k] * G[k * {m} + j];")
    o.w("Lg[j] = lg;")
    o.close()
    o.close()
    o.close("}  // namespace rev_static")
    o.w()


# --------------------------------------------------------------------------
# Relative degree two
# --------------------------------------------------------------------------
def emit_jet3(o: Out, widths, acts):
    d = len(widths) - 1
    maxw = max(widths)
    n0 = widths[0]
    o.w("// ---------------------------------------------------------------------")
    o.w("// jet3: three-component truncated algebra with basis (1, eta1, eta2),")
    o.w("// eta1^2 = 2 eta2, eta1 eta2 = eta2^2 = 0.  The seed x + v eta1 + a eta2")
    o.w("// returns h, h'(x) v, and v^T H v + h'(x) a.  Mixed terms follow from the")
    o.w("// polarization identity at two extra passes per input.")
    o.w("// ---------------------------------------------------------------------")
    o.open("namespace jet3 {")
    o.w(f"static constexpr int STATIC_FLOATS = {3 * maxw};")
    o.w(f"static float y0[{maxw}];")
    o.w(f"static float y1[{maxw}];")
    o.w(f"static float y2[{maxw}];")
    o.w()
    o.open("inline void pass(const float* x, const float* v, const float* a,"
           " float* o0, float* o1, float* o2) {")
    o.w(f"for (int i = 0; i < {n0}; ++i) {{ y0[i] = x[i]; y1[i] = v[i]; y2[i] = a[i]; }}")
    for li in range(d):
        n_in, n_out = widths[li], widths[li + 1]
        o.open(f"{{  // layer {li}: {n_in} -> {n_out}")
        o.w(f"float t0[{n_out}];")
        o.w(f"float t1[{n_out}];")
        o.w(f"float t2[{n_out}];")
        o.open(f"for (int i = 0; i < {n_out}; ++i) {{")
        o.w(f"float c0 = dual_cbf::b{li}[i];")
        o.w("float c1 = 0.0f, c2 = 0.0f;")
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"float w = dual_cbf::W{li}[i * {n_in} + j];")
        o.w("c0 += w * y0[j];")
        o.w("c1 += w * y1[j];")
        o.w("c2 += w * y2[j];")
        o.close()
        o.w("t0[i] = c0; t1[i] = c1; t2[i] = c2;")
        o.close()
        if li < d - 1:
            k = acts[li]
            o.open(f"for (int i = 0; i < {n_out}; ++i) {{  // {k}")
            o.w("float pre = t0[i];")
            o.w(f"float sig = {act_f(k, 'pre')};")
            o.w(f"float sig_p = {dact_f(k, 'pre', 'sig')};")
            o.w(f"float sig_pp = {ddact_f(k, 'pre', 'sig', 'sig_p')};")
            o.w("y0[i] = sig;")
            o.w("y1[i] = sig_p * t1[i];")
            o.w("y2[i] = sig_pp * t1[i] * t1[i] + sig_p * t2[i];")
            o.close()
        else:
            o.w("*o0 = t0[0]; *o1 = t1[0]; *o2 = t2[0];")
        o.close()
    o.close()
    o.w()
    o.open("inline void evaluate_cbf_2nd_order(const float* x, const float* f, const float* G,"
           " const float* Df_f, const float* Df_G, int m,"
           " float* h, float* Lf, float* L2f, float* LgLf) {")
    o.w("pass(x, f, Df_f, h, Lf, L2f);")
    o.open("for (int j = 0; j < m; ++j) {")
    o.w(f"float vp[{n0}], ap[{n0}], vm[{n0}], am[{n0}];")
    o.open(f"for (int k = 0; k < {n0}; ++k) {{")
    o.w("float g = G[k * m + j];")
    o.w("float dg = Df_G[k * m + j];")
    o.w("vp[k] = f[k] + g;  ap[k] = 2.0f * dg;")
    o.w("vm[k] = f[k] - g;  am[k] = -2.0f * dg;")
    o.close()
    o.w("float e0, e1, qp, qm;")
    o.w("pass(x, vp, ap, &e0, &e1, &qp);")
    o.w("pass(x, vm, am, &e0, &e1, &qm);")
    o.w("LgLf[j] = 0.25f * (qp - qm);")
    o.close()
    o.close()
    o.close("}  // namespace jet3")
    o.w()


def emit_rev2_static(o: Out, widths, acts):
    d = len(widths) - 1
    hidden = widths[1:-1]
    maxh = max(hidden)
    maxw = max(widths)
    n0 = widths[0]
    o.w("// ---------------------------------------------------------------------")
    o.w("// rev2_static: exact second order by forward-over-reverse.  A tangent")
    o.w("// forward sweep along f stores sigma'(a_i) and sigma''(a_i) da_i for all")
    o.w("// hidden layers; one backward sweep returns grad h and H f.  Then")
    o.w("//   L_f h = grad.f,  L_f^2 h = f.(H f) + grad.(f' f),")
    o.w("//   L_gj L_f h = g_j.(H f) + grad.(f' g_j).")
    o.w("// ---------------------------------------------------------------------")
    o.open("namespace rev2_static {")
    o.w(f"static constexpr int CACHE_FLOATS = {2 * sum(hidden)};")
    o.w(f"static constexpr int STATIC_FLOATS = {2 * sum(hidden) + 2 * maxw + 4 * maxh};")
    for k, n in enumerate(hidden, start=1):
        o.w(f"static float P{k}[{n}];  // sigma'(a_{k})")
        o.w(f"static float Q{k}[{n}];  // sigma''(a_{k}) * da_{k}")
    o.w(f"static float z[{maxw}];")
    o.w(f"static float zd[{maxw}];")
    for nm in ("gA", "gdA", "gB", "gdB"):
        o.w(f"static float {nm}[{maxh}];")
    o.w()
    o.open("inline void evaluate_cbf_2nd_order(const float* x, const float* f, const float* G,"
           " const float* Df_f, const float* Df_G, int m,"
           " float* h, float* Lf, float* L2f, float* LgLf) {")
    o.w(f"for (int i = 0; i < {n0}; ++i) {{ z[i] = x[i]; zd[i] = f[i]; }}")
    o.w("// tangent forward sweep")
    for li in range(d):
        n_in, n_out = widths[li], widths[li + 1]
        o.open(f"{{  // layer {li}: {n_in} -> {n_out}")
        o.w(f"float ta[{n_out}];")
        o.w(f"float td[{n_out}];")
        o.open(f"for (int i = 0; i < {n_out}; ++i) {{")
        o.w(f"float acc = dual_cbf::b{li}[i];")
        o.w("float accd = 0.0f;")
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"float w = dual_cbf::W{li}[i * {n_in} + j];")
        o.w("acc += w * z[j];")
        o.w("accd += w * zd[j];")
        o.close()
        o.w("ta[i] = acc; td[i] = accd;")
        o.close()
        if li < d - 1:
            k = acts[li]
            o.open(f"for (int i = 0; i < {n_out}; ++i) {{  // {k}")
            o.w("float pre = ta[i];")
            o.w(f"float sig = {act_f(k, 'pre')};")
            o.w(f"float sig_p = {dact_f(k, 'pre', 'sig')};")
            o.w(f"float sig_pp = {ddact_f(k, 'pre', 'sig', 'sig_p')};")
            o.w("z[i] = sig;")
            o.w("zd[i] = sig_p * td[i];")
            o.w(f"P{li + 1}[i] = sig_p;")
            o.w(f"Q{li + 1}[i] = sig_pp * td[i];")
            o.close()
        else:
            o.w("*h = ta[0];")
            o.w("*Lf = td[0];   // h'(x) f, the tangent of the output")
        o.close()
    o.w("// backward sweep for the adjoint g and its tangent gd")
    k = d - 1
    o.open(f"for (int j = 0; j < {widths[k]}; ++j) {{")
    o.w(f"float s = dual_cbf::W{d - 1}[j];")
    o.w(f"gA[j] = P{k}[j] * s;")
    o.w(f"gdA[j] = Q{k}[j] * s;")
    o.close()
    cur, nxt = "A", "B"
    for li in range(d - 2, 0, -1):
        n_in, n_out = widths[li], widths[li + 1]
        o.w(f"for (int j = 0; j < {n_in}; ++j) {{ g{nxt}[j] = 0.0f; gd{nxt}[j] = 0.0f; }}")
        o.open(f"for (int k = 0; k < {n_out}; ++k) {{  // W{li}^T")
        o.w(f"float gk = g{cur}[k];")
        o.w(f"float gdk = gd{cur}[k];")
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"float w = dual_cbf::W{li}[k * {n_in} + j];")
        o.w(f"g{nxt}[j] += w * gk;")
        o.w(f"gd{nxt}[j] += w * gdk;")
        o.close()
        o.close()
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"float s = g{nxt}[j];")
        o.w(f"float sd = gd{nxt}[j];")
        o.w(f"g{nxt}[j] = P{li}[j] * s;")
        o.w(f"gd{nxt}[j] = Q{li}[j] * s + P{li}[j] * sd;")
        o.close()
        cur, nxt = nxt, cur
    o.w(f"float grad[{n0}];")
    o.w(f"float Hf[{n0}];")
    o.w(f"for (int j = 0; j < {n0}; ++j) {{ grad[j] = 0.0f; Hf[j] = 0.0f; }}")
    o.open(f"for (int k = 0; k < {widths[1]}; ++k) {{  // W0^T")
    o.w(f"float gk = g{cur}[k];")
    o.w(f"float gdk = gd{cur}[k];")
    o.open(f"for (int j = 0; j < {n0}; ++j) {{")
    o.w(f"float w = dual_cbf::W0[k * {n0} + j];")
    o.w("grad[j] += w * gk;")
    o.w("Hf[j] += w * gdk;")
    o.close()
    o.close()
    o.w("float l2 = 0.0f;")
    o.w(f"for (int k = 0; k < {n0}; ++k) l2 += f[k] * Hf[k] + grad[k] * Df_f[k];")
    o.w("*L2f = l2;")
    o.open("for (int j = 0; j < m; ++j) {")
    o.w("float acc = 0.0f;")
    o.w(f"for (int k = 0; k < {n0}; ++k) acc += G[k * m + j] * Hf[k] + grad[k] * Df_G[k * m + j];")
    o.w("LgLf[j] = acc;")
    o.close()
    o.close()
    o.close("}  // namespace rev2_static")
    o.w()


# --------------------------------------------------------------------------
# Double-precision reference (forward-over-reverse in double)
# --------------------------------------------------------------------------
def emit_ref64(o: Out, widths, acts, second: bool):
    d = len(widths) - 1
    hidden = widths[1:-1]
    maxh = max(hidden)
    maxw = max(widths)
    n0 = widths[0]
    o.w("// ---------------------------------------------------------------------")
    o.w("// ref64: reference evaluation in double precision of the network with the")
    o.w("// float parameters of dual_cbf.h.  Returns h, grad h, and H v.")
    o.w("// Not timed; used only to measure the numerical error of each method.")
    o.w("// ---------------------------------------------------------------------")
    o.open("namespace ref64 {")
    for k, n in enumerate(hidden, start=1):
        o.w(f"static double P{k}[{n}];")
        o.w(f"static double Q{k}[{n}];")
    for nm in ("z", "zd", "ta", "td"):
        o.w(f"static double {nm}[{maxw}];")
    for nm in ("gA", "gdA", "gB", "gdB"):
        o.w(f"static double {nm}[{maxh}];")
    o.w()
    o.w("// min_pre: smallest |preactivation| over the ReLU neurons (1e300 if none).")
    o.open("inline void evaluate(const double* x, const double* v,"
           " double* h, double* grad, double* Hv, double* min_pre) {")
    o.w(f"for (int i = 0; i < {n0}; ++i) {{ z[i] = x[i]; zd[i] = v[i]; }}")
    o.w("*min_pre = 1e300;")
    for li in range(d):
        n_in, n_out = widths[li], widths[li + 1]
        o.open(f"for (int i = 0; i < {n_out}; ++i) {{")
        o.w(f"double acc = (double)dual_cbf::b{li}[i];")
        o.w("double accd = 0.0;")
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"double w = (double)dual_cbf::W{li}[i * {n_in} + j];")
        o.w("acc += w * z[j];")
        o.w("accd += w * zd[j];")
        o.close()
        o.w("ta[i] = acc; td[i] = accd;")
        o.close()
        if li < d - 1:
            k = acts[li]
            o.open(f"for (int i = 0; i < {n_out}; ++i) {{")
            o.w("double pre = ta[i];")
            if k == "relu":
                o.w("if (fabs(pre) < *min_pre) *min_pre = fabs(pre);")
            o.w(f"double sig = {act_d(k, 'pre')};")
            o.w(f"double sig_p = {dact_d(k, 'pre', 'sig')};")
            o.w(f"double sig_pp = {ddact_d(k, 'pre', 'sig', 'sig_p')};")
            o.w("z[i] = sig;")
            o.w("zd[i] = sig_p * td[i];")
            o.w(f"P{li + 1}[i] = sig_p;")
            o.w(f"Q{li + 1}[i] = sig_pp * td[i];")
            o.close()
        else:
            o.w("*h = ta[0];")
    k = d - 1
    o.open(f"for (int j = 0; j < {widths[k]}; ++j) {{")
    o.w(f"double s = (double)dual_cbf::W{d - 1}[j];")
    o.w(f"gA[j] = P{k}[j] * s;")
    o.w(f"gdA[j] = Q{k}[j] * s;")
    o.close()
    cur, nxt = "A", "B"
    for li in range(d - 2, 0, -1):
        n_in, n_out = widths[li], widths[li + 1]
        o.w(f"for (int j = 0; j < {n_in}; ++j) {{ g{nxt}[j] = 0.0; gd{nxt}[j] = 0.0; }}")
        o.open(f"for (int k = 0; k < {n_out}; ++k) {{")
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"double w = (double)dual_cbf::W{li}[k * {n_in} + j];")
        o.w(f"g{nxt}[j] += w * g{cur}[k];")
        o.w(f"gd{nxt}[j] += w * gd{cur}[k];")
        o.close()
        o.close()
        o.open(f"for (int j = 0; j < {n_in}; ++j) {{")
        o.w(f"double s = g{nxt}[j];")
        o.w(f"double sd = gd{nxt}[j];")
        o.w(f"g{nxt}[j] = P{li}[j] * s;")
        o.w(f"gd{nxt}[j] = Q{li}[j] * s + P{li}[j] * sd;")
        o.close()
        cur, nxt = nxt, cur
    o.w(f"for (int j = 0; j < {n0}; ++j) {{ grad[j] = 0.0; Hv[j] = 0.0; }}")
    o.open(f"for (int k = 0; k < {widths[1]}; ++k) {{")
    o.open(f"for (int j = 0; j < {n0}; ++j) {{")
    o.w(f"double w = (double)dual_cbf::W0[k * {n0} + j];")
    o.w(f"grad[j] += w * g{cur}[k];")
    o.w(f"Hv[j] += w * gd{cur}[k];")
    o.close()
    o.close()
    o.close()
    o.close("}  // namespace ref64")
    o.w()


def emit_meta(o: Out, widths, acts, second: bool, m: int):
    d = len(widths) - 1
    nparam = sum(widths[i] * widths[i + 1] + widths[i + 1] for i in range(d))
    w_df = 2 * max(widths[i] + widths[i + 1] for i in range(d))
    w_ad = sum(widths[1:-1])
    o.open("namespace meta {")
    o.w(f"static constexpr int DEPTH = {d};")
    o.w(f"static constexpr int N_INPUTS = {m};")
    o.w(f"static constexpr int RELATIVE_DEGREE = {2 if second else 1};")
    o.w(f"static constexpr int N_PARAM = {nparam};          // floats of read-only parameters")
    o.w(f"static constexpr int W_DF_FLOATS = {w_df};      // Proposition 3 ii)")
    o.w(f"static constexpr int W_AD_FLOATS = {w_ad};      // Proposition 3 i), lower bound")
    o.w('static constexpr const char* WIDTHS = "' + "-".join(map(str, widths)) + '";')
    o.w('static constexpr const char* ACTIVATIONS = "' + ",".join(acts) + '";')
    o.w(f"static constexpr int COMPILER_STATIC_FLOATS = {(4 if second else 2) * max(widths)};"
        "         // scratch buffer of dual_cbf.h")
    o.w(f"static constexpr int COMPILER_SHARED_STATIC_FLOATS = {(m + 2) * max(widths)};"
        "  // buffers of evaluate_cbf_shared")
    o.close("}  // namespace meta")
    o.w()
    o.w("// Layer table for generic (loop-based) cross-checks on the host.")
    o.open("namespace net {")
    o.w("struct Layer { const float* W; const float* b; int n_in; int n_out; int act; };")
    o.w("// act: 0 identity, 1 relu, 2 softplus, 3 tanh, 4 sigmoid")
    o.open(f"static const Layer LAYERS[{d}] = {{")
    for li in range(d):
        a = ACT_CODE[acts[li]] if li < d - 1 else 0
        o.w(f"{{dual_cbf::W{li}, dual_cbf::b{li}, {widths[li]}, {widths[li + 1]}, {a}}},")
    o.close("};")
    o.close("}  // namespace net")
    o.w()


def generate(header_text: str, m: int) -> str:
    widths, acts, second = parse_header(header_text)
    if not second and "evaluate_cbf_shared" not in header_text:
        raise SystemExit("compile the header with n_inputs (dual_cbf_compiler >= 0.2.0)")
    o = Out()
    o.w("// Auto-generated by tools/gen_methods.py. Do not edit.")
    o.w(f"// Network {'-'.join(map(str, widths))}, activations {','.join(acts)},"
        f" relative degree {2 if second else 1}, m = {m}.")
    o.w("// Include after dual_cbf.h; all methods read dual_cbf::W*, dual_cbf::b*.")
    o.w("#ifndef DUAL_CBF_METHODS_H")
    o.w("#define DUAL_CBF_METHODS_H")
    o.w()
    o.w("#include <math.h>")
    o.w()
    emit_meta(o, widths, acts, second, m)
    if second:
        emit_jet3(o, widths, acts)
        emit_rev2_static(o, widths, acts)
    else:
        emit_rev_static(o, widths, acts, m)
    emit_ref64(o, widths, acts, second)
    o.w("#endif  // DUAL_CBF_METHODS_H")
    return o.text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("header", type=Path, help="path to the compiled dual_cbf.h")
    ap.add_argument("--m", type=int, required=True, help="number of control inputs")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default: methods.h next to the header)")
    args = ap.parse_args()
    text = generate(args.header.read_text(), args.m)
    out = args.out or args.header.with_name("methods.h")
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
