// methods_rd2.inc -- method wrappers for relative degree two.
// Included once by <example>/main/methods.cpp after example.h.

#include "dual_cbf.h"
#include "methods.h"
#include "reverse_mode_ad.h"
#include "bench_method.h"

#define BENCH_NOINLINE __attribute__((noinline))

namespace bench {

static reverse_mode_ad::LinearSpec s_lin[meta::DEPTH];
static reverse_mode_ad::ActivationSpec s_act[meta::DEPTH];

void methods_init() {
    for (int i = 0; i < meta::DEPTH; ++i) {
        s_lin[i] = {net::LAYERS[i].W, net::LAYERS[i].b, net::LAYERS[i].n_in, net::LAYERS[i].n_out};
        switch (net::LAYERS[i].act) {
            case 2:  s_act[i].kind = reverse_mode_ad::ActKind::Softplus; break;
            case 3:  s_act[i].kind = reverse_mode_ad::ActKind::Tanh;     break;
            default: s_act[i].kind = reverse_mode_ad::ActKind::ReLU;     break;
        }
    }
}

BENCH_NOINLINE void m_empty(const ex::Case&, ex::Out&) {}

// Compiler output (four-component hyper-dual), m+1 passes.  h and Lf are the
// real and eps_1 components of the same passes.
BENCH_NOINLINE void m_hyperdual(const ex::Case& c, ex::Out& o) {
    dual_cbf::evaluate_hocbf(c.x, c.f, c.G, c.Df_f, c.Df_G, ex::M,
                             &o.q[0], &o.q[1], &o.q[2], &o.q[3]);
}

// Three-component algebra, 1 + 2m passes (polarization for the mixed terms).
BENCH_NOINLINE void m_jet3(const ex::Case& c, ex::Out& o) {
    jet3::evaluate_cbf_2nd_order(c.x, c.f, c.G, c.Df_f, c.Df_G, ex::M,
                                 &o.q[0], &o.q[1], &o.q[2], &o.q[3]);
}

// Exact second order by forward-over-reverse, static buffers.
BENCH_NOINLINE void m_rev2_static(const ex::Case& c, ex::Out& o) {
    rev2_static::evaluate_cbf_2nd_order(c.x, c.f, c.G, c.Df_f, c.Df_G, ex::M,
                                        &o.q[0], &o.q[1], &o.q[2], &o.q[3]);
}

// Reverse-mode gradients (heap cache) with one-sided finite differences for
// the Hessian-vector products, as in the originally submitted comparison.
BENCH_NOINLINE void m_rev_fd(const ex::Case& c, ex::Out& o) {
    constexpr float h_fd = 1e-3f;
    constexpr int n = ex::N;
    constexpr int m = ex::M;
    float h0 = 0.0f;
    auto get_grad = [&](const float* x_eval, float* grad_out, float* h_out) {
        reverse_mode_ad::ActivationCache cache =
            reverse_mode_ad::ActivationCache::allocate(s_lin, meta::DEPTH);
        float hv = reverse_mode_ad::forward_with_cache(s_lin, s_act, meta::DEPTH, x_eval, cache);
        if (h_out) *h_out = hv;
        reverse_mode_ad::backward(s_lin, s_act, meta::DEPTH, cache, grad_out);
        cache.free();
    };
    float grad0[n], gradp[n], xp[n];
    get_grad(c.x, grad0, &h0);
    for (int k = 0; k < n; ++k) xp[k] = c.x[k] + h_fd * c.f[k];
    get_grad(xp, gradp, nullptr);
    float lf = 0.0f, l2 = 0.0f;
    for (int k = 0; k < n; ++k) {
        lf += grad0[k] * c.f[k];
        l2 += ((gradp[k] - grad0[k]) / h_fd) * c.f[k] + grad0[k] * c.Df_f[k];
    }
    o.q[0] = h0;
    o.q[1] = lf;
    o.q[2] = l2;
    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < n; ++k) xp[k] = c.x[k] + h_fd * c.G[k * m + j];
        get_grad(xp, gradp, nullptr);
        float acc = 0.0f;
        for (int k = 0; k < n; ++k)
            acc += ((gradp[k] - grad0[k]) / h_fd) * c.f[k] + grad0[k] * c.Df_G[k * m + j];
        o.q[3 + j] = acc;
    }
}

const Method METHODS[] = {
    {"hyperdual",   m_hyperdual,   4 * meta::COMPILER_STATIC_FLOATS, true},
    {"jet3",        m_jet3,        4 * jet3::STATIC_FLOATS,          true},
    {"rev2_static", m_rev2_static, 4 * rev2_static::STATIC_FLOATS,   true},
    {"rev_fd",      m_rev_fd,      0,                                false},
};
const int N_METHODS = sizeof(METHODS) / sizeof(METHODS[0]);

const NetInfo NET = {meta::WIDTHS, meta::ACTIVATIONS, meta::RELATIVE_DEGREE, meta::N_INPUTS,
                     4 * meta::N_PARAM, 4 * meta::W_DF_FLOATS, 4 * meta::W_AD_FLOATS};

void ref_evaluate(const double* x, const double* v,
                  double* h, double* grad, double* Hv, double* min_pre) {
    ref64::evaluate(x, v, h, grad, Hv, min_pre);
}

}  // namespace bench
