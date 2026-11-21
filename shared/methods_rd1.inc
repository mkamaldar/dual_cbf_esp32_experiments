// methods_rd1.inc -- method wrappers for relative degree one.
// Included once by <example>/main/methods.cpp after example.h.
//
// Each wrapper is a non-inlined function with the same signature, so that all
// methods are timed through the same call and each has its own symbol in the
// ELF file (tools/size_report.py reads the code size from these symbols).

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

// Compiler output, sequential entry point: m+1 directional passes that reuse
// one scratch buffer (workspace independent of m).
BENCH_NOINLINE void m_dual_seq(const ex::Case& c, ex::Out& o) {
    dual_cbf::evaluate_cbf(c.x, c.f, c.G, ex::M, &o.q[0], &o.q[1], &o.q[2]);
}

// Compiler output, shared-primal entry point: one sweep, primal once, m+1
// tangents (workspace grows with m).
BENCH_NOINLINE void m_dual_shared(const ex::Case& c, ex::Out& o) {
    dual_cbf::evaluate_cbf_shared(c.x, c.f, c.G, &o.q[0], &o.q[1], &o.q[2]);
}

// Reverse mode, ahead of time, static buffers.
BENCH_NOINLINE void m_rev_static(const ex::Case& c, ex::Out& o) {
    rev_static::evaluate_cbf(c.x, c.f, c.G, &o.q[0], &o.q[1], &o.q[2]);
}

// Reverse mode, runtime style: cache allocated from the heap at each call.
BENCH_NOINLINE void m_rev_heap(const ex::Case& c, ex::Out& o) {
    reverse_mode_ad::evaluate_cbf(s_lin, s_act, meta::DEPTH, c.x, c.f, c.G, ex::M,
                                  &o.q[0], &o.q[1], &o.q[2]);
}

const Method METHODS[] = {
    {"dual_seq",    m_dual_seq,    4 * meta::COMPILER_STATIC_FLOATS,        true},
    {"dual_shared", m_dual_shared, 4 * meta::COMPILER_SHARED_STATIC_FLOATS, true},
    {"rev_static",  m_rev_static,  4 * rev_static::STATIC_FLOATS,           true},
    {"rev_heap",    m_rev_heap,    0,                                       true},
};
const int N_METHODS = sizeof(METHODS) / sizeof(METHODS[0]);

const NetInfo NET = {meta::WIDTHS, meta::ACTIVATIONS, meta::RELATIVE_DEGREE, meta::N_INPUTS,
                     4 * meta::N_PARAM, 4 * meta::W_DF_FLOATS, 4 * meta::W_AD_FLOATS};

void ref_evaluate(const double* x, const double* v,
                  double* h, double* grad, double* Hv, double* min_pre) {
    ref64::evaluate(x, v, h, grad, Hv, min_pre);
}

}  // namespace bench
