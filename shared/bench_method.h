// bench_method.h -- method table.  Include after the example's example.h.
#pragma once

#include <cstdint>

namespace bench {

struct Method {
    const char* name;
    void (*fn)(const ex::Case&, ex::Out&);
    int static_bytes;   // mutable static storage owned by the method (bytes)
    bool exact;         // false for the finite-difference baseline
};

extern const Method METHODS[];
extern const int N_METHODS;

void methods_init();
void m_empty(const ex::Case&, ex::Out&);   // call overhead only

// Network description, taken from the generated methods.h.
struct NetInfo {
    const char* widths;
    const char* activations;
    int relative_degree;
    int n_inputs;
    int parameter_bytes;   // read-only parameters (identical for all methods)
    int w_df_bytes;        // Proposition 3 ii)
    int w_ad_bytes;        // Proposition 3 i), lower bound
};
extern const NetInfo NET;

// Double-precision reference: h, grad h, H v, and the smallest
// |preactivation| over the ReLU neurons (1e300 if there are none).
void ref_evaluate(const double* x, const double* v,
                  double* h, double* grad, double* Hv, double* min_pre);

}  // namespace bench
