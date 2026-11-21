// example.h -- depth sweep: Van der Pol dynamics and target, ReLU, width 32, 4 hidden layers.
#pragma once

#include <cmath>

#include "bench_types.h"

namespace ex {

constexpr const char* NAME = "DEPTH4";
constexpr int N = 2;
constexpr int M = 1;
constexpr int RD = 1;
constexpr int Q = 2 + M;
constexpr const char* QNAME[Q] = {"h", "Lf", "Lg1"};

struct Case {
    float x[N];
    float f[N];
    float G[N * M];
};
struct Out {
    float q[Q];      // h, Lf, Lg1
};

constexpr float MU = 1.0f;
constexpr float ALPHA = 1.0f;
constexpr float U_LO[M] = {-5.0f};
constexpr float U_HI[M] = { 5.0f};
constexpr float U_NOM[M] = {0.0f};

inline void sample(bench::Prng& r, float* x) {
    x[0] = r.uniform(-4.0f, 4.0f);
    x[1] = r.uniform(-3.5f, 3.5f);
}

inline void dynamics(const float* x, Case& c) {
    c.x[0] = x[0];
    c.x[1] = x[1];
    c.f[0] = x[1];
    c.f[1] = -x[0] + MU * (1.0f - x[0] * x[0]) * x[1];
    c.G[0] = 0.0f;
    c.G[1] = 1.0f;
}

inline bool filter(const Case&, const Out& o, float* u) {
    const float a[M] = {o.q[2]};
    const float b = -ALPHA * o.q[0] - o.q[1];
    return bench::qp_halfspace_box<M>(U_NOM, a, b, U_LO, U_HI, u);
}

}  // namespace ex
