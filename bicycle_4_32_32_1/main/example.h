// example.h -- kinematic bicycle (Example 1): ReLU 4-32-32-1, m = 2, relative degree 1.
#pragma once

#include <cmath>

#include "bench_types.h"

namespace ex {

constexpr const char* NAME = "BICYCLE";
constexpr int N = 4;    // state dimension
constexpr int M = 2;    // inputs: u = [a, tan(delta)]
constexpr int RD = 1;   // relative degree
constexpr int Q = 2 + M;
constexpr const char* QNAME[Q] = {"h", "Lf", "Lg1", "Lg2"};

struct Case {
    float x[N];
    float f[N];
    float G[N * M];  // row-major: G(i,j) at G[i*M + j]
};
struct Out {
    float q[Q];      // h, Lf, Lg1, Lg2
};

constexpr float WHEELBASE = 2.5f;
constexpr float ALPHA = 1.0f;                       // alpha(h) = ALPHA * h
constexpr float U_LO[M] = {-3.0f, -0.5463025f};     // tan(0.5 rad) = 0.5463025
constexpr float U_HI[M] = { 3.0f,  0.5463025f};
constexpr float U_NOM[M] = {0.5f, 0.1003347f};      // tan(0.1 rad)

// Sampling box of the benchmark; equal to the training box of 01_train_models.py.
inline void sample(bench::Prng& r, float* x) {
    x[0] = r.uniform(-7.0f, 7.0f);      // p_x (m)
    x[1] = r.uniform(-5.0f, 5.0f);      // p_y (m)
    x[2] = r.uniform(-0.35f, 0.35f);    // psi (rad)
    x[3] = r.uniform(4.0f, 12.0f);      // v (m/s)
}

inline void dynamics(const float* x, Case& c) {
    for (int i = 0; i < N; ++i) c.x[i] = x[i];
    c.f[0] = x[3] * cosf(x[2]);
    c.f[1] = x[3] * sinf(x[2]);
    c.f[2] = 0.0f;
    c.f[3] = 0.0f;
    c.G[0 * M + 0] = 0.0f;  c.G[0 * M + 1] = 0.0f;
    c.G[1 * M + 0] = 0.0f;  c.G[1 * M + 1] = 0.0f;
    c.G[2 * M + 0] = 0.0f;  c.G[2 * M + 1] = x[3] / WHEELBASE;
    c.G[3 * M + 0] = 1.0f;  c.G[3 * M + 1] = 0.0f;
}

// Constraint Lf + Lg u >= -alpha(h), then the QP (7) over the box.
inline bool filter(const Case&, const Out& o, float* u) {
    const float a[M] = {o.q[2], o.q[3]};
    const float b = -ALPHA * o.q[0] - o.q[1];
    return bench::qp_halfspace_box<M>(U_NOM, a, b, U_LO, U_HI, u);
}

}  // namespace ex
