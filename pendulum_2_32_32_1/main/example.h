// example.h -- inverted pendulum (Example 3): softplus 2-32-32-1, m = 1, relative degree 2.
#pragma once

#include <cmath>

#include "bench_types.h"

namespace ex {

constexpr const char* NAME = "PENDULUM";
constexpr int N = 2;    // x = [phi, phi_dot]
constexpr int M = 1;    // torque
constexpr int RD = 2;
constexpr int Q = 3 + M;
constexpr const char* QNAME[Q] = {"h", "Lf", "L2f", "LgLf1"};

struct Case {
    float x[N];
    float f[N];
    float G[N * M];
    float Df_f[N];       // f'(x) f(x)
    float Df_G[N * M];   // f'(x) G(x)
};
struct Out {
    float q[Q];          // h, Lf, L2f, LgLf1
};

constexpr float GRAVITY = 9.81f;
constexpr float LENGTH = 1.0f;
constexpr float MASS = 1.0f;
// alpha_1(s) = K1 s, alpha_2(s) = K2 s.
constexpr float K1 = 3.0f;
constexpr float K2 = 6.0f;
constexpr float U_LO[M] = {-20.0f};
constexpr float U_HI[M] = { 20.0f};
constexpr float U_NOM[M] = {0.0f};

inline void sample(bench::Prng& r, float* x) {
    x[0] = r.uniform(-1.5f, 1.5f);
    x[1] = r.uniform(-3.0f, 3.0f);
}

inline void dynamics(const float* x, Case& c) {
    const float gl = GRAVITY / LENGTH;
    const float b = 1.0f / (MASS * LENGTH * LENGTH);
    c.x[0] = x[0];
    c.x[1] = x[1];
    c.f[0] = x[1];
    c.f[1] = gl * sinf(x[0]);
    c.G[0] = 0.0f;
    c.G[1] = b;
    // f'(x) = [[0, 1], [gl cos(phi), 0]]
    const float cphi = gl * cosf(x[0]);
    c.Df_f[0] = c.f[1];
    c.Df_f[1] = cphi * c.f[0];
    c.Df_G[0] = c.G[1];
    c.Df_G[1] = cphi * c.G[0];
}

// High-order CBF constraint with psi_1 = Lf h + alpha_1(h):
//   L2f + LgLf u + alpha_1'(h) Lf + alpha_2(Lf + alpha_1(h)) >= 0.
inline bool filter(const Case&, const Out& o, float* u) {
    const float h = o.q[0], Lf = o.q[1], L2f = o.q[2];
    const float a[M] = {o.q[3]};
    const float b = -(L2f + K1 * Lf + K2 * (Lf + K1 * h));
    return bench::qp_halfspace_box<M>(U_NOM, a, b, U_LO, U_HI, u);
}

}  // namespace ex
