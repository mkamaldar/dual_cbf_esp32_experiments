// bench_types.h -- types shared by the benchmark driver and the method wrappers.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bench {

// Seeded generator, so that all methods see the same states and a run can be
// repeated bit for bit.  (The earlier driver drew states from esp_random().)
struct Prng {
    uint32_t s;
    explicit Prng(uint32_t seed) : s(seed ? seed : 0x9E3779B9u) {}
    inline uint32_t next() {  // xorshift32
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    inline float uniform(float lo, float hi) {
        float u = static_cast<float>(next() >> 8) / 16777216.0f;  // 24 bits in [0,1)
        return lo + u * (hi - lo);
    }
};

// Exact solution of
//     min ||u - un||^2   s.t.   a.u >= b,   lo <= u <= hi
// by a breakpoint search on the multiplier of the affine constraint.  The
// minimizer is u(lam) = clip(un + lam a), and a.u(lam) is nondecreasing and
// piecewise linear in lam with at most 2M breakpoints.  No iteration, no
// allocation.  Returns false if the constraint cannot be met inside the box;
// u is then the box point that maximizes a.u.
template <int M>
inline bool qp_halfspace_box(const float* un, const float* a, float b,
                             const float* lo, const float* hi, float* u)
{
    auto clip = [&](int i, float v) { return v < lo[i] ? lo[i] : (v > hi[i] ? hi[i] : v); };
    float phi0 = 0.0f;
    for (int i = 0; i < M; ++i) { u[i] = clip(i, un[i]); phi0 += a[i] * u[i]; }
    if (phi0 >= b) return true;

    float bp[2 * M];
    int nb = 0;
    for (int i = 0; i < M; ++i) {
        if (a[i] > 0.0f) {
            if (un[i] < lo[i]) bp[nb++] = (lo[i] - un[i]) / a[i];
            if (un[i] < hi[i]) bp[nb++] = (hi[i] - un[i]) / a[i];
        } else if (a[i] < 0.0f) {
            if (un[i] > hi[i]) bp[nb++] = (hi[i] - un[i]) / a[i];
            if (un[i] > lo[i]) bp[nb++] = (lo[i] - un[i]) / a[i];
        }
    }
    for (int i = 1; i < nb; ++i) {  // insertion sort, nb <= 2M
        float key = bp[i];
        int j = i - 1;
        while (j >= 0 && bp[j] > key) { bp[j + 1] = bp[j]; --j; }
        bp[j + 1] = key;
    }
    float lam0 = 0.0f;
    for (int k = 0; k < nb; ++k) {
        float lam1 = bp[k];
        float phi1 = 0.0f;
        for (int i = 0; i < M; ++i) phi1 += a[i] * clip(i, un[i] + lam1 * a[i]);
        if (phi1 >= b) {
            float lam = lam0 + (b - phi0) * (lam1 - lam0) / (phi1 - phi0);
            for (int i = 0; i < M; ++i) u[i] = clip(i, un[i] + lam * a[i]);
            return true;
        }
        lam0 = lam1;
        phi0 = phi1;
    }
    for (int i = 0; i < M; ++i) u[i] = clip(i, un[i] + lam0 * a[i]);
    return false;
}

}  // namespace bench
