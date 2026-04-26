// main.cpp -- Inverted pendulum 2-32-32-1 (Softplus) benchmark on the ESP32-S3.
//
// This example exercises the relative-degree-two extension: dual_cbf provides
// evaluate_cbf_2nd_order (hyper-dual), which extracts L_f^2 h and L_G L_f h
// in a single forward pass per direction.
//
// Reverse-mode baseline for second-order Lie derivatives:
//   We use finite-difference Hessian-vector products on top of single
//   reverse-mode backprop. This matches what an embedded user without
//   double-backprop tooling would actually do, and avoids implementing a
//   tape-of-tapes that would not fit on an MCU. The trade-off is that
//   finite differences introduce O(h) truncation error -- not exact.
//   This is acknowledged in the paper text accompanying Example 3.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_random.h"

#include "dual_cbf.h"
#include "reverse_mode_ad.h"
#include "timer.h"

static constexpr int N_ITERS = 1000;

static uint32_t dual_cycles[N_ITERS];
static uint32_t rvad_cycles[N_ITERS];

// ---------------------------------------------------------------------------
// Adapter: same const arrays as the dual compiler. Pendulum uses softplus.
// ---------------------------------------------------------------------------

static const reverse_mode_ad::LinearSpec linear_specs[] = {
    {dual_cbf::W0, dual_cbf::b0, 2,  32},
    {dual_cbf::W1, dual_cbf::b1, 32, 32},
    {dual_cbf::W2, dual_cbf::b2, 32, 1},
};
static constexpr int DEPTH = 3;

static const reverse_mode_ad::ActivationSpec activations[] = {
    {reverse_mode_ad::ActKind::Softplus},
    {reverse_mode_ad::ActKind::Softplus},
};

// ---------------------------------------------------------------------------
// Pendulum state and dynamics:
//   state  = (theta, theta_dot)
//   f(x)   = (theta_dot, (g/L)*sin(theta))
//   G(x)   = (0, 1/(m L^2))
//   f'(x)  = [[0, 1], [(g/L)*cos(theta), 0]]
// ---------------------------------------------------------------------------

static constexpr float G_GRAVITY = 9.81f;
static constexpr float L_LENGTH  = 1.0f;
static constexpr float M_MASS    = 1.0f;

static inline float urand(float lo, float hi) {
    uint32_t r = esp_random();
    float u = static_cast<float>(r & 0x00FFFFFF) / static_cast<float>(0x01000000);
    return lo + u * (hi - lo);
}

static void sample_pendulum_state(
    float x[2], float f[2], float G[2 * 1],
    float Df_f[2], float Df_G[2 * 1])
{
    x[0] = urand(-1.5f, 1.5f);
    x[1] = urand(-3.0f, 3.0f);

    f[0] = x[1];
    f[1] = (G_GRAVITY / L_LENGTH) * sinf(x[0]);

    // G col 0 = [0, 1/(m L^2)]^T
    G[0*1 + 0] = 0.0f;
    G[1*1 + 0] = 1.0f / (M_MASS * L_LENGTH * L_LENGTH);

    // Df = [[0, 1], [(g/L) cos(theta), 0]]
    // Df_f = Df * f
    float c = (G_GRAVITY / L_LENGTH) * cosf(x[0]);
    Df_f[0] = f[1];
    Df_f[1] = c * f[0];
    // Df_G = Df * G  (only one input column)
    Df_G[0*1 + 0] = G[1*1 + 0];     // row 0 . G column = G[1]
    Df_G[1*1 + 0] = c * G[0*1 + 0]; // row 1 . G column = c * G[0] = 0
}

// ---------------------------------------------------------------------------
// Reverse-mode baseline for relative-degree-two. We implement a single
// backward pass to get grad h, then finite-difference grad h to estimate
// the Hessian-vector products needed for L_f^2 h and L_G L_f h.
// ---------------------------------------------------------------------------

static void rvad_evaluate_2nd_order(
    const float* x, const float* f, const float* G,
    const float* Df_f, const float* Df_G,
    int m, float* L2f_out, float* LgLf_out)
{
    constexpr float h_fd = 1e-3f;
    int n = linear_specs[0].n_in;

    // Get grad h(x), grad h(x + h * f), grad h(x + h * G[:,j]) for each j.
    // Reuse evaluate_cbf to get h, Lf, Lg, plus we also need the bare
    // gradient. Cleanest: run forward+backward by hand.
    auto get_grad = [&](const float* x_eval, float* grad_out) {
        reverse_mode_ad::ActivationCache cache =
            reverse_mode_ad::ActivationCache::allocate(linear_specs, DEPTH);
        reverse_mode_ad::forward_with_cache(
            linear_specs, activations, DEPTH, x_eval, cache);
        reverse_mode_ad::backward(
            linear_specs, activations, DEPTH, cache, grad_out);
        cache.free();
    };

    float grad0[2], grad_pf[2], grad_pG[2];
    get_grad(x, grad0);

    float xp[2];
    for (int k = 0; k < n; ++k) xp[k] = x[k] + h_fd * f[k];
    get_grad(xp, grad_pf);

    // L_f^2 h = (Hessian * f) . f + grad . Df_f
    //        ~ ((grad_pf - grad0) / h) . f + grad . Df_f
    float Hf_dot_f = 0.0f;
    for (int k = 0; k < n; ++k) {
        float Hfk = (grad_pf[k] - grad0[k]) / h_fd;
        Hf_dot_f += Hfk * f[k];
    }
    float grad_dot_Dff = 0.0f;
    for (int k = 0; k < n; ++k) grad_dot_Dff += grad0[k] * Df_f[k];
    *L2f_out = Hf_dot_f + grad_dot_Dff;

    // For each input j: L_{G_j} L_f h = (H * f) . G_j + grad . Df_G_j
    for (int j = 0; j < m; ++j) {
        float Gj[2];
        for (int k = 0; k < n; ++k) Gj[k] = G[k * m + j];
        for (int k = 0; k < n; ++k) xp[k] = x[k] + h_fd * Gj[k];
        get_grad(xp, grad_pG);

        float HG_dot_f = 0.0f;
        for (int k = 0; k < n; ++k) {
            float HGk = (grad_pG[k] - grad0[k]) / h_fd;
            HG_dot_f += HGk * f[k];
        }
        float grad_dot_DfG = 0.0f;
        for (int k = 0; k < n; ++k) grad_dot_DfG += grad0[k] * Df_G[k * m + j];
        LgLf_out[j] = HG_dot_f + grad_dot_DfG;
    }
}

// ---------------------------------------------------------------------------
// Sanity check
// ---------------------------------------------------------------------------

static bool sanity_check_agreement() {
    float x[2], f[2], G[2*1], Df_f[2], Df_G[2*1];
    float L2f_d, LgLf_d[1];
    float L2f_r, LgLf_r[1];
    constexpr int N_CHECK = 16;
    constexpr float TOL = 5e-2f;  // looser: rev-mode uses finite differences

    int agree = 0;
    for (int t = 0; t < N_CHECK; ++t) {
        sample_pendulum_state(x, f, G, Df_f, Df_G);
        dual_cbf::evaluate_cbf_2nd_order(
            x, f, G, Df_f, Df_G, 1, &L2f_d, LgLf_d);
        rvad_evaluate_2nd_order(
            x, f, G, Df_f, Df_G, 1, &L2f_r, LgLf_r);

        // Use relative tolerance for L2f (it can be small near equilibrium)
        float scale = fmaxf(fabsf(L2f_d), 1.0f);
        if (fabsf(L2f_d - L2f_r) / scale < TOL &&
            fabsf(LgLf_d[0] - LgLf_r[0]) / fmaxf(fabsf(LgLf_d[0]), 1.0f) < TOL) {
            agree++;
        }
    }
    // Allow a few outliers since finite differences can be noisy near
    // saddle points; require >= 75% agreement.
    printf("Agreement: %d / %d (rev-mode is finite-difference for 2nd order)\n",
           agree, N_CHECK);
    return agree >= 12;
}

static inline uint32_t bench_one_dual(
    const float* x, const float* f, const float* G,
    const float* Df_f, const float* Df_G,
    float* L2f, float* LgLf)
{
    uint32_t t0 = bench::cycles_now();
    dual_cbf::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, 1, L2f, LgLf);
    uint32_t t1 = bench::cycles_now();
    return t1 - t0;
}

static inline uint32_t bench_one_rvad(
    const float* x, const float* f, const float* G,
    const float* Df_f, const float* Df_G,
    float* L2f, float* LgLf)
{
    uint32_t t0 = bench::cycles_now();
    rvad_evaluate_2nd_order(x, f, G, Df_f, Df_G, 1, L2f, LgLf);
    uint32_t t1 = bench::cycles_now();
    return t1 - t0;
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));

    printf("\n\n========================================\n");
    printf("Pendulum 2-32-32-1 (Softplus, hyper-dual) benchmark\n");
    printf("CPU = %lu Hz, N_ITERS = %d\n", (unsigned long)bench::CPU_HZ, N_ITERS);
    printf("========================================\n");

    if (!sanity_check_agreement()) {
        printf("WARNING: dual and reverse-mode (FD) disagreed too often.\n");
        printf("Reporting timings anyway, but inspect the implementation.\n");
    }

    float x[2], f[2], G[2 * 1], Df_f[2], Df_G[2 * 1];
    float L2f, LgLf[1];

    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        sample_pendulum_state(x, f, G, Df_f, Df_G);
        dual_cycles[i] = bench_one_dual(x, f, G, Df_f, Df_G, &L2f, LgLf);
    }
    auto stats_dual = bench::summarize(dual_cycles, N_ITERS);
    size_t dual_dyn = reverse_mode_ad::peak_dynamic_bytes();

    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        sample_pendulum_state(x, f, G, Df_f, Df_G);
        rvad_cycles[i] = bench_one_rvad(x, f, G, Df_f, Df_G, &L2f, LgLf);
    }
    auto stats_rvad = bench::summarize(rvad_cycles, N_ITERS);
    size_t rvad_dyn = reverse_mode_ad::peak_dynamic_bytes();

    printf("\n----- RESULTS (CSV, parsed by host) -----\n");
    printf("DUAL_PEND,%.3f,%.3f,%.3f,%u\n",
           stats_dual.min_us, stats_dual.med_us, stats_dual.max_us,
           (unsigned)dual_dyn);
    printf("RVAD_PEND,%.3f,%.3f,%.3f,%u\n",
           stats_rvad.min_us, stats_rvad.med_us, stats_rvad.max_us,
           (unsigned)rvad_dyn);
    printf("----- END RESULTS -----\n\n");

    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
