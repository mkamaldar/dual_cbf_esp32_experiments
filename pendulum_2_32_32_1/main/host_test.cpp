// host_test.cpp -- Pendulum hyper-dual host regression test.
//
// Verifies that:
//   (a) dual_cbf::evaluate_cbf_2nd_order  (hyper-dual exact)
//   (b) reverse-mode + finite-difference HVP baseline
// agree on a batch of random pendulum states. The reverse-mode side uses
// finite differences to estimate the Hessian-vector products, so we use a
// loose tolerance and require agreement on a *fraction* of samples
// (some saddle-point neighborhoods produce noisy FD estimates).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>

static inline uint32_t esp_random() { return (uint32_t)rand(); }
extern "C" uint32_t esp_cpu_get_cycle_count() {
    using namespace std::chrono;
    static auto t0 = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    auto ns = duration_cast<nanoseconds>(now - t0).count();
    return (uint32_t)((ns * 240) / 1000);
}
static inline void vTaskDelay(uint32_t) {}
#define pdMS_TO_TICKS(ms) ((uint32_t)(ms))

#include "../../shared/timer.h"
#include "../../shared/reverse_mode_ad.h"
#include "dual_cbf.h"

static constexpr float G_GRAVITY = 9.81f;
static constexpr float L_LENGTH  = 1.0f;
static constexpr float M_MASS    = 1.0f;

static const reverse_mode_ad::LinearSpec linear_specs[] = {
    {dual_cbf::W0, dual_cbf::b0, 2,  32},
    {dual_cbf::W1, dual_cbf::b1, 32, 32},
    {dual_cbf::W2, dual_cbf::b2, 32, 1},
};
static const reverse_mode_ad::ActivationSpec activations[] = {
    {reverse_mode_ad::ActKind::Softplus},
    {reverse_mode_ad::ActKind::Softplus},
};

static void rvad_evaluate_2nd_order_fd(
    const float* x, const float* f, const float* G,
    const float* Df_f, const float* Df_G,
    int m, float* L2f_out, float* LgLf_out)
{
    constexpr float h_fd = 1e-3f;
    int n = linear_specs[0].n_in;

    auto get_grad = [&](const float* x_eval, float* grad_out) {
        auto cache = reverse_mode_ad::ActivationCache::allocate(linear_specs, 3);
        reverse_mode_ad::forward_with_cache(linear_specs, activations, 3, x_eval, cache);
        reverse_mode_ad::backward(linear_specs, activations, 3, cache, grad_out);
        cache.free();
    };

    float grad0[2], grad_pf[2], grad_pG[2];
    get_grad(x, grad0);

    float xp[2];
    for (int k = 0; k < n; ++k) xp[k] = x[k] + h_fd * f[k];
    get_grad(xp, grad_pf);

    float Hf_dot_f = 0.0f;
    for (int k = 0; k < n; ++k) Hf_dot_f += ((grad_pf[k] - grad0[k]) / h_fd) * f[k];
    float grad_dot_Dff = 0.0f;
    for (int k = 0; k < n; ++k) grad_dot_Dff += grad0[k] * Df_f[k];
    *L2f_out = Hf_dot_f + grad_dot_Dff;

    for (int j = 0; j < m; ++j) {
        float Gj[2];
        for (int k = 0; k < n; ++k) Gj[k] = G[k * m + j];
        for (int k = 0; k < n; ++k) xp[k] = x[k] + h_fd * Gj[k];
        get_grad(xp, grad_pG);

        float HG_dot_f = 0.0f;
        for (int k = 0; k < n; ++k) HG_dot_f += ((grad_pG[k] - grad0[k]) / h_fd) * f[k];
        float grad_dot_DfG = 0.0f;
        for (int k = 0; k < n; ++k) grad_dot_DfG += grad0[k] * Df_G[k * m + j];
        LgLf_out[j] = HG_dot_f + grad_dot_DfG;
    }
}

static void sample_pendulum_state(
    float x[2], float f[2], float G[2*1],
    float Df_f[2], float Df_G[2*1])
{
    x[0] = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;   // theta in [-1.5, 1.5]
    x[1] = ((float)rand() / RAND_MAX - 0.5f) * 6.0f;   // theta_dot in [-3, 3]

    f[0] = x[1];
    f[1] = (G_GRAVITY / L_LENGTH) * sinf(x[0]);

    G[0*1 + 0] = 0.0f;
    G[1*1 + 0] = 1.0f / (M_MASS * L_LENGTH * L_LENGTH);

    float c = (G_GRAVITY / L_LENGTH) * cosf(x[0]);
    Df_f[0] = f[1];
    Df_f[1] = c * f[0];
    Df_G[0*1 + 0] = G[1*1 + 0];
    Df_G[1*1 + 0] = c * G[0*1 + 0];
}

int main() {
    printf("Pendulum hyper-dual host regression run.\n");

    constexpr int N_CHECK = 64;
    int agree = 0;
    float worst_rel = 0.0f;
    for (int t = 0; t < N_CHECK; ++t) {
        float x[2], f[2], G[2*1], Df_f[2], Df_G[2*1];
        sample_pendulum_state(x, f, G, Df_f, Df_G);

        float L2f_d, LgLf_d[1];
        float L2f_r, LgLf_r[1];
        dual_cbf::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, 1, &L2f_d, LgLf_d);
        rvad_evaluate_2nd_order_fd(x, f, G, Df_f, Df_G, 1, &L2f_r, LgLf_r);

        float scale1 = fmaxf(fabsf(L2f_d), 1.0f);
        float scale2 = fmaxf(fabsf(LgLf_d[0]), 1.0f);
        float rel1 = fabsf(L2f_d - L2f_r) / scale1;
        float rel2 = fabsf(LgLf_d[0] - LgLf_r[0]) / scale2;
        worst_rel = fmaxf(worst_rel, fmaxf(rel1, rel2));

        if (rel1 < 5e-2f && rel2 < 5e-2f) agree++;
    }
    printf("Agreement: %d / %d  (worst relative error %.3e)\n",
           agree, N_CHECK, worst_rel);
    printf("Note: rev-mode uses FD Hessian-vector product, so 5%% rel tol.\n");
    if (agree < (3 * N_CHECK) / 4) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");

    // Bench
    constexpr int N = 500;
    auto* dual_cycles = new uint32_t[N];
    auto* rvad_cycles = new uint32_t[N];
    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N; ++i) {
        float x[2], f[2], G[2*1], Df_f[2], Df_G[2*1];
        sample_pendulum_state(x, f, G, Df_f, Df_G);
        float L2f, LgLf[1];

        uint32_t t0 = bench::cycles_now();
        dual_cbf::evaluate_cbf_2nd_order(x, f, G, Df_f, Df_G, 1, &L2f, LgLf);
        uint32_t t1 = bench::cycles_now();
        dual_cycles[i] = t1 - t0;

        t0 = bench::cycles_now();
        rvad_evaluate_2nd_order_fd(x, f, G, Df_f, Df_G, 1, &L2f, LgLf);
        t1 = bench::cycles_now();
        rvad_cycles[i] = t1 - t0;
    }
    auto sd = bench::summarize(dual_cycles, N);
    auto sr = bench::summarize(rvad_cycles, N);
    printf("[host bench, NOT MCU]\n  dual hyper:  med=%.2f us\n  rvad FD:     med=%.2f us  peak heap=%zu B\n",
           sd.med_us, sr.med_us, reverse_mode_ad::peak_dynamic_bytes());
    delete[] dual_cycles;
    delete[] rvad_cycles;
    return 0;
}
