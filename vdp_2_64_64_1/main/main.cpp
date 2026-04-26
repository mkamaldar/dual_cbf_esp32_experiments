// main.cpp -- Van der Pol 2-64-64-1 benchmark on the ESP32-S3.

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
// Adapter: point reverse-mode AD at the same const arrays the dual compiler
// embedded in dual_cbf.h.
// ---------------------------------------------------------------------------

static const reverse_mode_ad::LinearSpec linear_specs[] = {
    {dual_cbf::W0, dual_cbf::b0, 2,  64},
    {dual_cbf::W1, dual_cbf::b1, 64, 64},
    {dual_cbf::W2, dual_cbf::b2, 64, 1},
};
static constexpr int DEPTH = 3;

static const reverse_mode_ad::ActivationSpec activations[] = {
    {reverse_mode_ad::ActKind::ReLU},
    {reverse_mode_ad::ActKind::ReLU},
};

// ---------------------------------------------------------------------------
// State sampling: realistic Van der Pol states (mu = 1, x in [-4, 4]^2)
// ---------------------------------------------------------------------------

static inline float urand(float lo, float hi) {
    uint32_t r = esp_random();
    float u = static_cast<float>(r & 0x00FFFFFF) / static_cast<float>(0x01000000);
    return lo + u * (hi - lo);
}

static void sample_vdp_state(float x[2], float f[2], float G[2 * 1]) {
    x[0] = urand(-4.0f, 4.0f);
    x[1] = urand(-3.5f, 3.5f);

    // Van der Pol drift: f1 = x2, f2 = -x1 + mu*(1 - x1^2)*x2 with mu = 1
    constexpr float mu = 1.0f;
    f[0] = x[1];
    f[1] = -x[0] + mu * (1.0f - x[0] * x[0]) * x[1];

    // G is row-major (n=2) x (m=1):
    //   col 0 (single input): [0, 1]^T
    G[0*1 + 0] = 0.0f;
    G[1*1 + 0] = 1.0f;
}

// ---------------------------------------------------------------------------
// Sanity check
// ---------------------------------------------------------------------------

static bool sanity_check_agreement() {
    float x[2], f[2], G[2*1];
    float h_d, Lf_d, Lg_d[1];
    float h_r, Lf_r, Lg_r[1];
    constexpr int N_CHECK = 32;
    constexpr float TOL = 1e-3f;

    for (int t = 0; t < N_CHECK; ++t) {
        sample_vdp_state(x, f, G);
        dual_cbf::evaluate_cbf(x, f, G, 1, &h_d, &Lf_d, Lg_d);
        reverse_mode_ad::evaluate_cbf(
            linear_specs, activations, DEPTH, x, f, G, 1, &h_r, &Lf_r, Lg_r);

        if (fabsf(h_d - h_r) > TOL ||
            fabsf(Lf_d - Lf_r) > TOL ||
            fabsf(Lg_d[0] - Lg_r[0]) > TOL) {
            printf("AGREEMENT_FAIL,%d,h:%e/%e,Lf:%e/%e,Lg0:%e/%e\n",
                t, h_d, h_r, Lf_d, Lf_r, Lg_d[0], Lg_r[0]);
            return false;
        }
    }
    return true;
}

static inline uint32_t bench_one_dual(
    const float* x, const float* f, const float* G,
    float* h, float* Lf, float* Lg)
{
    uint32_t t0 = bench::cycles_now();
    dual_cbf::evaluate_cbf(x, f, G, 1, h, Lf, Lg);
    uint32_t t1 = bench::cycles_now();
    return t1 - t0;
}

static inline uint32_t bench_one_rvad(
    const float* x, const float* f, const float* G,
    float* h, float* Lf, float* Lg)
{
    uint32_t t0 = bench::cycles_now();
    reverse_mode_ad::evaluate_cbf(
        linear_specs, activations, DEPTH, x, f, G, 1, h, Lf, Lg);
    uint32_t t1 = bench::cycles_now();
    return t1 - t0;
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));

    printf("\n\n========================================\n");
    printf("Van der Pol 2-64-64-1 (ReLU) benchmark\n");
    printf("CPU = %lu Hz, N_ITERS = %d\n", (unsigned long)bench::CPU_HZ, N_ITERS);
    printf("========================================\n");

    if (!sanity_check_agreement()) {
        printf("Aborting: dual and reverse-mode disagree.\n");
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    printf("Agreement check: PASS\n");

    float x[2], f[2], G[2 * 1];
    float h, Lf, Lg[1];

    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        sample_vdp_state(x, f, G);
        dual_cycles[i] = bench_one_dual(x, f, G, &h, &Lf, Lg);
    }
    auto stats_dual = bench::summarize(dual_cycles, N_ITERS);
    size_t dual_dyn = reverse_mode_ad::peak_dynamic_bytes();

    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        sample_vdp_state(x, f, G);
        rvad_cycles[i] = bench_one_rvad(x, f, G, &h, &Lf, Lg);
    }
    auto stats_rvad = bench::summarize(rvad_cycles, N_ITERS);
    size_t rvad_dyn = reverse_mode_ad::peak_dynamic_bytes();

    printf("\n----- RESULTS (CSV, parsed by host) -----\n");
    printf("DUAL_VDP,%.3f,%.3f,%.3f,%u\n",
           stats_dual.min_us, stats_dual.med_us, stats_dual.max_us,
           (unsigned)dual_dyn);
    printf("RVAD_VDP,%.3f,%.3f,%.3f,%u\n",
           stats_rvad.min_us, stats_rvad.med_us, stats_rvad.max_us,
           (unsigned)rvad_dyn);
    printf("----- END RESULTS -----\n\n");

    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
