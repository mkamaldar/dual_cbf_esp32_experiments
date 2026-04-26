// main.cpp -- Bicycle 4-32-32-1 benchmark on the ESP32-S3.
//
// Runs N iterations of:
//   (1) dual_cbf::evaluate_cbf       (the proposed dual-algebraic compiler)
//   (2) reverse_mode_ad::evaluate_cbf (hand-tuned reverse-mode AD baseline)
// across random valid bicycle states, records per-call cycle counts, and
// prints summary statistics over UART for the host parser to ingest.
//
// Output format (one line per method, parsed by 04_collect_results.py):
//   DUAL_BICYCLE,<min_us>,<med_us>,<max_us>,<peak_dynamic_bytes>
//   RVAD_BICYCLE,<min_us>,<med_us>,<max_us>,<peak_dynamic_bytes>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_random.h"

#include "dual_cbf.h"          // emits dual_cbf::W0/b0/.../W2/b2 and INPUT_DIM, etc.
#include "reverse_mode_ad.h"
#include "timer.h"

// How many iterations to time. 1000 gives ~1% precision on the median.
static constexpr int N_ITERS = 1000;

// Arrays for cycle counts. Allocated statically so we don't pollute the
// memory measurement with our own bookkeeping.
static uint32_t dual_cycles[N_ITERS];
static uint32_t rvad_cycles[N_ITERS];

// ---------------------------------------------------------------------------
// Reverse-mode adapter: build LinearSpec/ActivationSpec arrays that point
// at the same const weight arrays embedded by the dual compiler.
// ---------------------------------------------------------------------------

static const reverse_mode_ad::LinearSpec linear_specs[] = {
    {dual_cbf::W0, dual_cbf::b0, 4,  32},
    {dual_cbf::W1, dual_cbf::b1, 32, 32},
    {dual_cbf::W2, dual_cbf::b2, 32, 1},
};
static constexpr int DEPTH = 3;

static const reverse_mode_ad::ActivationSpec activations[] = {
    {reverse_mode_ad::ActKind::ReLU},
    {reverse_mode_ad::ActKind::ReLU},
};

// ---------------------------------------------------------------------------
// State sampling. Mix positive and negative ReLU branches by sampling
// realistic bicycle states from a uniform distribution.
// ---------------------------------------------------------------------------

static inline float urand(float lo, float hi) {
    // 24-bit uniform from esp_random()
    uint32_t r = esp_random();
    float u = static_cast<float>(r & 0x00FFFFFF) / static_cast<float>(0x01000000);
    return lo + u * (hi - lo);
}

static void sample_bicycle_state(
    float x[4], float f[4], float G[4 * 2])
{
    x[0] = urand(-7.0f, 7.0f);          // px
    x[1] = urand(-7.0f, 7.0f);          // py
    x[2] = urand(-3.14159f, 3.14159f);  // psi
    x[3] = urand(0.0f, 35.0f);          // v

    f[0] = x[3] * cosf(x[2]);
    f[1] = x[3] * sinf(x[2]);
    f[2] = 0.0f;
    f[3] = 0.0f;

    // G is row-major (n=4) x (m=2):
    //   col 0 (acceleration): [0, 0, 0, 1]^T
    //   col 1 (steering):     [0, 0, v/L, 0]^T  with L = 2.5 m
    const float L = 2.5f;
    G[0*2 + 0] = 0.0f;       G[0*2 + 1] = 0.0f;
    G[1*2 + 0] = 0.0f;       G[1*2 + 1] = 0.0f;
    G[2*2 + 0] = 0.0f;       G[2*2 + 1] = x[3] / L;
    G[3*2 + 0] = 1.0f;       G[3*2 + 1] = 0.0f;
}

// ---------------------------------------------------------------------------
// Sanity check: agreement between dual and reverse-mode AD on identical
// inputs. Catches bugs in the reverse-mode adapter before reporting timings.
// ---------------------------------------------------------------------------

static bool sanity_check_agreement() {
    float x[4], f[4], G[4*2];
    float h_d, Lf_d, Lg_d[2];
    float h_r, Lf_r, Lg_r[2];
    constexpr int N_CHECK = 32;
    constexpr float TOL = 1e-3f;

    for (int t = 0; t < N_CHECK; ++t) {
        sample_bicycle_state(x, f, G);
        dual_cbf::evaluate_cbf(x, f, G, 2, &h_d, &Lf_d, Lg_d);
        reverse_mode_ad::evaluate_cbf(
            linear_specs, activations, DEPTH, x, f, G, 2, &h_r, &Lf_r, Lg_r);

        if (fabsf(h_d - h_r) > TOL ||
            fabsf(Lf_d - Lf_r) > TOL ||
            fabsf(Lg_d[0] - Lg_r[0]) > TOL ||
            fabsf(Lg_d[1] - Lg_r[1]) > TOL) {
            printf("AGREEMENT_FAIL,%d,h:%e/%e,Lf:%e/%e,Lg0:%e/%e,Lg1:%e/%e\n",
                t, h_d, h_r, Lf_d, Lf_r, Lg_d[0], Lg_r[0], Lg_d[1], Lg_r[1]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Benchmark drivers. Each takes a single sample, runs the method, returns
// cycles elapsed (including the sample-fetch and any output writes -- we
// keep both methods on equal footing).
// ---------------------------------------------------------------------------

static inline uint32_t bench_one_dual(
    const float* x, const float* f, const float* G,
    float* h, float* Lf, float* Lg)
{
    uint32_t t0 = bench::cycles_now();
    dual_cbf::evaluate_cbf(x, f, G, 2, h, Lf, Lg);
    uint32_t t1 = bench::cycles_now();
    return t1 - t0;
}

static inline uint32_t bench_one_rvad(
    const float* x, const float* f, const float* G,
    float* h, float* Lf, float* Lg)
{
    uint32_t t0 = bench::cycles_now();
    reverse_mode_ad::evaluate_cbf(
        linear_specs, activations, DEPTH, x, f, G, 2, h, Lf, Lg);
    uint32_t t1 = bench::cycles_now();
    return t1 - t0;
}

// ---------------------------------------------------------------------------
// app_main
// ---------------------------------------------------------------------------

extern "C" void app_main(void) {
    // Give the USB-Serial-JTAG link time to come up before we print.
    vTaskDelay(pdMS_TO_TICKS(1500));

    printf("\n\n========================================\n");
    printf("Bicycle 4-32-32-1 (ReLU) benchmark\n");
    printf("CPU = %lu Hz, N_ITERS = %d\n", (unsigned long)bench::CPU_HZ, N_ITERS);
    printf("========================================\n");

    // Sanity check before reporting timings.
    if (!sanity_check_agreement()) {
        printf("Aborting: dual and reverse-mode disagree.\n");
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    printf("Agreement check: PASS (32 random states, |err| < 1e-3)\n");

    // Pre-allocate per-iteration scratch.
    float x[4], f[4], G[4 * 2];
    float h, Lf, Lg[2];

    // ----- Dual compiler -----
    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        sample_bicycle_state(x, f, G);
        dual_cycles[i] = bench_one_dual(x, f, G, &h, &Lf, Lg);
    }
    auto stats_dual = bench::summarize(dual_cycles, N_ITERS);
    size_t dual_dyn = reverse_mode_ad::peak_dynamic_bytes();  // expected: 0

    // ----- Reverse-mode AD -----
    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        sample_bicycle_state(x, f, G);
        rvad_cycles[i] = bench_one_rvad(x, f, G, &h, &Lf, Lg);
    }
    auto stats_rvad = bench::summarize(rvad_cycles, N_ITERS);
    size_t rvad_dyn = reverse_mode_ad::peak_dynamic_bytes();

    // Machine-readable lines for 04_collect_results.py
    printf("\n----- RESULTS (CSV, parsed by host) -----\n");
    printf("DUAL_BICYCLE,%.3f,%.3f,%.3f,%u\n",
           stats_dual.min_us, stats_dual.med_us, stats_dual.max_us,
           (unsigned)dual_dyn);
    printf("RVAD_BICYCLE,%.3f,%.3f,%.3f,%u\n",
           stats_rvad.min_us, stats_rvad.med_us, stats_rvad.max_us,
           (unsigned)rvad_dyn);
    printf("----- END RESULTS -----\n\n");

    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
