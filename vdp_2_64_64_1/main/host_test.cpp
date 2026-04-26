// host_test.cpp -- compile and run the Van der Pol benchmark logic on the host.
// Stub ESP-IDF includes the same way as bicycle/main/host_test.cpp.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <chrono>

static inline uint32_t esp_random() { return (uint32_t)rand(); }

extern "C" uint32_t esp_cpu_get_cycle_count() {
    using namespace std::chrono;
    static auto t0 = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    auto ns = duration_cast<nanoseconds>(now - t0).count();
    return (uint32_t)((ns * 240) / 1000);
}

static inline void vTaskDelay(uint32_t /*ticks*/) {}
#define pdMS_TO_TICKS(ms) ((uint32_t)(ms))

#include "../../shared/timer.h"
#include "../../shared/reverse_mode_ad.h"
#include "dual_cbf.h"

int main() {
    printf("Van der Pol host regression run.\n");

    static const reverse_mode_ad::LinearSpec linear_specs[] = {
        {dual_cbf::W0, dual_cbf::b0, 2,  64},
        {dual_cbf::W1, dual_cbf::b1, 64, 64},
        {dual_cbf::W2, dual_cbf::b2, 64, 1},
    };
    static const reverse_mode_ad::ActivationSpec activations[] = {
        {reverse_mode_ad::ActKind::ReLU},
        {reverse_mode_ad::ActKind::ReLU},
    };

    constexpr int N_CHECK = 64;
    constexpr float TOL = 1e-4f;
    float worst = 0.0f;
    for (int t = 0; t < N_CHECK; ++t) {
        float x[2] = {
            ((float)rand() / RAND_MAX - 0.5f) * 8.0f,
            ((float)rand() / RAND_MAX - 0.5f) * 7.0f,
        };
        constexpr float mu = 1.0f;
        float f[2] = {x[1], -x[0] + mu * (1.0f - x[0]*x[0]) * x[1]};
        float G[2 * 1] = {0.0f, 1.0f};

        float h_d, Lf_d, Lg_d[1];
        float h_r, Lf_r, Lg_r[1];

        dual_cbf::evaluate_cbf(x, f, G, 1, &h_d, &Lf_d, Lg_d);
        reverse_mode_ad::evaluate_cbf(
            linear_specs, activations, 3, x, f, G, 1, &h_r, &Lf_r, Lg_r);

        float err = fmaxf(fabsf(h_d - h_r),
                  fmaxf(fabsf(Lf_d - Lf_r), fabsf(Lg_d[0] - Lg_r[0])));
        worst = fmaxf(worst, err);
    }
    printf("Worst absolute error over %d random states: %.3e\n", N_CHECK, worst);
    if (worst > TOL) { printf("FAIL\n"); return 1; }
    printf("PASS\n");

    // Quick host bench, NOT representative of MCU timings.
    constexpr int N = 500;
    auto* dual_cycles = new uint32_t[N];
    auto* rvad_cycles = new uint32_t[N];
    reverse_mode_ad::peak_dynamic_bytes() = 0;
    for (int i = 0; i < N; ++i) {
        float x[2] = {
            ((float)rand() / RAND_MAX - 0.5f) * 8.0f,
            ((float)rand() / RAND_MAX - 0.5f) * 7.0f,
        };
        constexpr float mu = 1.0f;
        float f[2] = {x[1], -x[0] + mu * (1.0f - x[0]*x[0]) * x[1]};
        float G[2*1] = {0.0f, 1.0f};
        float h, Lf, Lg[1];
        uint32_t t0 = bench::cycles_now();
        dual_cbf::evaluate_cbf(x, f, G, 1, &h, &Lf, Lg);
        uint32_t t1 = bench::cycles_now();
        dual_cycles[i] = t1 - t0;

        t0 = bench::cycles_now();
        reverse_mode_ad::evaluate_cbf(
            linear_specs, activations, 3, x, f, G, 1, &h, &Lf, Lg);
        t1 = bench::cycles_now();
        rvad_cycles[i] = t1 - t0;
    }
    auto sd = bench::summarize(dual_cycles, N);
    auto sr = bench::summarize(rvad_cycles, N);
    printf("[host bench, NOT MCU]\n  dual:  med=%.2f us\n  rvad:  med=%.2f us  peak heap=%zu B\n",
           sd.med_us, sr.med_us, reverse_mode_ad::peak_dynamic_bytes());
    delete[] dual_cycles;
    delete[] rvad_cycles;
    return 0;
}
