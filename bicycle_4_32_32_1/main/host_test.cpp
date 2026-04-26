// host_test.cpp -- compile and run the embedded benchmark logic on the host.
//
// This is NOT used on the ESP32-S3. Its only job is to verify that
// main.cpp + reverse_mode_ad.h + dual_cbf.h are mutually consistent
// (compile cleanly, link cleanly, produce numerically agreeing output)
// before we boot the actual hardware. ESP-IDF includes are stubbed.

// ---- Stub the ESP-IDF includes ----
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <chrono>

// Substitute esp_random with rand()
static inline uint32_t esp_random() { return (uint32_t)rand(); }

// Substitute esp_cpu_get_cycle_count with a high-resolution monotonic clock.
// We pretend the host runs at 240 MHz so the conversion in timer.h still
// produces something sensible.
extern "C" uint32_t esp_cpu_get_cycle_count() {
    using namespace std::chrono;
    static auto t0 = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    auto ns = duration_cast<nanoseconds>(now - t0).count();
    // Fake 240 MHz: 240 cycles/us = 0.24 cycles/ns
    return (uint32_t)((ns * 240) / 1000);
}

// Stub FreeRTOS pieces used by main.cpp.
static inline void vTaskDelay(uint32_t /*ticks*/) {}
#define pdMS_TO_TICKS(ms) ((uint32_t)(ms))

// Suppress the freertos headers
#define FREERTOS_FREERTOS_H
#define FREERTOS_TASK_H
#define ESP_SYSTEM_H
#define ESP_RANDOM_H
#define ESP_CPU_H
namespace freertos_dummy {} // appease the include guards

// ---- Provide a no-op extern "C" app_main entry point ----
#include "../shared/timer.h"
#include "../shared/reverse_mode_ad.h"
// dual_cbf.h is in the same main/ directory, included via the compile-time -I

#include "dual_cbf.h"

int main() {
    printf("Host-side regression run.\n");

    // Build a state vector and dynamics that match the bicycle example.
    float x[4] = {1.0f, 0.5f, 0.1f, 8.0f};
    float f[4] = {x[3]*cosf(x[2]), x[3]*sinf(x[2]), 0.0f, 0.0f};
    constexpr float L = 2.5f;
    float G[4 * 2] = {
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, x[3] / L,
        1.0f, 0.0f,
    };

    // Reverse-mode adapter pointing at the dual compiler's static weights.
    static const reverse_mode_ad::LinearSpec linear_specs[] = {
        {dual_cbf::W0, dual_cbf::b0, 4,  32},
        {dual_cbf::W1, dual_cbf::b1, 32, 32},
        {dual_cbf::W2, dual_cbf::b2, 32, 1},
    };
    static const reverse_mode_ad::ActivationSpec activations[] = {
        {reverse_mode_ad::ActKind::ReLU},
        {reverse_mode_ad::ActKind::ReLU},
    };

    float h_d, Lf_d, Lg_d[2];
    float h_r, Lf_r, Lg_r[2];

    dual_cbf::evaluate_cbf(x, f, G, 2, &h_d, &Lf_d, Lg_d);
    reverse_mode_ad::evaluate_cbf(
        linear_specs, activations, 3, x, f, G, 2, &h_r, &Lf_r, Lg_r);

    printf("dual:  h=%+e  Lf=%+e  Lg=[%+e, %+e]\n",
           h_d, Lf_d, Lg_d[0], Lg_d[1]);
    printf("rvad:  h=%+e  Lf=%+e  Lg=[%+e, %+e]\n",
           h_r, Lf_r, Lg_r[0], Lg_r[1]);

    float max_err = 0.0f;
    max_err = fmaxf(max_err, fabsf(h_d - h_r));
    max_err = fmaxf(max_err, fabsf(Lf_d - Lf_r));
    max_err = fmaxf(max_err, fabsf(Lg_d[0] - Lg_r[0]));
    max_err = fmaxf(max_err, fabsf(Lg_d[1] - Lg_r[1]));
    printf("max abs error: %e\n", max_err);
    if (max_err > 1e-4f) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");

    // Bench just for fun.
    constexpr int N = 1000;
    uint32_t* dual_cycles = new uint32_t[N];
    uint32_t* rvad_cycles = new uint32_t[N];
    for (int i = 0; i < N; ++i) {
        x[0] = ((float)rand() / RAND_MAX - 0.5f) * 14.0f;
        x[1] = ((float)rand() / RAND_MAX - 0.5f) * 14.0f;
        x[2] = ((float)rand() / RAND_MAX - 0.5f) * 6.28f;
        x[3] = ((float)rand() / RAND_MAX) * 35.0f;
        f[0] = x[3]*cosf(x[2]); f[1] = x[3]*sinf(x[2]);
        G[2*2 + 1] = x[3] / L;

        uint32_t t0 = bench::cycles_now();
        dual_cbf::evaluate_cbf(x, f, G, 2, &h_d, &Lf_d, Lg_d);
        uint32_t t1 = bench::cycles_now();
        dual_cycles[i] = t1 - t0;

        t0 = bench::cycles_now();
        reverse_mode_ad::evaluate_cbf(
            linear_specs, activations, 3, x, f, G, 2, &h_r, &Lf_r, Lg_r);
        t1 = bench::cycles_now();
        rvad_cycles[i] = t1 - t0;
    }
    auto sd = bench::summarize(dual_cycles, N);
    auto sr = bench::summarize(rvad_cycles, N);
    printf("[host bench, NOT representative of MCU]\n");
    printf("  dual:  min=%.2f us, med=%.2f us, max=%.2f us\n",
           sd.min_us, sd.med_us, sd.max_us);
    printf("  rvad:  min=%.2f us, med=%.2f us, max=%.2f us\n",
           sr.min_us, sr.med_us, sr.max_us);
    printf("  rvad peak dynamic bytes: %zu\n",
           reverse_mode_ad::peak_dynamic_bytes());
    delete[] dual_cycles;
    delete[] rvad_cycles;
    return 0;
}
