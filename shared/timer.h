// timer.h -- cycle-accurate timing on the ESP32-S3.
//
// The ESP32-S3 exposes the Xtensa CCOUNT register (PRID_CYCLE_COUNT in
// ESP-IDF). At a fixed CPU clock of 240 MHz, one tick equals 1/240 microsecond.
// This header wraps the read into a simple inline routine and provides
// a microsecond conversion. Reading CCOUNT takes a single cycle and never
// touches FreeRTOS, so it is safe to call from inside a critical section.

#pragma once

#include <stdint.h>

#ifdef DUAL_CBF_HOST_TEST
// Host build: caller provides esp_cpu_get_cycle_count() as a stub.
extern "C" uint32_t esp_cpu_get_cycle_count();
#else
#include "esp_cpu.h"
#endif

namespace bench {

// CPU clock in Hz. We pin to 240 MHz in sdkconfig.defaults.
static constexpr uint32_t CPU_HZ = 240000000u;

inline uint32_t cycles_now() {
    return esp_cpu_get_cycle_count();
}

inline float cycles_to_us(uint32_t delta) {
    // Cast first to avoid integer truncation; CPU_HZ / 1e6 = 240 ticks/us.
    return static_cast<float>(delta) / (static_cast<float>(CPU_HZ) / 1.0e6f);
}

// Drop-in helper: take many samples and report min/median/max.
struct Stats {
    uint32_t min_cycles;
    uint32_t med_cycles;
    uint32_t max_cycles;
    float min_us;
    float med_us;
    float max_us;
};

inline Stats summarize(const uint32_t* samples, int n) {
    // Tiny insertion sort -- n is small (we use 1000 samples), and we
    // explicitly avoid heap allocation. For larger n you would want
    // qsort or median-of-medians.
    static uint32_t buf[10000];
    for (int i = 0; i < n; ++i) buf[i] = samples[i];
    for (int i = 1; i < n; ++i) {
        uint32_t key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) { buf[j+1] = buf[j]; --j; }
        buf[j+1] = key;
    }
    Stats s;
    s.min_cycles = buf[0];
    s.med_cycles = buf[n/2];
    s.max_cycles = buf[n-1];
    s.min_us = cycles_to_us(s.min_cycles);
    s.med_us = cycles_to_us(s.med_cycles);
    s.max_us = cycles_to_us(s.max_cycles);
    return s;
}

}  // namespace bench
