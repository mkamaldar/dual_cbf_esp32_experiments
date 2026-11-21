// bench_main.inc -- benchmark driver.  Included once by <example>/main/main.cpp
// after example.h.  Builds for the ESP32-S3 (ESP-IDF v5.x) and, with
// -DDUAL_CBF_HOST_TEST, for a host PC (functional pre-flight only).
//
// Output lines (parsed by tools/collect.py):
//   INFO,<key>,<value>
//   TIME,<example>,<method>,<window>,<irq>,<n>,<min_us>,<med_us>,<p99_us>,<max_us>,<mean_us>
//        window: ASM  = constraint assembly only (lines 1-6 of Algorithm 1)
//                STEP = f, G (and f' products), assembly, and the QP
//        irq:    EN   = interrupts enabled (the tick interrupt can fall in the window)
//                MASK = interrupts masked on the measuring core
//   COLD,<example>,<method>,<first_call_us>
//   MEM,<example>,<method>,<static_B>,<stack_B>,<heap_peak_B>,<heap_allocs>
//   ERR,<example>,<method>,<quantity>,<max_abs_err>,<max_scaled_err>
//        scaled error = |value - ref| / max(1, |ref|), ref in double precision;
//        states at least KINK_TOL away from all ReLU activation boundaries
//   KINK,<example>,<tol>,<states closer than tol to a boundary>,<states>
//   ERRK,...  same as ERR, over the near-boundary states only
//   FLIP,<example>,<method>,<near-boundary states with scaled error > 1e-3>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "bench_types.h"
#include "bench_method.h"

#ifdef DUAL_CBF_HOST_TEST
#include <chrono>
#include <new>
#else
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_attr.h"
#include "esp_app_desc.h"
#include "esp_chip_info.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_system.h"
#include "esp_timer.h"
#endif

#ifndef BENCH_RD2
#define BENCH_RD2 0
#endif

namespace bench {

#ifndef BENCH_N_ITERS
#define BENCH_N_ITERS 1000
#endif
#ifndef BENCH_N_VAL
#ifdef DUAL_CBF_HOST_TEST
#define BENCH_N_VAL 100000
#else
#define BENCH_N_VAL 10000
#endif
#endif
static constexpr int N_ITERS = BENCH_N_ITERS;
static constexpr int N_VAL = BENCH_N_VAL;
static constexpr int N_WARMUP = 16;
static constexpr uint32_t SEED_TIMING = 20261109u;
static constexpr uint32_t SEED_VALID = 20261110u;
static constexpr uint32_t SEED_MEMORY = 20261111u;

// ---------------------------------------------------------------- timer ----
#ifdef DUAL_CBF_HOST_TEST
static inline uint32_t cycles_now() {
    using namespace std::chrono;
    return static_cast<uint32_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
static constexpr float CYCLES_PER_US = 1000.0f;   // nanoseconds
static inline uint32_t irq_mask() { return 0; }
static inline void irq_restore(uint32_t) {}
#else
static inline uint32_t cycles_now() { return esp_cpu_get_cycle_count(); }
static constexpr float CYCLES_PER_US = static_cast<float>(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
// Inline interrupt masking (no call frames, so the stack probe is not disturbed).
static inline uint32_t irq_mask() { return portSET_INTERRUPT_MASK_FROM_ISR(); }
static inline void irq_restore(uint32_t prev) { portCLEAR_INTERRUPT_MASK_FROM_ISR(prev); }
#endif

static inline float to_us(uint32_t cycles) { return static_cast<float>(cycles) / CYCLES_PER_US; }

// ---------------------------------------------------------- heap tracker ----
// Counts every allocation made anywhere in the linked program while armed
// (compiled code, libm, libc), not only allocation call sites in the source.
struct HeapTrack {
    volatile bool armed;
    uint32_t allocs;
    uint32_t frees;
    size_t live;
    size_t peak;
    void* ptr[32];
    size_t size[32];
};
static HeapTrack s_heap;

static inline __attribute__((always_inline)) void heap_on_alloc(void* p, size_t n) {
    if (!s_heap.armed || p == nullptr) return;
    s_heap.allocs++;
    for (int i = 0; i < 32; ++i) {
        if (s_heap.ptr[i] == nullptr) { s_heap.ptr[i] = p; s_heap.size[i] = n; break; }
    }
    s_heap.live += n;
    if (s_heap.live > s_heap.peak) s_heap.peak = s_heap.live;
}
static inline __attribute__((always_inline)) void heap_on_free(void* p) {
    if (!s_heap.armed || p == nullptr) return;
    s_heap.frees++;
    for (int i = 0; i < 32; ++i) {
        if (s_heap.ptr[i] == p) { s_heap.live -= s_heap.size[i]; s_heap.ptr[i] = nullptr; break; }
    }
}
static inline void heap_arm() {
    for (int i = 0; i < 32; ++i) s_heap.ptr[i] = nullptr;
    s_heap.allocs = s_heap.frees = 0;
    s_heap.live = s_heap.peak = 0;
    s_heap.armed = true;
}
static inline void heap_disarm() { s_heap.armed = false; }

}  // namespace bench

#ifdef DUAL_CBF_HOST_TEST
void* operator new(std::size_t n) { void* p = std::malloc(n); bench::heap_on_alloc(p, n); return p; }
void* operator new[](std::size_t n) { void* p = std::malloc(n); bench::heap_on_alloc(p, n); return p; }
void operator delete(void* p) noexcept { bench::heap_on_free(p); std::free(p); }
void operator delete[](void* p) noexcept { bench::heap_on_free(p); std::free(p); }
void operator delete(void* p, std::size_t) noexcept { bench::heap_on_free(p); std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { bench::heap_on_free(p); std::free(p); }
#define BENCH_HEAP_HOOKS 1
#elif CONFIG_HEAP_USE_HOOKS
// Definitions of the weak hooks declared in esp_heap_caps.h (CONFIG_HEAP_USE_HOOKS).
extern "C" void esp_heap_trace_alloc_hook(void* ptr, size_t size, uint32_t caps) {
    (void)caps;
    bench::heap_on_alloc(ptr, size);
}
extern "C" void esp_heap_trace_free_hook(void* ptr) { bench::heap_on_free(ptr); }
#define BENCH_HEAP_HOOKS 1
#else
#define BENCH_HEAP_HOOKS 0
#endif

namespace bench {

// ----------------------------------------------------------- stack probe ----
// stack_probe(false) fills PROBE_BYTES of stack below the caller with a
// pattern; stack_probe(true), called from the same frame, returns the distance
// from the deepest overwritten byte to the top of the filled region.  The
// method under test runs between the two calls with interrupts masked, so the
// result is the stack depth written by that method and its callees.
static constexpr int PROBE_BYTES = 4096;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
__attribute__((noinline, optimize("O0"))) static int stack_probe(bool scan) {
    volatile uint8_t buf[PROBE_BYTES];
    int n = 0;
    if (!scan) {
        for (int i = 0; i < PROBE_BYTES; ++i) buf[i] = 0xA5;
    } else {
        while (n < PROBE_BYTES && buf[n] == 0xA5) ++n;
        n = PROBE_BYTES - n;
    }
    return n;
}
#pragma GCC diagnostic pop

// ----------------------------------------------------------------- stats ----
static uint32_t s_cyc[N_ITERS];
static uint32_t s_sorted[N_ITERS];

struct Stats { float min_us, med_us, p99_us, max_us, mean_us; };

static Stats summarize(const uint32_t* v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { s_sorted[i] = v[i]; sum += v[i]; }
    for (int i = 1; i < n; ++i) {
        uint32_t key = s_sorted[i];
        int j = i - 1;
        while (j >= 0 && s_sorted[j] > key) { s_sorted[j + 1] = s_sorted[j]; --j; }
        s_sorted[j + 1] = key;
    }
    Stats s;
    s.min_us = to_us(s_sorted[0]);
    s.med_us = to_us(s_sorted[n / 2]);
    s.p99_us = to_us(s_sorted[(99 * n) / 100]);
    s.max_us = to_us(s_sorted[n - 1]);
    s.mean_us = static_cast<float>(sum / n) / CYCLES_PER_US;
    return s;
}

static volatile float s_sink;

// ---------------------------------------------------------------- timing ----
static void run_timing(const Method& m, bool step, bool masked) {
    Prng rng(SEED_TIMING);
    ex::Case c;
    ex::Out o;
    float u[ex::M];
    float x[ex::N];
    uint32_t cold = 0;
    for (int i = -N_WARMUP; i < N_ITERS; ++i) {
        ex::sample(rng, x);
        if (!step) ex::dynamics(x, c);
        uint32_t prev = 0;
        if (masked) prev = irq_mask();
        uint32_t t0 = cycles_now();
        if (step) {
            ex::dynamics(x, c);
            m.fn(c, o);
            ex::filter(c, o, u);
        } else {
            m.fn(c, o);
        }
        uint32_t t1 = cycles_now();
        if (masked) irq_restore(prev);
        if (i == -N_WARMUP) cold = t1 - t0;
        if (i >= 0) s_cyc[i] = t1 - t0;
        s_sink = s_sink + o.q[0] + (step ? u[0] : 0.0f);
    }
    Stats s = summarize(s_cyc, N_ITERS);
    printf("TIME,%s,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", ex::NAME, m.name,
           step ? "STEP" : "ASM", masked ? "MASK" : "EN", N_ITERS,
           s.min_us, s.med_us, s.p99_us, s.max_us, s.mean_us);
    if (!step && !masked) printf("COLD,%s,%s,%.3f\n", ex::NAME, m.name, to_us(cold));
}

// ---------------------------------------------------------------- memory ----
__attribute__((noinline)) static int probe_method(void (*fn)(const ex::Case&, ex::Out&),
                                                  const ex::Case& c, ex::Out& o) {
    stack_probe(false);
    uint32_t prev = irq_mask();
    fn(c, o);
    irq_restore(prev);
    return stack_probe(true);
}

static void run_memory() {
    Prng rng(SEED_MEMORY);
    static ex::Case c;
    static ex::Out o;
    float x[ex::N];
    ex::sample(rng, x);
    ex::dynamics(x, c);
    int base = probe_method(m_empty, c, o);
    printf("INFO,stack_probe_empty_call_B,%d\n", base);
    for (int k = 0; k < N_METHODS; ++k) {
        const Method& m = METHODS[k];
        int deepest = 0;
        for (int r = 0; r < 8; ++r) {   // several states: libm paths can differ
            ex::sample(rng, x);
            ex::dynamics(x, c);
            int used = probe_method(m.fn, c, o);
            if (used > deepest) deepest = used;
        }
        heap_arm();
        for (int r = 0; r < 8; ++r) {
            ex::sample(rng, x);
            ex::dynamics(x, c);
            m.fn(c, o);
        }
        heap_disarm();
#if BENCH_HEAP_HOOKS
        printf("MEM,%s,%s,%d,%d,%u,%u\n", ex::NAME, m.name, m.static_bytes,
               deepest - base, (unsigned)s_heap.peak, (unsigned)(s_heap.allocs / 8));
#else
        printf("MEM,%s,%s,%d,%d,NA,NA\n", ex::NAME, m.name, m.static_bytes, deepest - base);
#endif
    }
}

// ------------------------------------------------------------ validation ----
// Returns the smallest |preactivation| over the ReLU neurons, in double
// precision: the distance of the state to the nearest activation boundary.
static double reference(const ex::Case& c, double* ref) {
    double xd[ex::N], fd[ex::N], h, grad[ex::N], Hf[ex::N], min_pre;
    for (int i = 0; i < ex::N; ++i) { xd[i] = c.x[i]; fd[i] = c.f[i]; }
    ref_evaluate(xd, fd, &h, grad, Hf, &min_pre);
    double lf = 0.0;
    for (int i = 0; i < ex::N; ++i) lf += grad[i] * fd[i];
    ref[0] = h;
    ref[1] = lf;
#if BENCH_RD2
    double l2 = 0.0;
    for (int i = 0; i < ex::N; ++i) l2 += fd[i] * Hf[i] + grad[i] * (double)c.Df_f[i];
    ref[2] = l2;
    for (int j = 0; j < ex::M; ++j) {
        double acc = 0.0;
        for (int i = 0; i < ex::N; ++i)
            acc += (double)c.G[i * ex::M + j] * Hf[i] + grad[i] * (double)c.Df_G[i * ex::M + j];
        ref[3 + j] = acc;
    }
#else
    for (int j = 0; j < ex::M; ++j) {
        double acc = 0.0;
        for (int i = 0; i < ex::N; ++i) acc += grad[i] * (double)c.G[i * ex::M + j];
        ref[2 + j] = acc;
    }
#endif
    return min_pre;
}

// A ReLU network is differentiable at a state only if no preactivation is
// zero.  If a preactivation is within floating-point roundoff of zero, the
// single-precision gate can differ from the exact gate, and the returned
// derivative is then that of the adjacent affine piece.  States whose
// smallest |preactivation| is below KINK_TOL are counted and reported
// separately (ERRK lines); ERR lines cover the remaining states.
static constexpr double KINK_TOL = 1e-4;
static double s_err_abs[8][ex::Q];
static double s_err_scl[8][ex::Q];
static double s_errk_abs[8][ex::Q];
static double s_errk_scl[8][ex::Q];
static int s_flips[8];

static bool run_validation() {
    Prng rng(SEED_VALID);
    ex::Case c;
    ex::Out o;
    float x[ex::N];
    double ref[ex::Q];
    for (int k = 0; k < N_METHODS; ++k) {
        s_flips[k] = 0;
        for (int q = 0; q < ex::Q; ++q) {
            s_err_abs[k][q] = s_err_scl[k][q] = 0.0;
            s_errk_abs[k][q] = s_errk_scl[k][q] = 0.0;
        }
    }
    int n_kink = 0;
    for (int i = 0; i < N_VAL; ++i) {
        ex::sample(rng, x);
        ex::dynamics(x, c);
        const bool near_kink = reference(c, ref) < KINK_TOL;
        if (near_kink) ++n_kink;
        for (int k = 0; k < N_METHODS; ++k) {
            METHODS[k].fn(c, o);
            bool flip = false;
            for (int q = 0; q < ex::Q; ++q) {
                double e = fabs((double)o.q[q] - ref[q]);
                double s = e / (fabs(ref[q]) > 1.0 ? fabs(ref[q]) : 1.0);
                double* ea = near_kink ? s_errk_abs[k] : s_err_abs[k];
                double* es = near_kink ? s_errk_scl[k] : s_err_scl[k];
                if (e > ea[q]) ea[q] = e;
                if (s > es[q]) es[q] = s;
                if (near_kink && s > 1e-3) flip = true;
            }
            if (flip) ++s_flips[k];
        }
#ifndef DUAL_CBF_HOST_TEST
        if ((i & 63) == 63) vTaskDelay(1);   // the double-precision reference is slow
#endif
    }
    printf("KINK,%s,%.1e,%d,%d\n", ex::NAME, KINK_TOL, n_kink, N_VAL);
    bool ok = true;
    for (int k = 0; k < N_METHODS; ++k) {
        for (int q = 0; q < ex::Q; ++q) {
            printf("ERR,%s,%s,%s,%.3e,%.3e\n", ex::NAME, METHODS[k].name, ex::QNAME[q],
                   s_err_abs[k][q], s_err_scl[k][q]);
            if (METHODS[k].exact && s_err_scl[k][q] > 1e-3) ok = false;
        }
    }
    if (n_kink > 0) {
        for (int k = 0; k < N_METHODS; ++k) {
            for (int q = 0; q < ex::Q; ++q)
                printf("ERRK,%s,%s,%s,%.3e,%.3e\n", ex::NAME, METHODS[k].name, ex::QNAME[q],
                       s_errk_abs[k][q], s_errk_scl[k][q]);
            printf("FLIP,%s,%s,%d\n", ex::NAME, METHODS[k].name, s_flips[k]);
        }
    }
    return ok;
}

// ------------------------------------------------------------ provenance ----
static void print_info() {
    printf("INFO,example,%s\n", ex::NAME);
    printf("INFO,network,%s\n", NET.widths);
    printf("INFO,activations,%s\n", NET.activations);
    printf("INFO,relative_degree,%d\n", NET.relative_degree);
    printf("INFO,n_inputs,%d\n", NET.n_inputs);
    printf("INFO,parameter_bytes,%d\n", NET.parameter_bytes);
    printf("INFO,w_df_bytes,%d\n", NET.w_df_bytes);
    printf("INFO,w_ad_bytes,%d\n", NET.w_ad_bytes);
    printf("INFO,n_iters,%d\n", N_ITERS);
    printf("INFO,n_warmup,%d\n", N_WARMUP);
    printf("INFO,n_validation,%d\n", N_VAL);
    printf("INFO,seed_timing,%u\n", (unsigned)SEED_TIMING);
    printf("INFO,seed_validation,%u\n", (unsigned)SEED_VALID);
    printf("INFO,compiler,%s\n", __VERSION__);
#ifdef __OPTIMIZE__
    printf("INFO,optimize,1\n");
#else
    printf("INFO,optimize,0\n");
#endif
#ifdef __FAST_MATH__
    printf("INFO,fast_math,1\n");
#else
    printf("INFO,fast_math,0\n");
#endif
#ifdef BENCH_CXX_FLAGS
    printf("INFO,cxx_flags,%s\n", BENCH_CXX_FLAGS);
#endif
#ifdef DUAL_CBF_HOST_TEST
    printf("INFO,target,host\n");
#else
    const esp_app_desc_t* app = esp_app_get_description();
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    printf("INFO,target,%s\n", CONFIG_IDF_TARGET);
    printf("INFO,idf_version,%s\n", esp_get_idf_version());
    printf("INFO,app_version,%s\n", app->version);       // git describe of this repository
    printf("INFO,build,%s %s\n", app->date, app->time);
    printf("INFO,chip_model,%d\n", (int)chip.model);
    printf("INFO,chip_revision,v%d.%d\n", chip.revision / 100, chip.revision % 100);
    printf("INFO,chip_cores,%d\n", (int)chip.cores);
    printf("INFO,cpu_mhz,%d\n", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    {   // check the cycle counter against the microsecond timer
        int64_t t0 = esp_timer_get_time();
        uint32_t c0 = cycles_now();
        while (esp_timer_get_time() - t0 < 100000) {}
        uint32_t c1 = cycles_now();
        int64_t t1 = esp_timer_get_time();
        printf("INFO,measured_cycles_per_us,%.3f\n", (double)(c1 - c0) / (double)(t1 - t0));
    }
    printf("INFO,core_id,%d\n", esp_cpu_get_core_id());
    printf("INFO,tick_hz,%d\n", CONFIG_FREERTOS_HZ);
#ifdef CONFIG_ESP32S3_DATA_CACHE_SIZE
    printf("INFO,dcache_bytes,%d\n", (int)CONFIG_ESP32S3_DATA_CACHE_SIZE);
    printf("INFO,icache_bytes,%d\n", (int)CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE);
#endif
    printf("INFO,heap_hooks,%d\n", BENCH_HEAP_HOOKS);
    printf("INFO,free_heap_B,%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_DEFAULT));
#endif
}

static int run_all() {
    methods_init();
    printf("\n----- BEGIN %s -----\n", ex::NAME);
    print_info();

    printf("INFO,status,validating against the double-precision reference (a few minutes on the ESP32-S3)\n");
    fflush(stdout);
    bool ok = run_validation();
    printf("INFO,validation,%s\n", ok ? "PASS" : "FAIL");

    for (int k = 0; k < N_METHODS; ++k) run_timing(METHODS[k], false, false);
    for (int k = 0; k < N_METHODS; ++k) run_timing(METHODS[k], true, false);
#ifndef DUAL_CBF_HOST_TEST
    for (int k = 0; k < N_METHODS; ++k) run_timing(METHODS[k], false, true);
    for (int k = 0; k < N_METHODS; ++k) run_timing(METHODS[k], true, true);
#endif
    run_memory();
    printf("----- END %s -----\n\n", ex::NAME);
    return ok ? 0 : 1;
}

}  // namespace bench

#ifdef DUAL_CBF_HOST_TEST
int main() { return bench::run_all(); }
#else
extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));   // let the console come up
    bench::run_all();
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
#endif
