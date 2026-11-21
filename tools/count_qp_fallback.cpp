// count_qp_fallback.cpp -- host count of the QP fallback at the timing states.
//
// Replays the timing loop of shared/bench_main.inc (16 warm-up calls, then
// 1000 states drawn with SEED_TIMING) for every method of one project and
// counts the states at which the constraint cannot be met inside the input
// box, so that qp_halfspace_box returns false and the fallback input (the box
// point that maximizes a.u) is used.  Host build, no ESP32 needed:
//
//   g++ -std=c++17 -O2 -DDUAL_CBF_HOST_TEST -I shared -I <project>/main \
//       tools/count_qp_fallback.cpp <project>/main/methods.cpp -o count
//   ./count            # prints <example>,<method>,<fallback states of 1000>
//
#include <cstdio>
#include "example.h"
#include "bench_types.h"
#include "bench_method.h"

int main() {
    bench::methods_init();
    for (int k = 0; k < bench::N_METHODS; ++k) {
        const bench::Method& m = bench::METHODS[k];
        bench::Prng rng(20261109u);          // SEED_TIMING of bench_main.inc
        ex::Case c;
        ex::Out o;
        float u[ex::M];
        float x[ex::N];
        int fallback = 0;
        for (int i = -16; i < 1000; ++i) {   // N_WARMUP = 16, N_ITERS = 1000
            ex::sample(rng, x);
            ex::dynamics(x, c);
            m.fn(c, o);
            const bool feasible = ex::filter(c, o, u);
            if (i >= 0 && !feasible) ++fallback;
        }
        std::printf("%s,%s,%d\n", ex::NAME, m.name, fallback);
    }
    return 0;
}
