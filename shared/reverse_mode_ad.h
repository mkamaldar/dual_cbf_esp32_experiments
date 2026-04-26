// reverse_mode_ad.h -- hand-tuned reverse-mode AD baseline for neural CBFs.
//
// This header implements the reverse-mode recurrence from eq. (29) of the
// paper. It is meant to be the fairest possible "general-purpose AD runtime"
// baseline runnable on the ESP32-S3:
//
//   * The activation cache is allocated from the heap (operator new[]),
//     matching the behavior of a real AD framework that builds a tape during
//     the forward pass. We measure peak heap usage explicitly so the
//     "dynamic memory" column in Fig. 3 is faithful.
//   * The backward recurrence is the textbook one: delta_{i-1} = W_i^T *
//     diag(sigma'_i(a_i)) * delta_i, terminating with grad h = delta_0.
//   * After the gradient is extracted, we form the inner products
//     grad h^T f and grad h^T G to assemble the CBF safety constraint
//     (per Remark 5 of the paper).
//
// Each example specializes this template by passing in:
//   - the network topology as a NetworkSpec,
//   - pointers to its weight matrices (the same const arrays embedded by
//     the dual-cbf compiler -- we reuse them so the comparison is fair),
//   - the scalar activation derivative for each hidden layer.

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

namespace reverse_mode_ad {

// ---------------------------------------------------------------------------
// Activation derivatives. Matches the dual compiler's expression library.
// ---------------------------------------------------------------------------

inline float relu_fwd(float x)        { return x > 0.0f ? x : 0.0f; }
inline float relu_dfwd(float x)       { return x > 0.0f ? 1.0f : 0.0f; }

inline float softplus_fwd(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
}
inline float softplus_dfwd(float x)   { return 1.0f / (1.0f + expf(-x)); }

inline float tanh_fwd(float x)        { return tanhf(x); }
inline float tanh_dfwd(float x)       { float t = tanhf(x); return 1.0f - t*t; }

// ---------------------------------------------------------------------------
// Shape descriptors for a feedforward CBF network. The user supplies these
// per example; here we only need pointers and dimensions.
// ---------------------------------------------------------------------------

enum class ActKind { ReLU, Softplus, Tanh };

struct LinearSpec {
    const float* W;   // row-major, shape (n_out, n_in)
    const float* b;   // length n_out
    int n_in;
    int n_out;
};

struct ActivationSpec {
    ActKind kind;
};

// Tracker for the largest single allocation we ever hold. The ESP-IDF
// program reads this after each pass to populate the "dynamic memory"
// numbers in the paper. We measure the cache itself; the embedded weights
// are static and identical between dual and reverse-mode, so we exclude
// them by convention.
inline size_t& peak_dynamic_bytes() {
    static size_t s = 0;
    return s;
}

// ---------------------------------------------------------------------------
// Forward pass that caches every pre-activation a_i and post-activation
// hat_a_i. The cache is heap-allocated, mimicking what a real AD runtime
// does when it builds a tape.
// ---------------------------------------------------------------------------

struct ActivationCache {
    float**  pre;       // pre[i] of length linear[i].n_out, i in [0, depth)
    float**  post;      // post[i] of length linear[i].n_out, i in [0, depth)
    int      depth;     // number of linear layers
    size_t   bytes;     // total heap bytes, for memory accounting

    static ActivationCache allocate(const LinearSpec* linear, int depth) {
        ActivationCache c;
        c.depth = depth;
        c.pre  = new float*[depth];
        c.post = new float*[depth];
        c.bytes = 2 * depth * sizeof(float*);
        for (int i = 0; i < depth; ++i) {
            int n = linear[i].n_out;
            c.pre[i]  = new float[n];
            c.post[i] = new float[n];
            c.bytes += 2 * n * sizeof(float);
        }
        if (c.bytes > peak_dynamic_bytes()) peak_dynamic_bytes() = c.bytes;
        return c;
    }

    void free() {
        for (int i = 0; i < depth; ++i) {
            delete[] pre[i];
            delete[] post[i];
        }
        delete[] pre;
        delete[] post;
    }
};

// ---------------------------------------------------------------------------
// Apply componentwise activation
// ---------------------------------------------------------------------------

inline void apply_activation(ActKind kind, const float* in, float* out, int n) {
    switch (kind) {
        case ActKind::ReLU:     for (int i = 0; i < n; ++i) out[i] = relu_fwd(in[i]);     break;
        case ActKind::Softplus: for (int i = 0; i < n; ++i) out[i] = softplus_fwd(in[i]); break;
        case ActKind::Tanh:     for (int i = 0; i < n; ++i) out[i] = tanh_fwd(in[i]);     break;
    }
}

inline float activation_deriv(ActKind kind, float pre) {
    switch (kind) {
        case ActKind::ReLU:     return relu_dfwd(pre);
        case ActKind::Softplus: return softplus_dfwd(pre);
        case ActKind::Tanh:     return tanh_dfwd(pre);
    }
    return 0.0f;
}

// ---------------------------------------------------------------------------
// y = W x + b   (W row-major)
// ---------------------------------------------------------------------------

inline void linear_apply(
    const LinearSpec& L, const float* x, float* y)
{
    for (int i = 0; i < L.n_out; ++i) {
        float acc = L.b[i];
        for (int j = 0; j < L.n_in; ++j) acc += L.W[i * L.n_in + j] * x[j];
        y[i] = acc;
    }
}

// y = W^T d   (no bias)
inline void linear_transpose_apply(
    const LinearSpec& L, const float* d, float* y)
{
    for (int j = 0; j < L.n_in; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < L.n_out; ++i) acc += L.W[i * L.n_in + j] * d[i];
        y[j] = acc;
    }
}

// ---------------------------------------------------------------------------
// Public API.
//
//   forward_with_cache: runs the network, caches activations.
//   backward:           runs eq. (29) and writes grad h(x) into out_grad.
//   evaluate_cbf:       full reverse-mode CBF assembly = forward + backward
//                       + (m+1) inner products.
// ---------------------------------------------------------------------------

// In-place forward. Returns h(x).
inline float forward_with_cache(
    const LinearSpec* linear,
    const ActivationSpec* activations,  // length depth-1
    int depth,
    const float* x,
    ActivationCache& cache)
{
    // For correctness regardless of widths, we use the cache itself as the
    // intermediate buffer.
    const float* in = x;
    for (int i = 0; i < depth; ++i) {
        linear_apply(linear[i], in, cache.pre[i]);
        if (i < depth - 1) {
            apply_activation(activations[i].kind, cache.pre[i], cache.post[i], linear[i].n_out);
            in = cache.post[i];
        } else {
            // Terminal linear layer: post = pre.
            for (int k = 0; k < linear[i].n_out; ++k) cache.post[i][k] = cache.pre[i][k];
        }
    }
    return cache.post[depth - 1][0];
}

// Backward pass. Writes grad h(x) into out_grad (length linear[0].n_in).
//
// Implements eq. (29): delta_{i-1} = W_i^T diag(sigma'_i(a_i)) delta_i.
// We scratch through two heap buffers per pass; this matches what an AD
// framework's tape replay does.
inline void backward(
    const LinearSpec* linear,
    const ActivationSpec* activations,
    int depth,
    const ActivationCache& cache,
    float* out_grad)
{
    // delta starts at output (length 1) with value 1.
    // We walk backward; at each step delta is of length linear[i].n_out.
    int max_w = 0;
    for (int i = 0; i < depth; ++i)
        if (linear[i].n_out > max_w) max_w = linear[i].n_out;
    if (linear[0].n_in > max_w) max_w = linear[0].n_in;

    float* delta = new float[max_w];
    float* tmp   = new float[max_w];
    size_t local_bytes = 2 * max_w * sizeof(float);
    // Add the backward scratch on top of the activation cache and update
    // the peak. peak_dynamic_bytes() should reflect the largest LIVE
    // allocation observed -- cache + backward scratch are both alive
    // during the backward pass.
    size_t cache_bytes = cache.bytes;
    size_t total = cache_bytes + local_bytes;
    if (total > peak_dynamic_bytes()) peak_dynamic_bytes() = total;

    delta[0] = 1.0f;  // d h / d h = 1
    int delta_len = 1;

    for (int i = depth - 1; i >= 0; --i) {
        // 1) scale by activation derivative if a hidden activation exists
        //    above this linear layer (i.e. between linear[i] and linear[i+1])
        if (i < depth - 1) {
            int n = linear[i].n_out;
            for (int k = 0; k < n; ++k) {
                float sigp = activation_deriv(activations[i].kind, cache.pre[i][k]);
                delta[k] *= sigp;
            }
        }
        // 2) multiply by W_i^T  -> length n_in
        linear_transpose_apply(linear[i], delta, tmp);
        delta_len = linear[i].n_in;
        for (int k = 0; k < delta_len; ++k) delta[k] = tmp[k];
    }

    // delta now holds grad h(x) (length linear[0].n_in)
    for (int k = 0; k < linear[0].n_in; ++k) out_grad[k] = delta[k];

    delete[] delta;
    delete[] tmp;
}

// Full reverse-mode CBF constraint assembly:
//     h, L_f h, L_G h  <-  forward + backward + inner products
//
//   x  : state, length n
//   f  : drift, length n
//   G  : input field, ROW-MAJOR shape (n x m), index G[i*m + j]
//   m  : number of control inputs
inline void evaluate_cbf(
    const LinearSpec* linear,
    const ActivationSpec* activations,
    int depth,
    const float* x,
    const float* f,
    const float* G,
    int m,
    float* h_out,
    float* Lf_out,
    float* Lg_out)
{
    ActivationCache cache = ActivationCache::allocate(linear, depth);
    *h_out = forward_with_cache(linear, activations, depth, x, cache);

    int n = linear[0].n_in;
    float* grad = new float[n];
    backward(linear, activations, depth, cache, grad);

    // grad^T f
    float lf = 0.0f;
    for (int k = 0; k < n; ++k) lf += grad[k] * f[k];
    *Lf_out = lf;

    // grad^T G column by column
    for (int j = 0; j < m; ++j) {
        float lg = 0.0f;
        for (int k = 0; k < n; ++k) lg += grad[k] * G[k * m + j];
        Lg_out[j] = lg;
    }

    delete[] grad;
    cache.free();
}

}  // namespace reverse_mode_ad
