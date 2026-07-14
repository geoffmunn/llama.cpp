// BY_Q8 accessor / descriptor utilities
//
// Replaces outlier_count-based loop bounds with an explicit FP16-zero
// sentinel check.  This matches the pattern already used in
// ggml_vec_dot_q3_k_hifi_q8_K (quants.c) and removes the need to
// track a separate count field at dequantize time.

#include "llama-kernel.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

// -----------------------------------------------------------------
// FP16 <-> FP32 conversion helpers (minimal subset)
// -----------------------------------------------------------------

#ifndef GGML_FP16_TO_FP32
#define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

static float ggml_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 0x0001;
    uint32_t f = (h >> 10) & 0x001f;
    const uint32_t exp = 0x71 - f;

    if (f == 31) {
        return (h & 0x3ff) ? std::nan("") : (sign ? -std::numeric_limits<float>::infinity()
                                                   :  std::numeric_limits<float>::infinity());
    }
    if (f == 0) {
        return std::ldexp((h & 0x3ff), -24);
    }
    return std::ldexp(static_cast<float>((h & 0x3ff) | 0x400), static_cast<int>(exp) - 10);
}

// -----------------------------------------------------------------
// Sentinel-based outlier iteration (C API)
// -----------------------------------------------------------------

/// Count valid outliers in a block by scanning for the FP16-zero
/// sentinel.  Returns 0 when the first entry is already zero.
int count_outliers_by_sentinel(const ggml_half *vals, int max) {
    int count = 0;
    for (int j = 0; j < max; ++j) {
        if (is_outlier_sentinel(vals[j])) break;
        ++count;
    }
    return count;
}

/// Dequantize a single HIFI block's outlier positions into a float
/// output array, using the FP16-zero sentinel to stop early.
void restore_outliers(float *out,
                      const uint8_t *idx_arr,
                      const ggml_half *val_arr,
                      int max) {
    for (int j = 0; j < max; ++j) {
        if (is_outlier_sentinel(val_arr[j])) break;
        out[idx_arr[j]] = GGML_FP16_TO_FP32(val_arr[j]);
    }
}
