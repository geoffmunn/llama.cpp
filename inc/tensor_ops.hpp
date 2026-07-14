#pragma once

#include <stddef.h>
#include <stdint.h>

// Vector dot-product kernels for HIFI outlier-corrected formats.

/// Full-precision dequant-then-dot with per-outlier FP16 correction.
/// Dequantises each block_q4_k_hifi to float, dots against block_q8_K,
/// then replaces every outlier position by adding the difference
/// (true_FP16_value - q4k_decoded_value) * q8 * dy.
void ggml_vec_dot_q4_k_hifi_q8_K(
    int n,
    float *s,
    const void *vx,
    const void *vy,
    const int *n_left,
    const int *n_mat,
    const int *n_rows,
    const int *n_cols);
