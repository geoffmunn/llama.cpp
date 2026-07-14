#pragma once

// Backend CPU accumulation helpers for HIFI/LITE quantization types
//
// Provides macro-generated vec_dot routines that dequantize a HIFI/LITE block
// to float, dequantize the matching Q8_K block to float, and accumulate the
// float dot product across all blocks in the vector.

#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------
// Generic HIFI/LITE vec_dot generator
// -----------------------------------------------------------------------
// Expands to: void ggml_vec_dot_<name>_q8_K(int n, float *s, ...)
//
// Pattern: dequantize one HIFI/LITE block + one Q8_K block to float[256],
// accumulate the float dot product, repeat for all blocks.
// All HIFI/LITE types share blck_size == 256 (QK_K).
// -----------------------------------------------------------------------

#define HIFI_VEC_DOT_Q8K(name, block_type, deq_fn)            \
void ggml_vec_dot_##name##_q8_K(int n,                        \
                                float * GGML_RESTRICT s,     \
                                size_t bs,                    \
                                const void * GGML_RESTRICT vx, \
                                size_t bx,                    \
                                const void * GGML_RESTRICT vy, \
                                size_t by,                    \
                                int nrc) {                    \
    assert(n % QK_K == 0);                                    \
    const int nb = n / QK_K;                                  \
    const block_type * bx_hifi = (const block_type *)vx;      \
    const block_q8_K * by_q8  = (const block_q8_K *)vy;       \
    float sumf = 0.0f;                                        \
    float tmp_x[QK_K];                                        \
    float tmp_y[QK_K];                                        \
    for (int i = 0; i < nb; i++) {                            \
        deq_fn(&bx_hifi[i], tmp_x, QK_K);                     \
        dequantize_row_q8_K(&by_q8[i], tmp_y);                \
        for (int j = 0; j < QK_K; j++) {                      \
            sumf += tmp_x[j] * tmp_y[j];                      \
        }                                                     \
    }                                                         \
    *s = sumf;                                                \
}

#ifdef __cplusplus
}
#endif
