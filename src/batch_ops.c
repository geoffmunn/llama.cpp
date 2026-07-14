// SPDX-License-Identifier: MIT

/**
 * @file batch_ops.c
 * @brief Batched quantization wrappers for HIFI types.
 */

#define GGML_COMMON_DECL_C
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

/* ----------------------------------------------------------------- */
/*  Q2_K_HIFI                                                        */
/* ----------------------------------------------------------------- */

size_t quantize_q2_k_hifi(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix) {
    size_t row_size = n_per_row / Q2_K_HIFI_BLOCK_SIZE * sizeof(block_q2_k_hifi);
    if (!imatrix) {
        quantize_row_q2_k_hifi_ref(src, (block_q2_k_hifi *)dst, nrows * n_per_row);
    } else {
        // TODO: imatrix-guided quantization
        quantize_row_q2_k_hifi_ref(src, (block_q2_k_hifi *)dst, nrows * n_per_row);
    }
    return nrows * row_size;
}
