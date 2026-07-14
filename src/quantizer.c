// SPDX-License-Identifier: MIT

/**
 * @file quantizer.c
 * @brief Public API wrappers for HIFI quantization types.
 */

#define GGML_COMMON_DECL_C
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

/* ----------------------------------------------------------------- */
/*  Q6_K_HIFI_DYNAMIC                                                */
/* ----------------------------------------------------------------- */

size_t quantize_q6_k_hifi_dynamic(const float * src, void * dst, int64_t nrows,
                                   int64_t n_per_row, const float * imatrix) {
    size_t row_size = n_per_row / QK_K * sizeof(block_q6_k_hifi_dynamic);
    if (!imatrix) {
        quantize_row_q6_k_hifi_dynamic_ref(src, (block_q6_k_hifi_dynamic *)dst, nrows * n_per_row);
    } else {
        // TODO: imatrix-guided quantization
        quantize_row_q6_k_hifi_dynamic_ref(src, (block_q6_k_hifi_dynamic *)dst, nrows * n_per_row);
    }
    return nrows * row_size;
}
