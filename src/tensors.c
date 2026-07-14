// SPDX-License-Identifier: MIT

/**
 * @file tensors.c
 * @brief Tensor metadata utilities for HiFi quantization types.
 *
 * Ensures outlier index arrays are sorted ascending before being stored
 * into tensor metadata structures. Backend kernels (CPU, CUDA, Metal)
 * rely on this ordering for correct dequantization and dot-product
 * computation.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants-hifi.h"

/* ----------------------------------------------------------------- */
/*  Compare helper for qsort (ascending uint8_t)                      */
/* ----------------------------------------------------------------- */

static int compare_uint8(const void *a, const void *b) {
    return (*(const uint8_t *)a - *(const uint8_t *)b);
}

/* ----------------------------------------------------------------- */
/*  Sort outlier indices ascending                                    */
/* ----------------------------------------------------------------- */

/**
 * Sort the outlier index array in ascending order.
 *
 * Several backend kernels assume outlier_idx[] is sorted ascending.
 * Call this immediately prior to storing outliers into any tensor
 * metadata structure (HIFI blocks, LITE blocks, etc.).
 *
 * @param outlier_idx  Array of outlier position indices (uint8_t).
 * @param count        Number of valid entries in the array.
 */
void ggml_sort_outlier_indices(uint8_t *outlier_idx, int count) {
    if (count > 1) {
        qsort(outlier_idx, (size_t)count, sizeof(uint8_t), compare_uint8);
    }
}

/* ----------------------------------------------------------------- */
/*  Store outliers into tensor metadata (sorts first)                 */
/* ----------------------------------------------------------------- */

/**
 * Sort outlier indices ascending, then copy them into the destination
 * block's outlier_idx field for backend kernel compatibility.
 *
 * @param dst_idx  Destination outlier index array inside the block struct.
 * @param src_idx  Source outlier index array (may be unsorted).
 * @param count    Number of valid entries.
 */
void ggml_store_outlier_indices(uint8_t *dst_idx, const uint8_t *src_idx, int count) {
    /* Max outlier count across all HiFi block types is 8 */
    uint8_t tmp[8];
    memcpy(tmp, src_idx, (size_t)count * sizeof(uint8_t));
    ggml_sort_outlier_indices(tmp, count);
    memcpy(dst_idx, tmp, (size_t)count * sizeof(uint8_t));
}
