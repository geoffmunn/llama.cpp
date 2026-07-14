#include "ggml/include/gguf.h"
#include <math.h>
#include <float.h>

#define Q3_K_LITE_BLOCK_SIZE    256
#define Q3_K_LITE_MAX_RESIDUALS 8

typedef struct {
    // Q2_K base (84 bytes)
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    GGML_EXTENSION union { ... } GGML_COMMON_AGGR_U;
    // INT8 extension (20 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q3_K_LITE_MAX_RESIDUALS];  // 8 bytes
    int8_t    residual_vals[Q3_K_LITE_MAX_RESIDUALS]; // 8 bytes
    uint8_t   _pad;
    ggml_half residual_scale;                          // 2 bytes
} block_q3_k_lite;
// 84 + 20 = 104 bytes
static_assert(sizeof(block_q3_k_lite) == 104, "wrong q3_k_lite block size/padding");

#define Q4_K_LITE_BLOCK_SIZE    256
#define Q4_K_LITE_MAX_RESIDUALS 7

typedef struct {
    // Q3_K base (110 bytes): hmask[32] + qs[64] + scales[12] + d[2]
    uint8_t   hmask[QK_K/8];
    uint8_t   qs[QK_K/4];
    uint8_t   scales[K_SCALE_SIZE];
    ggml_half d;
    // INT8 extension (18 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q4_K_LITE_MAX_RESIDUALS];  // 7 bytes
    int8_t    residual_vals[Q4_K_LITE_MAX_RESIDUALS]; // 7 bytes
    uint8_t   _pad;
    ggml_half residual_scale;                          // 2 bytes
} block_q4_k_lite;
// 110 + 18 = 128 bytes
static_assert(sizeof(block_q4_k_lite) == 128, "wrong q4_k_lite block size/padding");

// -----------------------------------------------------------------------
// HiFi importance computation
// -----------------------------------------------------------------------

float ggml_hifi_compute_tensor_importance(const float * imatrix_data, int64_t n_elements) {
    if (imatrix_data == NULL || n_elements <= 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < n_elements; i++) {
        sum += fabsf(imatrix_data[i]);
    }

    float mean = sum / (float) n_elements;

    // Normalize to [0.0, 1.0] using a sigmoid-like scaling
    // Typical imatrix values are small; scale so that mean~1 maps to ~0.5
    return 1.0f / (1.0f + expf(-mean));
}
