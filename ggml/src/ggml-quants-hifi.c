// SPDX-License-Identifier: MIT

/**
 * @file ggml-quants-hifi.c
 * @brief Layer-adaptive (HiFi) quantization context and utility API.
 */

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-quants-hifi.h"

/* ----------------------------------------------------------------- */
/*  Thread-local per-tensor outlier state                            */
/* ----------------------------------------------------------------- */

static thread_local int   g_tls_tensor_outliers   = 0;
static thread_local float g_tls_tensor_importance = 0.0f;

void ggml_q3_hifi_set_tensor_outliers(int outliers) {
    g_tls_tensor_outliers = outliers;
}

int ggml_q3_hifi_get_tensor_outliers(void) {
    return g_tls_tensor_outliers;
}

void ggml_q3_hifi_set_tensor_importance(float importance) {
    g_tls_tensor_importance = importance;
}

float ggml_q3_hifi_get_tensor_importance(void) {
    return g_tls_tensor_importance;
}

void ggml_q3_hifi_reset_tensor_state(void) {
    g_tls_tensor_outliers   = 0;
    g_tls_tensor_importance = 0.0f;
}

/* ----------------------------------------------------------------- */
/*  Q3_K_HIFI model-size classification                              */
/* ----------------------------------------------------------------- */

ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b) {
    if (model_params_b <= 1.75f) {
        return Q3_HIFI_SIZE_TINY;
    } else if (model_params_b <= 8.5f) {
        return Q3_HIFI_SIZE_MEDIUM;
    } else {
        return Q3_HIFI_SIZE_LARGE;
    }
}

/* ----------------------------------------------------------------- */
/*  Block-level outlier helpers                                      */
/* ----------------------------------------------------------------- */

int ggml_q3_hifi_compute_block_outliers(float block_outlier_ratio,
                                         int base_outlier_count, float model_params_b) {
    int extra = (int)(block_outlier_ratio * (float)Q3_K_HIFI_MAX_OUTLIERS);
    int total = base_outlier_count + extra;
    return total > Q3_K_HIFI_MAX_OUTLIERS ? Q3_K_HIFI_MAX_OUTLIERS : total;
}

/* ----------------------------------------------------------------- */
/*  Q4_K_HIFI helpers                                                */
/* ----------------------------------------------------------------- */

int ggml_q4_hifi_get_max_outliers(float model_params_b) {
    (void)model_params_b;
    return Q4_K_HIFI_MAX_OUTLIERS;
}

/* ----------------------------------------------------------------- */
/*  K_LITE tier-based residual budget                                */
/* ----------------------------------------------------------------- */

int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                   int max_residuals) {
    float importance_clamped = tensor_importance < 0.0f ? 0.0f :
                              tensor_importance > 1.0f ? 1.0f : tensor_importance;
    float size_factor = model_params_b > 0.0f ? (1.0f / (1.0f + model_params_b)) : 1.0f;
    int budget = (int)(importance_clamped * size_factor * (float)max_residuals);
    return budget < 0 ? 0 : (budget > max_residuals ? max_residuals : budget);
}
