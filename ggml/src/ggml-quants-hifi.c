#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-quants-hifi.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// =============================================================================
// Thread-local HIFI quantization context
// =============================================================================

#if defined(_MSC_VER)
static __declspec(thread) const ggml_hifi_quant_context * g_hifi_ctx = NULL;
static __declspec(thread) int   g_tensor_outliers  = 0;
static __declspec(thread) float g_tensor_importance = 0.0f;
#else
static _Thread_local const ggml_hifi_quant_context * g_hifi_ctx = NULL;
static _Thread_local int   g_tensor_outliers   = 0;
static _Thread_local float g_tensor_importance = 0.0f;
#endif

const ggml_hifi_quant_context * ggml_hifi_get_context(void) {
    return g_hifi_ctx;
}

void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx) {
    g_hifi_ctx = ctx;
}

void ggml_q3_hifi_set_tensor_outliers(int outliers) {
    g_tensor_outliers = outliers;
}

int ggml_q3_hifi_get_tensor_outliers(void) {
    return g_tensor_outliers;
}

void ggml_q3_hifi_set_tensor_importance(float importance) {
    g_tensor_importance = importance;
}

float ggml_q3_hifi_get_tensor_importance(void) {
    return g_tensor_importance;
}

void ggml_q3_hifi_reset_tensor_state(void) {
    g_tensor_outliers   = 0;
    g_tensor_importance = 0.0f;
}

// =============================================================================
// Model-size classification and outlier count helpers
// =============================================================================

ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b) {
    if (model_params_b <= 1.7f) return Q3_HIFI_SIZE_TINY;
    if (model_params_b <= 8.0f) return Q3_HIFI_SIZE_MEDIUM;
    return Q3_HIFI_SIZE_LARGE;
}

int ggml_q3_hifi_get_max_outliers(float model_params_b) {
    switch (ggml_q3_hifi_get_size_category(model_params_b)) {
        case Q3_HIFI_SIZE_TINY:   return 2;
        case Q3_HIFI_SIZE_MEDIUM: return 8;
        case Q3_HIFI_SIZE_LARGE:  return 6;
        default:                  return 8;
    }
}

float ggml_q3_hifi_get_outlier_threshold(float model_params_b) {
    if (model_params_b <= 1.7f) return 2.5f;
    if (model_params_b <= 5.0f) return 2.0f;
    return 1.8f;
}

float ggml_q3_hifi_get_attn_v_threshold(float model_params_b) {
    if (model_params_b <= 1.0f)  return 0.0f;
    if (model_params_b <= 1.7f)  return 0.0f;
    if (model_params_b <= 5.0f)  return 0.25f;
    if (model_params_b <= 10.0f) return 0.15f;
    if (model_params_b <= 20.0f) return 0.08f;
    return 0.05f;
}

int ggml_q4_hifi_get_max_outliers(float model_params_b) {
    if (model_params_b <= 2.0f) return 4;
    if (model_params_b <= 7.0f) return 6;
    return 8;
}

// =============================================================================
// Tensor importance computation
// =============================================================================

float ggml_hifi_compute_tensor_importance(const float * imatrix_data, int64_t n_elements) {
    if (!imatrix_data || n_elements <= 0) return 0.0f;
    double sum   = 0.0;
    double sumsq = 0.0;
    for (int64_t i = 0; i < n_elements; i++) {
        sum   += (double)imatrix_data[i];
        sumsq += (double)imatrix_data[i] * (double)imatrix_data[i];
    }
    double mean = sum / (double)n_elements;
    double var  = sumsq / (double)n_elements - mean * mean;
    return (var > 0.0) ? (float)(mean / sqrt(var + 1e-10)) : 0.0f;
}

float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size) {
    if (!imatrix_block || block_size <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < block_size; i++) {
        sum += (double)imatrix_block[i];
    }
    return (float)(sum / (double)block_size);
}

// =============================================================================
// Adaptive outlier count computation
// =============================================================================

int ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                     float layer_importance, float model_params_b) {
    (void)model_params_b;
    if (total_layers <= 0) return Q6_K_HIFI_RES8_MAX_OUTLIERS;
    float frac = (float)layer_idx / (float)total_layers;
    int base = (frac < 0.3f) ? Q6_K_HIFI_RES8_MAX_OUTLIERS :
               (frac < 0.6f) ? Q6_K_HIFI_RES8_MAX_OUTLIERS - 2 :
                               Q6_K_HIFI_RES8_MAX_OUTLIERS - 4;
    if (layer_importance > 0.5f) base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    return base < 1 ? 1 : base;
}

int ggml_hifi_compute_block_outlier_count(float block_importance,
                                           int base_outlier_count, float model_params_b) {
    (void)model_params_b;
    if (block_importance > 0.8f) return base_outlier_count;
    if (block_importance > 0.5f) return base_outlier_count > 2 ? base_outlier_count - 1 : base_outlier_count;
    return base_outlier_count > 2 ? base_outlier_count - 2 : base_outlier_count;
}

float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n) {
    if (!weights || n <= 0) return 0.0f;
    double sum   = 0.0;
    double sumsq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double v = (double)weights[i];
        sum   += v < 0 ? -v : v;
        sumsq += v * v;
    }
    double mean_abs = sum / (double)n;
    double rms      = sqrt(sumsq / (double)n + 1e-20);
    return (mean_abs > 0.0) ? (float)(rms / mean_abs) : 0.0f;
}

int ggml_q3_hifi_compute_block_outliers(float block_outlier_ratio,
                                         int base_outlier_count, float model_params_b) {
    (void)model_params_b;
    if (block_outlier_ratio > 2.0f) return base_outlier_count;
    if (block_outlier_ratio > 1.5f) return base_outlier_count > 1 ? base_outlier_count - 1 : base_outlier_count;
    return base_outlier_count > 2 ? base_outlier_count - 2 : base_outlier_count;
}

// =============================================================================
// Tensor enhancement helpers
// =============================================================================

int ggml_q3_hifi_should_enhance_tensor(const char * tensor_name,
                                        const float * weights, int64_t n_elements,
                                        float model_params_b,
                                        int * enhanced_count, int max_enhanced) {
    (void)weights;
    (void)n_elements;
    if (model_params_b <= 1.7f) return 0;
    if (*enhanced_count >= max_enhanced) return 0;
    if (!tensor_name) return 0;
    if (strstr(tensor_name, "attn_v") || strstr(tensor_name, "v_proj") ||
        strstr(tensor_name, "attn.v")  || strstr(tensor_name, ".v.weight")) {
        (*enhanced_count)++;
        return 1;
    }
    return 0;
}

int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding) {
    (void)is_embedding;
    return (model_params_b > 1.7f) ? 1 : 0;
}

// =============================================================================
// LITE residual budget
// =============================================================================

int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                   int max_residuals) {
    (void)model_params_b;
    if (tensor_importance > 0.8f) return max_residuals;
    if (tensor_importance > 0.5f) return max_residuals > 1 ? max_residuals - 1 : max_residuals;
    return max_residuals > 2 ? max_residuals / 2 : max_residuals;
}
