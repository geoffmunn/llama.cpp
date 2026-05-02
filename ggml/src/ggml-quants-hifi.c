#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-quants-hifi.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

// Thread-local HIFI quantization context
#if defined(_MSC_VER)
static __declspec(thread) const ggml_hifi_quant_context * g_hifi_ctx = NULL;
static __declspec(thread) int   g_tensor_outliers    = 0;
static __declspec(thread) float g_tensor_importance  = 0.0f;
#else
static _Thread_local const ggml_hifi_quant_context * g_hifi_ctx       = NULL;
static _Thread_local int                             g_tensor_outliers = 0;
static _Thread_local float                           g_tensor_importance = 0.0f;
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

ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b) {
    if (model_params_b <= 1.7f) return Q3_HIFI_SIZE_TINY;
    if (model_params_b <= 8.0f) return Q3_HIFI_SIZE_MEDIUM;
    return Q3_HIFI_SIZE_LARGE;
}

int ggml_q3_hifi_get_max_outliers(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    switch (cat) {
        case Q3_HIFI_SIZE_TINY:   return 2;
        case Q3_HIFI_SIZE_MEDIUM: return 8;
        case Q3_HIFI_SIZE_LARGE:  return 6;
        default:                  return 8;
    }
}

float ggml_q3_hifi_get_outlier_threshold(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    switch (cat) {
        case Q3_HIFI_SIZE_TINY:   return 3.5f;
        case Q3_HIFI_SIZE_MEDIUM: return 2.5f;
        case Q3_HIFI_SIZE_LARGE:  return 2.0f;
        default:                  return 2.5f;
    }
}

float ggml_q3_hifi_get_attn_v_threshold(float model_params_b) {
    if (model_params_b <= 1.0f)  return 0.0f;
    if (model_params_b <= 1.7f)  return 0.0f;
    if (model_params_b <= 5.0f)  return 0.25f;
    if (model_params_b <= 10.0f) return 0.15f;
    if (model_params_b <= 20.0f) return 0.08f;
    return 0.05f;
}

float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n) {
    if (n <= 0) return 0.0f;

    double sum   = 0.0;
    double sumsq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sum   += (double)weights[i];
        sumsq += (double)weights[i] * (double)weights[i];
    }
    double mean    = sum / (double)n;
    double var     = sumsq / (double)n - mean * mean;
    double std_dev = (var > 0.0) ? sqrt(var) : 1e-8;
    double thresh  = 2.5 * std_dev;

    int outliers = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabs((double)weights[i] - mean) > thresh) outliers++;
    }
    return (float)outliers / (float)n;
}

int ggml_q3_hifi_should_enhance_tensor(const char * tensor_name,
                                        const float * weights, int64_t n_elements,
                                        float model_params_b,
                                        int * enhanced_count, int max_enhanced) {
    (void)tensor_name;
    (void)weights;
    (void)n_elements;
    (void)model_params_b;
    (void)enhanced_count;
    (void)max_enhanced;
    return 0;
}

int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding) {
    (void)model_params_b;
    (void)is_embedding;
    return 0;
}

int ggml_q3_hifi_compute_block_outliers(float block_outlier_ratio,
                                         int base_outlier_count, float model_params_b) {
    (void)model_params_b;
    int n = base_outlier_count;
    if (block_outlier_ratio > 0.05f) n++;
    if (block_outlier_ratio > 0.10f) n++;
    if (n > Q3_K_HIFI_MAX_OUTLIERS) n = Q3_K_HIFI_MAX_OUTLIERS;
    return n;
}

// Layer-adaptive outlier count for RES8 types
int ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                     float layer_importance, float model_params_b) {
    (void)model_params_b;
    if (total_layers <= 0) return Q6_K_HIFI_RES8_MAX_OUTLIERS;

    int base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    // Early layers (first 20%) get max outliers
    float frac = (float)layer_idx / (float)total_layers;
    if (frac < 0.20f) {
        base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    } else if (frac < 0.80f) {
        base = Q6_K_HIFI_RES8_MAX_OUTLIERS / 2;
    } else {
        base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    }

    // Boost for high importance tensors
    if (layer_importance > 0.7f) {
        base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    }

    return base;
}

float ggml_hifi_compute_tensor_importance(const float * imatrix_data, int64_t n_elements) {
    if (!imatrix_data || n_elements <= 0) return 0.0f;

    double sum = 0.0;
    double max_val = 0.0;
    for (int64_t i = 0; i < n_elements; i++) {
        double v = fabs((double)imatrix_data[i]);
        sum += v;
        if (v > max_val) max_val = v;
    }
    if (max_val < 1e-10) return 0.0f;
    double avg = sum / (double)n_elements;
    return (float)(avg / max_val);
}

float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size) {
    return ggml_hifi_compute_tensor_importance(imatrix_block, (int64_t)block_size);
}

int ggml_hifi_compute_block_outlier_count(float block_importance,
                                           int base_outlier_count, float model_params_b) {
    (void)model_params_b;
    int n = base_outlier_count;
    if (block_importance > 0.8f) n++;
    if (n > Q6_K_HIFI_RES8_MAX_OUTLIERS) n = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    return n;
}

int ggml_q4_hifi_get_max_outliers(float model_params_b) {
    if (model_params_b <= 2.0f) return 4;
    if (model_params_b <= 7.0f) return 6;
    return Q4_K_HIFI_MAX_OUTLIERS;
}

int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                   int max_residuals) {
    (void)model_params_b;
    if (tensor_importance > 0.8f) return max_residuals;
    if (tensor_importance > 0.5f) return (max_residuals * 3) / 4;
    return max_residuals / 2;
}
