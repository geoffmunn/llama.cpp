// Required: activates GGML_COMMON_DECL guard so constants such as
// Q3_K_HIFI_MAX_OUTLIERS are visible.  Must appear before any other includes.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-quants-hifi.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Thread-local HIFI quantization context
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Layer-adaptive outlier count
// ---------------------------------------------------------------------------

int ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                    float layer_importance, float model_params_b) {
    (void)model_params_b;
    if (total_layers <= 0) return Q6_K_HIFI_RES8_MAX_OUTLIERS;

    float layer_frac = (float)layer_idx / (float)total_layers;

    // Early layers and late layers get more outliers
    int base = 4;
    if (layer_frac < 0.1f || layer_frac > 0.9f) base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    else if (layer_frac < 0.2f || layer_frac > 0.8f) base = 6;

    // Boost by importance
    if (layer_importance > 0.8f) base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    else if (layer_importance > 0.6f && base < 6) base = 6;

    if (base > Q6_K_HIFI_RES8_MAX_OUTLIERS) base = Q6_K_HIFI_RES8_MAX_OUTLIERS;
    return base;
}

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
    if (var < 0.0) var = 0.0;
    double std = sqrt(var);
    if (mean <= 0.0) return 0.0f;
    return (float)(std / mean);  // coefficient of variation as proxy for importance
}

float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size) {
    return ggml_hifi_compute_tensor_importance(imatrix_block, (int64_t)block_size);
}

int ggml_hifi_compute_block_outlier_count(float block_importance,
                                           int base_outlier_count, float model_params_b) {
    (void)model_params_b;
    if (block_importance > 0.8f) return base_outlier_count;
    if (block_importance > 0.5f) return (base_outlier_count > 1) ? base_outlier_count - 1 : 1;
    return (base_outlier_count > 2) ? base_outlier_count - 2 : 1;
}

// ---------------------------------------------------------------------------
// Q3_K_HIFI model-size helpers
// ---------------------------------------------------------------------------

ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b) {
    if (model_params_b <= 1.7f) return Q3_HIFI_SIZE_TINY;
    if (model_params_b <= 8.0f) return Q3_HIFI_SIZE_MEDIUM;
    return Q3_HIFI_SIZE_LARGE;
}

int ggml_q3_hifi_get_max_outliers(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    switch (cat) {
        case Q3_HIFI_SIZE_TINY:   return 4;
        case Q3_HIFI_SIZE_MEDIUM: return Q3_K_HIFI_MAX_OUTLIERS;
        case Q3_HIFI_SIZE_LARGE:  return 6;
    }
    return Q3_K_HIFI_MAX_OUTLIERS;
}

float ggml_q3_hifi_get_outlier_threshold(float model_params_b) {
    // Threshold (in sigma) above which a weight is considered an outlier
    if (model_params_b <= 1.7f) return 3.5f;
    if (model_params_b <= 5.0f) return 2.5f;
    return 2.0f;
}

float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n) {
    if (!weights || n <= 0) return 0.0f;

    double sum   = 0.0;
    double sumsq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double w = (double)weights[i];
        sum   += w;
        sumsq += w * w;
    }
    double mean = sum / (double)n;
    double var  = sumsq / (double)n - mean * mean;
    if (var < 0.0) var = 0.0;
    double std_dev = sqrt(var);
    if (std_dev < 1e-8) return 0.0f;

    int64_t count = 0;
    double threshold = fabs(mean) + 3.0 * std_dev;
    for (int64_t i = 0; i < n; i++) {
        if (fabs((double)weights[i]) > threshold) count++;
    }
    return (float)count / (float)n;
}

int ggml_q3_hifi_should_enhance_tensor(const char * tensor_name,
                                        const float * weights, int64_t n_elements,
                                        float model_params_b,
                                        int * enhanced_count, int max_enhanced) {
    (void)tensor_name;
    if (*enhanced_count >= max_enhanced) return 0;
    float ratio = ggml_q3_hifi_compute_outlier_ratio(weights, n_elements);
    float threshold = ggml_q3_hifi_get_outlier_threshold(model_params_b) / 100.0f;
    if (ratio > threshold) {
        (*enhanced_count)++;
        return 1;
    }
    return 0;
}

int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding) {
    (void)model_params_b;
    (void)is_embedding;
    return 1;  // 1 = use HIFI enhancement
}

float ggml_q3_hifi_get_attn_v_threshold(float model_params_b) {
    if (model_params_b <= 1.0f)  return 0.0f;
    if (model_params_b <= 1.7f)  return 0.0f;
    if (model_params_b <= 5.0f)  return 0.25f;
    if (model_params_b <= 10.0f) return 0.15f;
    if (model_params_b <= 20.0f) return 0.08f;
    return 0.05f;
}

// TLS per-tensor outlier control
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

int ggml_q3_hifi_compute_block_outliers(float block_outlier_ratio,
                                         int base_outlier_count, float model_params_b) {
    (void)model_params_b;
    if (block_outlier_ratio > 0.05f) return base_outlier_count;
    if (block_outlier_ratio > 0.02f) return (base_outlier_count > 1) ? base_outlier_count - 1 : 1;
    return (base_outlier_count > 2) ? base_outlier_count - 2 : 1;
}

// Q4_K_HIFI
int ggml_q4_hifi_get_max_outliers(float model_params_b) {
    if (model_params_b <= 2.0f) return 4;
    if (model_params_b <= 8.0f) return Q4_K_HIFI_MAX_OUTLIERS;
    return 6;
}

// LITE residual budget
int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                   int max_residuals) {
    (void)model_params_b;
    if (tensor_importance > 0.8f) return max_residuals;
    if (tensor_importance > 0.5f) return (max_residuals * 3) / 4;
    if (tensor_importance > 0.2f) return max_residuals / 2;
    return max_residuals / 4;
}
