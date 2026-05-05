#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Thread-local context passed into HIFI quantization functions.
typedef struct {
    int   outlier_count;    // 1-8; number of outliers/residuals to preserve
    float layer_importance; // 0.0-1.0; from imatrix aggregation
    int   layer_idx;        // current layer (debugging)
    int   total_layers;     // total model layers (debugging)
    int   is_active;        // 1 = adaptive mode on
    float model_params_b;   // model size in billions
} ggml_hifi_quant_context;

GGML_API const ggml_hifi_quant_context * ggml_hifi_get_context(void);
GGML_API void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx);

GGML_API int   ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                                float layer_importance, float model_params_b);
GGML_API float ggml_hifi_compute_tensor_importance(const float * imatrix_data, int64_t n_elements);
GGML_API float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size);
GGML_API int   ggml_hifi_compute_block_outlier_count(float block_importance,
                                                      int base_outlier_count, float model_params_b);

// Q3_K_HIFI model-size classification
typedef enum {
    Q3_HIFI_SIZE_TINY   = 0,  // <=1.7B
    Q3_HIFI_SIZE_MEDIUM = 1,  // 2B-8B (sweet spot)
    Q3_HIFI_SIZE_LARGE  = 2,  // 14B+
} ggml_q3_hifi_size_category;

GGML_API ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b);
GGML_API int   ggml_q3_hifi_get_max_outliers(float model_params_b);
GGML_API float ggml_q3_hifi_get_outlier_threshold(float model_params_b);
GGML_API float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n);
GGML_API int   ggml_q3_hifi_should_enhance_tensor(const char * tensor_name,
                                                    const float * weights, int64_t n_elements,
                                                    float model_params_b,
                                                    int * enhanced_count, int max_enhanced);
GGML_API int   ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding);
GGML_API float ggml_q3_hifi_get_attn_v_threshold(float model_params_b);

// TLS per-tensor outlier control (reused by Q4_K_HIFI)
GGML_API void  ggml_q3_hifi_set_tensor_outliers(int outliers);
GGML_API int   ggml_q3_hifi_get_tensor_outliers(void);
GGML_API void  ggml_q3_hifi_set_tensor_importance(float importance);
GGML_API float ggml_q3_hifi_get_tensor_importance(void);
GGML_API void  ggml_q3_hifi_reset_tensor_state(void);
GGML_API int   ggml_q3_hifi_compute_block_outliers(float block_outlier_ratio,
                                                    int base_outlier_count, float model_params_b);

// Q4_K_HIFI
GGML_API int ggml_q4_hifi_get_max_outliers(float model_params_b);

// LITE tier-based residual budget
GGML_API int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                            int max_residuals);

#ifdef __cplusplus
}
#endif
