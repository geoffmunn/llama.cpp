// SPDX-License-Identifier: MIT

/**
 * @file ggml-quants-hifi.h
 * @brief Layer-adaptive (HiFi) quantization context and utility API.
 *
 * Include path rules:
 * ------------------
 * This header lives under ggml/src/ but is built with ggml/include/ as the
 * root include directory.  Therefore the correct way to pull in the core
 * GGML types is:
 *
 *      #include "ggml.h"          // ✅ resolves via ggml/include/
 *
 * and **not**:
 *
 *      #include "ggml/ggml.h"     // ❌ no such path on the compiler search path
 *
 * The same rule applies to every other header in ggml/src/: always use the
 * bare filename (e.g. `"ggml.h"`, `"ggml-alloc.h"`) rather than prefixing
 * with `ggml/`.
 */

#ifndef GGML_QUANTS_HIFI_H
#define GGML_QUANTS_HIFI_H

#include "ggml.h"
#include "ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------- */
/*  Thread-local quantization context                                */
/* ----------------------------------------------------------------- */

/// Context passed into quantization functions for layer-adaptive behaviour
typedef struct {
    int   outlier_count;     // 1–8; number of outliers to preserve
    float layer_importance;  // 0.0–1.0; from imatrix aggregation
    int   layer_idx;         // current layer (debugging)
    int   total_layers;      // total model layers (debugging)
    int   is_active;         // 1 = adaptive mode on
    float model_params_b;    // model size in billions
} ggml_hifi_quant_context;

/// Get the current thread-local HiFi quantization context
GGML_API const ggml_hifi_quant_context * ggml_hifi_get_context(void);

/// Set the thread-local HiFi quantization context
GGML_API void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx);

/* ----------------------------------------------------------------- */
/*  Importance / outlier helpers                                     */
/* ----------------------------------------------------------------- */

/// Compute the number of outliers to preserve for a given layer
GGML_API int ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                             float layer_importance, float model_params_b);

/// Compute tensor-level importance from an importance matrix
GGML_API float ggml_hifi_compute_tensor_importance(const float * imatrix_data, int64_t n_elements);

/// Compute block-level importance from a slice of an importance matrix
GGML_API float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size);

/// Compute the per-block outlier count given block importance
GGML_API int ggml_hifi_compute_block_outlier_count(float block_importance,
                                                   int base_outlier_count, float model_params_b);

/* ----------------------------------------------------------------- */
/*  Q3_K_HIFI model-size classification                              */
/* ----------------------------------------------------------------- */

/// Size category used by Q3_K_HIFI to tune outlier counts
typedef enum {
    Q3_HIFI_SIZE_TINY   = 0,  // ≤1.7B
    Q3_HIFI_SIZE_MEDIUM = 1,  // 2B–8B (sweet spot)
    Q3_HIFI_SIZE_LARGE  = 2,  // 14B+
} ggml_q3_hifi_size_category;

/// Classify a model by parameter count (billions)
GGML_API ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b);

/// Maximum outlier count for the given model size
GGML_API int ggml_q3_hifi_get_max_outliers(float model_params_b);

/// Outlier threshold (fraction) for the given model size
GGML_API float ggml_q3_hifi_get_outlier_threshold(float model_params_b);

/// Compute the ratio of weights that are "outliers"
GGML_API float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n);

/// Decide whether a tensor should use enhanced quantization
GGML_API int ggml_q3_hifi_should_enhance_tensor(const char * tensor_name,
                                                const float * weights, int64_t n_elements,
                                                float model_params_b,
                                                int * enhanced_count, int max_enhanced);

/// Enhancement type selection based on model size
GGML_API int   ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding);

/// Attention V threshold for the given model size
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

// K_LITE tier-based residual budget
GGML_API int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                            int max_residuals);

// INT8 Residual Correction — Step 3: compute per-element errors
GGML_API void ggml_q6_k_hifi_res8_compute_errors(const float * GGML_RESTRICT x,
                                                  const float * GGML_RESTRICT base_decoded,
                                                  float * GGML_RESTRICT err,
                                                  int64_t k);

// INT8 Residual Correction — Step 4: select top-N positions by |err[i]| * imatrix_importance[i]
GGML_API void ggml_q6_k_hifi_res8_select_top_n(const float * GGML_RESTRICT err,
                                                const float * GGML_RESTRICT imatrix_importance,
                                                int64_t k,
                                                int max_outliers,
                                                uint8_t * GGML_RESTRICT outlier_idx,
                                                int * GGML_RESTRICT actual_count);

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_HIFI_H
