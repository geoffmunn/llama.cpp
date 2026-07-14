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

/* -----------------------------------------------------------------
 *  Outlier count defaults per size category
 *
 *  3-sigma rule: weights beyond 3 standard deviations from the mean
 *  are outlier candidates.  The base counts below are tuned so that
 *  roughly the top-N by |weight| x importance survive the filter.
 * -----------------------------------------------------------------
 */

static int ggml_q3_hifi_outlier_count_for_size(ggml_q3_hifi_size_category cat) {
    switch (cat) {
        case Q3_HIFI_SIZE_TINY:   return 2;  // small models need fewer outliers
        case Q3_HIFI_SIZE_MEDIUM: return 4;  // sweet spot
        default:
        case Q3_HIFI_SIZE_LARGE:  return 8;  // large models preserve more
    }
}

int ggml_q3_hifi_get_tensor_outliers(void) {
    /* If the caller explicitly set an outlier count, honour it. */
    if (g_tls_tensor_outliers > 0) {
        return g_tls_tensor_outliers;
    }

    /* Derive a sensible default from model size and importance.
     *
     * Selection rules (guide §5.2):
     *   1. 3- sigma rule identifies candidate positions.
     *   2. Magnitude ranking by |weight| x importance orders them.
     *   3. N = base count scaled by importance, capped at max.
     */
    float model_params_b = 0.0f;
    const ggml_hifi_quant_context * ctx = ggml_hifi_get_context();
    if (ctx != NULL) {
        model_params_b = ctx->model_params_b;
    }

    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    int base = ggml_q3_hifi_outlier_count_for_size(cat);

    /* Scale by tensor importance (0..1).  Low-importance tensors get
     * fewer outliers; high-importance tensors get the full budget. */
    float imp = g_tls_tensor_importance;
    if (imp <= 0.0f) {
        imp = 0.5f;  // neutral default when importance unknown
    }
    int n = (int)((float)base * imp);
    if (n < 1) n = 1;
    if (n > Q3_K_HIFI_MAX_OUTLIERS) n = Q3_K_HIFI_MAX_OUTLIERS;

    return n;
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

// test edit
int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                   int max_residuals) {
    float importance_clamped = tensor_importance < 0.0f ? 0.0f :
                              tensor_importance > 1.0f ? 1.0f : tensor_importance;
    float size_factor = model_params_b > 0.0f ? (1.0f / (1.0f + model_params_b)) : 1.0f;
    int budget = (int)(importance_clamped * size_factor * (float)max_residuals);
    return budget < 0 ? 0 : (budget > max_residuals ? max_residuals : budget);
}

/* ----------------------------------------------------------------- */
/*  INT8 Residual Correction — Step 3: compute per-element errors     */
/* -----------------------------------------------------------------
 *
 *  After quantizing a block with the base Q6_K function and immediately
 *  dequantizing it, compute the per-element error:
 *
 *      err[i] = x[i] - base_decoded[i]
 *
 *  These errors are used by later steps to select top-N positions and
 *  quantize INT8 residual corrections.
 * ----------------------------------------------------------------- */

void ggml_q6_k_hifi_res8_compute_errors(const float * GGML_RESTRICT x,
                                        const float * GGML_RESTRICT base_decoded,
                                        float * GGML_RESTRICT err,
                                        int64_t k) {
    assert(k % QK_K == 0);
    for (int64_t i = 0; i < k; ++i) {
        err[i] = x[i] - base_decoded[i];
    }
}

/* ----------------------------------------------------------------- */
/*  INT8 Residual Correction — Step 4: select top-N positions         */
/* -----------------------------------------------------------------
 *
 *  Select the top-N positions by weighted residual magnitude:
 *
 *      score[i] = |err[i]| * imatrix_importance[i]
 *
 *  Uses a partial sort (selection into the N largest slots) to avoid
 *  sorting all k elements.
 * ----------------------------------------------------------------- */

void ggml_q6_k_hifi_res8_select_top_n(const float * GGML_RESTRICT err,
                                      const float * GGML_RESTRICT imatrix_importance,
                                      int64_t k,
                                      int max_outliers,
                                      uint8_t * GGML_RESTRICT outlier_idx,
                                      int * GGML_RESTRICT actual_count) {
    assert(max_outliers > 0);
    assert(actual_count != NULL);

    /* Build candidate list with weighted scores */
    struct {
        float score;
        int   idx;
    } candidates[(int)k];

    for (int64_t j = 0; j < k; ++j) {
        candidates[j].idx = (int)j;
        candidates[j].score = fabsf(err[j]) * imatrix_importance[j];
    }

    /* Partial sort: bubble the top max_outliers to the front */
    for (int o = 0; o < max_outliers; ++o) {
        for (int64_t j = o + 1; j < k; ++j) {
            if (candidates[j].score > candidates[o].score) {
                float tmp_score = candidates[o].score;
                candidates[o].score = candidates[j].score;
                candidates[j].score = tmp_score;
                int tmp_idx = candidates[o].idx;
                candidates[o].idx = candidates[j].idx;
                candidates[j].idx = tmp_idx;
            }
        }
    }

    /* Collect results — skip zero-score entries */
    int count = 0;
    for (int o = 0; o < max_outliers; ++o) {
        if (candidates[o].score > 0.0f) {
            outlier_idx[count++] = (uint8_t)candidates[o].idx;
        }
    }
    *actual_count = count;
}
