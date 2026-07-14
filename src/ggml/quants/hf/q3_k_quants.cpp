// SPDX-License-Identifier: MIT

/**
 * @file q3_k_quants.cpp
 * @brief Q3_K_HIFI enhancement type selection.
 */

#include "ggml.h"

/* ----------------------------------------------------------------- */
/*  Enhancement type constants                                       */
/* ----------------------------------------------------------------- */

/// No enhancement (baseline Q3_K behaviour)
#define Q3_HIFI_ENHANCEMENT_NONE      0
/// Light outlier preservation (small models, embeddings)
#define Q3_HIFI_ENHANCEMENT_LIGHT     1
/// Standard outlier + residual correction (medium models)
#define Q3_HIFI_ENHANCEMENT_STANDARD  2
/// Aggressive outlier + full residual correction (large models)
#define Q3_HIFI_ENHANCEMENT_HEAVY     3

/* ----------------------------------------------------------------- */
/*  Enhancement type selection                                       */
/* ----------------------------------------------------------------- */

GGML_API int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding) {
    if (is_embedding) {
        return Q3_HIFI_ENHANCEMENT_LIGHT;
    }
    if (model_params_b <= 1.75f) {
        return Q3_HIFI_ENHANCEMENT_LIGHT;
    } else if (model_params_b <= 8.5f) {
        return Q3_HIFI_ENHANCEMENT_STANDARD;
    } else {
        return Q3_HIFI_ENHANCEMENT_HEAVY;
    }
}

/* ----------------------------------------------------------------- */
/*  Attention V threshold                                            */
/* ----------------------------------------------------------------- */

GGML_API float ggml_q3_hifi_get_attn_v_threshold(float model_params_b) {
    if (model_params_b <= 1.75f) {
        return 0.02f;
    } else if (model_params_b <= 8.5f) {
        return 0.05f;
    } else {
        return 0.1f;
    }
}
