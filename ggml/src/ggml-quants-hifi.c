// SPDX-License-Identifier: MIT

/**
 * @file ggml-quants-hifi.c
 * @brief Layer-adaptive (HiFi) quantization context and utility API.
 */

#include "ggml-quants-hifi.h"

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
