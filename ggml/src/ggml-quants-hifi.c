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

static thread_local int g_tls_tensor_outliers = 0;

void ggml_q3_hifi_set_tensor_outliers(int outliers) {
    g_tls_tensor_outliers = outliers;
}

int ggml_q3_hifi_get_tensor_outliers(void) {
    return g_tls_tensor_outliers;
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
