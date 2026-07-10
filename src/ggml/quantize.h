#ifndef QUANTIZE_H
#define QUANTIZE_H

// Number of high-precision floating-point corrections stored per block header
#define Q4_K_HIFI_OUTLIERS 8

#include "ggml.h"
#include "hifi-threshold-config.h"

// For Q4_K_HIFI enhancement of critical tensors
static ggml_type get_hifi_enhanced_type(float model_params_b) {
    return (model_params_b <= HIFI_MODEL_MEDIUM_B) ? GGML_TYPE_Q5_K_HIFI_RES8
                                                   : GGML_TYPE_Q6_K_HIFI_RES8;
}

// For Q5_K_HIFI enhancement of critical tensors
static ggml_type get_q5_hifi_enhanced_type(float model_params_b) {
    if (model_params_b <= HIFI_MODEL_MEDSMALL_B) return GGML_TYPE_Q6_K;          // no HIFI overhead
    if (model_params_b <= HIFI_MODEL_MEDIUM_B)  return GGML_TYPE_Q5_K_HIFI_RES8;
    return GGML_TYPE_Q6_K_HIFI_RES8;
}

// Q4_K_HIFI: fraction of early attn_v layers to enhance
static float get_hifi_enhancement_threshold(float model_params_b) {
    if (model_params_b <= HIFI_MODEL_TINY_B)   return HIFI_Q4_ENHANCE_TINY_F;
    if (model_params_b <= HIFI_MODEL_MEDSMALL_B) return HIFI_Q4_ENHANCE_SMALL_F;
    if (model_params_b <= HIFI_MODEL_XLARGE_B)  return HIFI_Q4_ENHANCE_MEDIUM_F;
    return HIFI_Q4_ENHANCE_NONE_F;
}

#endif // QUANTIZE_H
