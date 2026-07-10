#ifndef QUANTIZE_H
#define QUANTIZE_H

// Number of high-precision floating-point corrections stored per block header
#define Q4_K_HIFI_OUTLIERS 8

#include "ggml.h"

// For Q4_K_HIFI enhancement of critical tensors
static ggml_type get_hifi_enhanced_type(float model_params_b) {
    return (model_params_b <= 5.0f) ? GGML_TYPE_Q5_K_HIFI_RES8
                                    : GGML_TYPE_Q6_K_HIFI_RES8;
}

#endif // QUANTIZE_H
