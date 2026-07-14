#pragma once

#include <stddef.h>

#ifndef GGML_COMMON_H
#define GGML_COMMON_H
#endif

// --- Q2_K_HiFi reference quantization ---

#ifdef __cplusplus
extern "C" {
#endif

void quantize_row_q2_k_hifi_ref(const float * x, void * y, int64_t k);
void dequantize_row_q2_k_hifi(const void * x, float * y, int64_t k);

#ifdef __cplusplus
}
#endif
