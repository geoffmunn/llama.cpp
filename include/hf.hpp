// SPDX-License-Identifier: MIT

/**
 * @file hf.hpp
 * @brief HiFi (layer-adaptive) quantization helpers for C++.
 */

#ifndef HF_HPP
#define HF_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Compute block-level importance from a slice of an importance matrix.
/// Returns the mean absolute value across the block elements.
float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size);

#ifdef __cplusplus
}
#endif

#endif // HF_HPP
