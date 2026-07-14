// SPDX-License-Identifier: MIT

/**
 * @file qimpl.cc
 * @brief Quantization implementation helpers for layer-adaptive (HiFi) modes.
 */

#include "ggml-quants-hifi.h"

namespace {

/// Outlier threshold used to decide whether a block warrants extra outliers.
/// Higher importance blocks cross this threshold more easily.
static float get_block_outlier_threshold(float model_params_b) {
    // Smaller models need fewer outliers; larger models benefit from more.
    if (model_params_b <= 1.7f) {
        return 0.75f;   // tiny: conservative
    } else if (model_params_b <= 8.0f) {
        return 0.50f;  // medium: balanced
    } else {
        return 0.35f;  // large: aggressive
    }
}

} // anonymous namespace

int ggml_hifi_compute_block_outlier_count(float block_importance,
                                          int base_outlier_count,
                                          float model_params_b) {
    // Clamp base count to valid range [1, 8]
    if (base_outlier_count < 1) base_outlier_count = 1;
    if (base_outlier_count > 8) base_outlier_count = 8;

    float threshold = get_block_outlier_threshold(model_params_b);

    // If the block importance exceeds the threshold, grant one extra outlier
    // (up to the hard cap of 8). Otherwise keep the base count.
    if (block_importance > threshold) {
        int enhanced = base_outlier_count + 1;
        if (enhanced > 8) enhanced = 8;
        return enhanced;
    }

    return base_outlier_count;
}
