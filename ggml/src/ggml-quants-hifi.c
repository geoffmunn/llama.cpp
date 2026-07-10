#include "ggml-quants-hifi.h"
#include <string.h>
#include <math.h>

// ------------------------------------------------------------------
// Global quantization context
// ------------------------------------------------------------------

static ggml_hifi_quant_context g_hifi_context = {
    .outlier_count    = 4,
    .layer_importance = 0.5f,
    .layer_idx        = -1,
    .total_layers     = -1,
    .is_active        = 0,
    .model_params_b   = 0.0f,
};

const ggml_hifi_quant_context * ggml_hifi_get_context(void) {
    return &g_hifi_context;
}

void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx) {
    if (ctx != NULL) {
        memcpy(&g_hifi_context, ctx, sizeof(ggml_hifi_quant_context));
    } else {
        memset(&g_hifi_context, 0, sizeof(ggml_hifi_quant_context));
        g_hifi_context.outlier_count    = 4;
        g_hifi_context.layer_importance = 0.5f;
        g_hifi_context.layer_idx        = -1;
        g_hifi_context.total_layers     = -1;
    }
}

// ------------------------------------------------------------------
// Outlier count computation
// ------------------------------------------------------------------

int ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                     float layer_importance, float model_params_b) {
    // Base outlier count: larger models need more outliers to preserve precision
    int base = 2;
    if (model_params_b > 8.0f) {
        base = 4;
    }
    if (model_params_b > 14.0f) {
        base = 6;
    }

    // Middle layers typically have higher variance distributions
    float depth_ratio = 0.0f;
    if (total_layers > 0) {
        depth_ratio = (float) layer_idx / (float) total_layers;
    }
    float middle_factor = 1.0f - 2.0f * fabsf(depth_ratio - 0.5f);

    // Scale by importance and position
    float adjusted = (float) base + middle_factor * 2.0f + layer_importance * 2.0f;

    // Clamp to [1, 8]
    int result = (int) (adjusted + 0.5f);
    if (result < 1) result = 1;
    if (result > 8) result = 8;

    return result;
}
