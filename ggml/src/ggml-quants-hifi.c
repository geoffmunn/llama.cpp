#include "ggml-quants-hifi.h"
#include <string.h>

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
