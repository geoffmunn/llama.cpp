#include "quantize.h"

/*
 * Threshold breakpoints mirror those used by get_hifi_enhanced_type():
 *   <= 1.75 B  → small  models  (more generous promotion)
 *   <= 8.5 B   → medium models
 *   >  8.5 B   → large  models  (conservative promotion)
 *
 * The returned value is multiplied against qs.n_attention_wv in the
 * MOSTLY_Q4_K_HIFI branch.  If the running count of processed attention-V
 * tensors stays below that product, the enhanced type is chosen instead
 * of the base Q4_K_HIFI.
 */
float get_hifi_enhancement_threshold(float model_params_b) {
    if (model_params_b <= 1.75f) {
        return 0.5f;
    } else if (model_params_b <= 8.5f) {
        return 0.25f;
    } else {
        return 0.125f;
    }
}
