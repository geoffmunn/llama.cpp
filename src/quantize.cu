// CUDA-side helper for HIFI enhancement thresholds
// Mirrors the CPU-side logic so the CUDA quantisation path can decide
// which attn_v layers get upgraded per model-size bucket.

#include "ggml/hifi-threshold-config.h"

/// Return the fraction of early attention-V layers that should be
/// enhanced for a given model size (in billions of parameters).
/// Used by the LLAMA_FTYPE_MOSTLY_Q4_K_HIFI selection logic.
float get_hifi_enhancement_threshold(float model_params_b) {
    if (model_params_b <= HIFI_MODEL_TINY_B)      return HIFI_Q4_ENHANCE_TINY_F;
    if (model_params_b <= HIFI_MODEL_MEDSMALL_B)  return HIFI_Q4_ENHANCE_SMALL_F;
    if (model_params_b <= HIFI_MODEL_XLARGE_B)    return HIFI_Q4_ENHANCE_MEDIUM_F;
    return HIFI_Q4_ENHANCE_NONE_F;
}
