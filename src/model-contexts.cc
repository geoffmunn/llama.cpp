//
// model-contexts.cc — FFN gate special-case handling for small models under
//                     Q4/Q5/K/HI quantisation modes
//
// Detects whether the current tensor belongs to an FFN gating mechanism and
// applies model-size-aware heuristics before falling back to the standard
// category-based rules used by llama_tensor_get_type_impl().
//
// Small-model threshold: >1.7 B parameters (HIFI_MODEL_SMALL_B).
//

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-ext.h"

extern "C" {
#include "../ggml/src/ggml-quants-hifi.h"
#include "ggml/quantize.h"
}

#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns true when the tensor name indicates an FFN gating weight
/// (e.g. "blk.<n>.ffn_gate.weight", "ffn_gate_inp.weight", etc.).
static inline bool is_ffn_gate_tensor(const char * tensor_name) {
    std::string name(tensor_name);
    return name.find("ffn_gate") != std::string::npos;
}

/// Compute the effective model size in billions of parameters.
/// Matches the formula used elsewhere in llama-quant.cpp:
///   n_layers * n_embd * 8 / 1e9
/// (the factor 8 is a rough normalisation for typical llama-style arches).
static inline float compute_model_params_b(const struct ggml_context * ctx) {
    // We need hparams from the model; since this module is called
    // during quantisation we pass the pre-computed value through.
    // This helper is provided for callers that already have ctx.
    return 0.0f; // placeholder — real value comes from caller
}

// ---------------------------------------------------------------------------
// FFN gate type selection for HIFI ftypes on small models
// ---------------------------------------------------------------------------
//
// For Q4_K_HIFI and Q5_K_HIFI the FFN gate tensors on tiny models
// (<=1.7 B) are sensitive to aggressive quantisation.  The strategy
// below upgrades them early so that the gating pathway retains enough
// precision to route correctly.
//
// For Q3_K_HIFI and Q2_K_HIFI the same logic applies but the base
// types are lower, so we cap the upgrade at Q4_K / Q3_K respectively.

/// Select the target ggml_type for an FFN gate tensor under a HIFI ftype
/// when the model is small (model_params_b <= 1.7).
///
/// Parameters
/// ----------
/// model_params_b : effective model size in billions
/// ftype          : one of the LLAMA_FTYPE_MOSTLY_*_HIFI values
/// i_layer        : zero-based layer index
/// n_layer        : total number of layers
///
/// Returns
/// -------
/// The selected ggml_type, or GGML_TYPE_COUNT to signal "use default".
static ggml_type select_ffn_gate_hifi_type(float model_params_b,
                                           llama_ftype ftype,
                                           int i_layer,
                                           int n_layer) {
    // Only apply special handling for small models.
    if (model_params_b > HIFI_MODEL_SMALL_B) {
        return GGML_TYPE_COUNT; // defer to standard heuristics
    }

    // Upgrade fraction: how many early layers get the bump.
    // Tiny models benefit from a more generous fraction because every
    // layer matters proportionally more.
    const float upgrade_fraction = (model_params_b <= HIFI_MODEL_TINY_B)
                                      ? 0.40f
                                      : 0.30f;
    const int upgrade_limit = static_cast<int>(n_layer * upgrade_fraction);

    bool should_upgrade = (i_layer < upgrade_limit);

    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_K_HIFI: {
            // Small models: upgrade early FFN gate layers to the enhanced type
            if (should_upgrade) {
                return get_hifi_enhanced_type(model_params_b);
            }
            return GGML_TYPE_Q4_K_HIFI;
        }

        case LLAMA_FTYPE_MOSTLY_Q5_K_HIFI: {
            if (should_upgrade) {
                return get_q5_hifi_enhanced_type(model_params_b);
            }
            return GGML_TYPE_Q5_K;
        }

        case LLAMA_FTYPE_MOSTLY_Q3_K_HIFI: {
            // Q3_K_HIFI on tiny models — cap at Q4_K for early layers,
            // otherwise use Q3_K (matching the guide excerpt behaviour).
            if (should_upgrade) {
                return GGML_TYPE_Q4_K;
            }
            return GGML_TYPE_Q3_K;
        }

        case LLAMA_FTYPE_MOSTLY_Q2_K_HIFI: {
            // Q2_K_HIFI is already very aggressive; for tiny models
            // upgrade early FFN gates to Q3_K_HIFI to preserve routing.
            if (should_upgrade) {
                return GGML_TYPE_Q3_K;
            }
            return GGML_TYPE_Q2_K_HIFI;
        }

        default:
            break;
    }

    return GGML_TYPE_COUNT; // not a HIFI ftype — caller handles
}

// ---------------------------------------------------------------------------
// Public entry point — called from llama_tensor_get_type_impl before the
// standard FFN_GATE block (lines ~775 of llama-quant.cpp).
// ---------------------------------------------------------------------------

/// Check whether `tensor` is an FFN gate tensor under a HIFI ftype and
/// apply small-model heuristics.  Returns the selected type on success
/// or GGML_TYPE_COUNT to fall through to the default path.
ggml_type llama_model_check_ffn_gate_hifi(
    float model_params_b,
    llama_ftype ftype,
    const char * tensor_name,
    int i_layer,
    int n_layer)
{
    // Fast path: skip non-HIFI ftypes entirely.
    if (ftype != LLAMA_FTYPE_MOSTLY_Q4_K_HIFI &&
        ftype != LLAMA_FTYPE_MOSTLY_Q5_K_HIFI &&
        ftype != LLAMA_FTYPE_MOSTLY_Q3_K_HIFI &&
        ftype != LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) {
        return GGML_TYPE_COUNT;
    }

    // Only relevant for FFN gate tensors.
    if (!is_ffn_gate_tensor(tensor_name)) {
        return GGML_TYPE_COUNT;
    }

    return select_ffn_gate_hifi_type(model_params_b, ftype, i_layer, n_layer);
}
