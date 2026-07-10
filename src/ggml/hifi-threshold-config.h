#ifndef HIFI_THRESHOLD_CONFIG_H
#define HIFI_THRESHOLD_CONFIG_H

///-------------------------------------------------------------------
/// HIFI Enhancement Threshold Configuration
///
/// Hardcoded model-size thresholds (in billions of parameters) and
/// enhancement fractions used by the HIFI quantisation paths.
///-------------------------------------------------------------------

// ---- Model-size breakpoints (billions of parameters) ----------------

static constexpr float HIFI_MODEL_TINY_B   = 1.0f;
static constexpr float HIFI_MODEL_SMALL_B  = 1.7f;
static constexpr float HIFI_MODEL_MEDSMALL_B = 2.0f;
static constexpr float HIFI_MODEL_MEDIUM_B = 5.0f;
static constexpr float HIFI_MODEL_LARGE_B  = 10.0f;
static constexpr float HIFI_MODEL_XLARGE_B = 15.0f;
static constexpr float HIFI_MODEL_XXLARGE_B = 20.0f;

// ---- Q4_K_HIFI enhancement thresholds
// Fraction of early attn_v layers to upgrade per model size bucket -------

static constexpr float HIFI_Q4_ENHANCE_TINY_F   = 0.32f;  // <= 1.0B
static constexpr float HIFI_Q4_ENHANCE_SMALL_F  = 0.25f;  // <= 2.0B
static constexpr float HIFI_Q4_ENHANCE_MEDIUM_F = 0.20f;  // <= 15.0B
static constexpr float HIFI_Q4_ENHANCE_NONE_F   = 0.0f;   // > 15.0B

// ---- Q3_K_HIFI enhancement thresholds ---------------------------------

static constexpr float HIFI_Q3_ENHANCE_TINY_F    = 0.0f;   // <= 1.0B  (skip tiny)
static constexpr float HIFI_Q3_ENHANCE_SMALL_F   = 0.0f;   // <= 1.7B  (disabled)
static constexpr float HIFI_Q3_ENHANCE_MEDIUM_F  = 0.25f;  // <= 5.0B
static constexpr float HIFI_Q3_ENHANCE_LARGE_F   = 0.15f;  // <= 10.0B
static constexpr float HIFI_Q3_ENHANCE_XLARGE_F  = 0.08f;  // <= 20.0B
static constexpr float HIFI_Q3_ENHANCE_XXLARGE_F = 0.05f;  // > 20.0B

#endif // HIFI_THRESHOLD_CONFIG_H
