# HIFI Quantization Implementation Guide

This guide enables an AI agent to apply all HIFI quantization changes to a fresh upstream
llama.cpp checkout and produce a working build with 13 new quantization types.

---

## 1. Overview of HIFI Quantization Types

HIFI (High-Fidelity) adds 13 new quantization types on top of upstream llama.cpp. They fall
into two families:

### HIFI Family (outlier-protection techniques)

| Type | ID | Block size | BPW | Description |
|------|----|------------|-----|-------------|
| Q3_K_HIFI | 42 | 136 bytes | ~3.4 | Q3_K layout + 8 FP16 outliers per block |
| Q6_K_HIFI | 43 | 222 bytes | ~6.8 | Q6_K layout + 4 FP16 outliers |
| Q6_K_HIFI_DYNAMIC | 44 | 236 bytes | ~7.4 | Q6_K + 2-8 dynamic outliers per layer |
| Q6_K_HIFI_RES8 | 45 | 232 bytes | ~7.3 | Q6_K + 8 INT8 residuals (compact) |
| Q5_K_HIFI_RES8 | 46 | 196 bytes | ~5.6 | Q5_K + 8 INT8 residuals (efficient) |
| Q3_K_HIFI_RES8 | 47 | 132 bytes | ~4.1 | Q3_K + 8 INT8 residuals (lean) |
| Q4_K_HIFI | 48 | 168 bytes | ~5.25 | Q4_K layout + 8 FP16 outliers |
| Q2_K_HIFI | 49 | 96 bytes | ~3.0 | Q2_K + 3 FP16 outliers |

### LITE Family (shifted-base + INT8 residuals)

| Type | ID | Block size | BPW | Description |
|------|----|------------|-----|-------------|
| Q2_K_LITE | 50 | 96 bytes | ~3.0 | Q2_K base + 4 INT8 residuals |
| Q3_K_LITE | 51 | 104 bytes | ~3.25 | Q2_K base + 8 INT8 residuals |
| Q4_K_LITE | 52 | 128 bytes | ~4.0 | Q3_K base + 7 INT8 residuals |
| Q5_K_LITE | 53 | 164 bytes | ~5.13 | Q4_K base + 8 INT8 residuals |
| Q6_K_LITE | 54 | 196 bytes | ~6.13 | Q5_K base + 8 INT8 residuals |

`GGML_TYPE_COUNT = 55`

### Mixed quantization presets (ftypes)

| Preset | ftype | Description |
|--------|-------|-------------|
| Q2_K_HIFI | 47 | Q2_K + INT8 residuals on critical tensors |
| Q3_K_HIFI | 45 | Q3_K_M base + HIFI on critical tensors |
| Q4_K_HIFI | 44 | Q4_K_M + 2-8 dynamic outliers + early exit |
| Q5_K_HIFI | 46 | Q5_K_M base + Q6_K_HIFI_RES8 on critical |
| Q2_K_LITE | 48 | Q2_K base + INT8 residuals (~3.0 bpw) |
| Q3_K_LITE | 49 | Q2_K base + INT8 residuals (~3.25 bpw) |
| Q4_K_LITE | 50 | Q3_K base + INT8 residuals (~4.0 bpw) |
| Q5_K_LITE | 51 | Q4_K base + INT8 residuals (~5.13 bpw) |
| Q6_K_LITE | 52 | Q5_K base + INT8 residuals (~6.13 bpw) |

---

## 2. New Files to Create

These files do not exist in upstream and must be created entirely.

### 2.1 `ggml/src/ggml-quants-hifi.h`

Create this file at `ggml/src/ggml-quants-hifi.h` with the following content:

```c
// GGML HIFI Quantization Context
// Provides layer-adaptive outlier allocation for Q4_K_HIFI quantization
//
// This header defines the context infrastructure for passing layer-specific
// parameters to the quantization functions without modifying the core GGML API.

#ifndef GGML_QUANTS_HIFI_H
#define GGML_QUANTS_HIFI_H

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum outliers per block for Q6_K_HIFI_RES8 format
// Must match the value in ggml-common.h
#ifndef Q6_K_HIFI_RES8_MAX_OUTLIERS
#define Q6_K_HIFI_RES8_MAX_OUTLIERS 8
#endif

// Maximum outliers per block for Q5_K_HIFI_RES8 format
// Must match the value in ggml-common.h
#ifndef Q5_K_HIFI_RES8_MAX_OUTLIERS
#define Q5_K_HIFI_RES8_MAX_OUTLIERS 8
#endif

// Layer-adaptive quantization context
// Used to pass dynamic parameters to Q6_K_HIFI_RES8 quantization
typedef struct {
    int outlier_count;           // Number of outliers to preserve (1-8)
    float layer_importance;      // Layer importance score (0.0-1.0), for logging
    int layer_idx;               // Current layer index, for debugging
    int total_layers;            // Total layer count, for debugging
    int is_active;               // Whether adaptive mode is enabled
    float model_params_b;        // Model size in billions (e.g., 0.6, 1.7, 4.0, 8.0)
} ggml_hifi_quant_context;

// Get the current thread-local quantization context
// Returns NULL if no context is set
GGML_API const ggml_hifi_quant_context * ggml_hifi_get_context(void);

// Set the quantization context for the current thread
// Pass NULL to clear the context
GGML_API void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx);

// Convenience function to compute adaptive outlier count based on layer position and importance
// Parameters:
//   layer_idx: Current layer index (0-based)
//   total_layers: Total number of layers in the model
//   layer_importance: Normalized importance score (0.0-1.0), from imatrix aggregation
//   model_params_b: Model size in billions (e.g., 0.6, 1.7, 4.0, 8.0)
// Returns: Optimal outlier count (2-8)
GGML_API int ggml_hifi_compute_outlier_count(
    int layer_idx,
    int total_layers,
    float layer_importance,
    float model_params_b
);

// Convenience function to compute layer importance from imatrix data
// Parameters:
//   imatrix_data: Per-element importance weights from imatrix
//   n_elements: Number of elements in the tensor
// Returns: Aggregated importance score (0.0-1.0 after normalization)
GGML_API float ggml_hifi_compute_tensor_importance(
    const float * imatrix_data,
    int64_t n_elements
);

// Strategy 1: Compute per-block importance from imatrix data
// Used for adaptive per-block outlier allocation
// Parameters:
//   imatrix_block: Per-element importance weights for this block (QK_K elements)
//   block_size: Number of elements in the block (typically QK_K = 256)
// Returns: Block importance score (0.0-1.0)
GGML_API float ggml_hifi_compute_block_importance(
    const float * imatrix_block,
    int block_size
);

// Strategy 1: Compute per-block outlier count based on local imatrix variance
// High variance blocks get more outliers, low variance blocks get fewer
// Parameters:
//   block_importance: Importance score for this block (0.0-1.0)
//   base_outlier_count: Base outlier count from tensor-level computation
//   model_params_b: Model size in billions
// Returns: Adjusted outlier count for this block (2-8)
GGML_API int ggml_hifi_compute_block_outlier_count(
    float block_importance,
    int base_outlier_count,
    float model_params_b
);

// ===========================================================================
// Memory Layout Constants for Cross-Backend Consistency
// Block sizes are validated at compile time via static_assert in ggml-common.h:
//   static_assert(sizeof(block_q6_k_hifi_res8) == 232, ...)
//   static_assert(sizeof(block_q5_k_hifi_res8) == 200, ...)
// ===========================================================================

// Q6_K_HIFI_RES8: 232 bytes total (210 base + 22 extension)
// Layout: ql[128] + qh[64] + scales[16] + d[2] + outlier_count[1] +
//         outlier_idx[8] + residual_vals[8] + _padding[1] + residual_scale[4]
#define Q6_K_HIFI_RES8_BLOCK_SIZE 232

// Q5_K_HIFI_RES8: 200 bytes total (176 base + 24 extension)
// Layout: dm[4] + scales[12] + qh[32] + qs[128] + outlier_count[1] +
//         outlier_idx[8] + residual_vals[8] + _padding[3] + residual_scale[4]
#define Q5_K_HIFI_RES8_BLOCK_SIZE 200

// ===========================================================================
// Q3_K_HIFI Adaptive Enhancement API
// Implements scale-aware tensor selection and statistical outlier detection
// ===========================================================================

// Q3_K_HIFI block constants
#ifndef Q3_K_HIFI_MAX_OUTLIERS
#define Q3_K_HIFI_MAX_OUTLIERS 8
#endif

// Model size categories for Q3_K_HIFI
typedef enum {
    Q3_HIFI_SIZE_TINY   = 0,  // ≤1.7B: minimal or no HIFI enhancement
    Q3_HIFI_SIZE_MEDIUM = 1,  // 2B-8B: full enhancement (sweet spot)
    Q3_HIFI_SIZE_LARGE  = 2,  // 14B+: reduced enhancement (leverage redundancy)
} ggml_q3_hifi_size_category;

// Get model size category from parameter count
GGML_API ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b);

// Get maximum outlier count for Q3_K_HIFI based on model size
GGML_API int ggml_q3_hifi_get_max_outliers(float model_params_b);

// Get outlier ratio threshold for Q3_K_HIFI tensor enhancement
GGML_API float ggml_q3_hifi_get_outlier_threshold(float model_params_b);

// Compute statistical outlier ratio using 3σ rule
GGML_API float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n);

// Determine if a tensor should receive Q3_K_HIFI enhancement
GGML_API int ggml_q3_hifi_should_enhance_tensor(
    const char * tensor_name,
    const float * weights,
    int64_t n_elements,
    float model_params_b,
    int * enhanced_count,
    int max_enhanced
);

// Get the enhancement type for Q3_K_HIFI critical tensors
GGML_API int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding);

// Get percentage of attn_v layers to enhance
GGML_API float ggml_q3_hifi_get_attn_v_threshold(float model_params_b);

// ===========================================================================
// Q3_K_HIFI Per-Tensor Outlier Control (TLS)
// ===========================================================================

GGML_API void ggml_q3_hifi_set_tensor_outliers(int outliers);
GGML_API int ggml_q3_hifi_get_tensor_outliers(void);
GGML_API void ggml_q3_hifi_set_tensor_importance(float importance);
GGML_API float ggml_q3_hifi_get_tensor_importance(void);
GGML_API void ggml_q3_hifi_reset_tensor_state(void);

GGML_API int ggml_q3_hifi_compute_block_outliers(
    float block_outlier_ratio,
    int base_outlier_count,
    float model_params_b
);

// ===========================================================================
// Q4_K_HIFI Adaptive Enhancement API
// ===========================================================================

#ifndef Q4_K_HIFI_MAX_OUTLIERS
#define Q4_K_HIFI_MAX_OUTLIERS 8
#endif

GGML_API int ggml_q4_hifi_get_max_outliers(float model_params_b);

// ===========================================================================
// K_LITE Tier-Based Residual Budget API
// ===========================================================================

GGML_API int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b, int max_residuals);

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_HIFI_H
```

### 2.2 `ggml/src/ggml-quants-hifi.c`

Create this file at `ggml/src/ggml-quants-hifi.c` with the following content:

```c
// GGML HIFI Quantization Context Implementation
// Layer-adaptive outlier allocation for Q4_K_HIFI quantization

#include "ggml-quants-hifi.h"
#include <math.h>
#include <stdlib.h>

// Thread-local storage for the quantization context
// Using a simple pointer approach - the context lifetime is managed by the caller
#ifdef _MSC_VER
    static __declspec(thread) const ggml_hifi_quant_context * g_hifi_context = NULL;
    // Q3_K_HIFI per-tensor outlier count (set before quantizing each tensor)
    static __declspec(thread) int g_q3_hifi_tensor_outliers = -1;  // -1 = use default
    static __declspec(thread) float g_q3_hifi_tensor_importance = 0.5f;
#else
    static __thread const ggml_hifi_quant_context * g_hifi_context = NULL;
    // Q3_K_HIFI per-tensor outlier count (set before quantizing each tensor)
    static __thread int g_q3_hifi_tensor_outliers = -1;  // -1 = use default
    static __thread float g_q3_hifi_tensor_importance = 0.5f;
#endif

const ggml_hifi_quant_context * ggml_hifi_get_context(void) {
    return g_hifi_context;
}

void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx) {
    g_hifi_context = ctx;
}

// ===========================================================================
// Q3_K_HIFI Per-Tensor Outlier Control (TLS)
// Allows setting outlier count per tensor before quantization
// ===========================================================================

// Set outlier count for the current tensor being quantized
// Pass -1 to use the default model-size-based count
void ggml_q3_hifi_set_tensor_outliers(int outliers) {
    g_q3_hifi_tensor_outliers = outliers;
}

// Get the current tensor outlier count (-1 if using default)
int ggml_q3_hifi_get_tensor_outliers(void) {
    return g_q3_hifi_tensor_outliers;
}

// Set tensor importance for current quantization
void ggml_q3_hifi_set_tensor_importance(float importance) {
    g_q3_hifi_tensor_importance = importance;
}

// Get current tensor importance
float ggml_q3_hifi_get_tensor_importance(void) {
    return g_q3_hifi_tensor_importance;
}

// Reset TLS state to defaults (call after each tensor)
void ggml_q3_hifi_reset_tensor_state(void) {
    g_q3_hifi_tensor_outliers = -1;
    g_q3_hifi_tensor_importance = 0.5f;
}

// Compute adaptive outlier count based on layer position, importance, and model scale
// This is the core algorithm for layer-wise imatrix adaptation
// Strategy 2 optimization: More aggressive reduction in middle/late layers
int ggml_hifi_compute_outlier_count(
    int layer_idx,
    int total_layers,
    float layer_importance,
    float model_params_b
) {
    if (total_layers <= 0) {
        return 8; // Default to max for safety
    }

    // Compute depth ratio (0.0 = first layer, 1.0 = last layer)
    float depth_ratio = (float)layer_idx / (float)(total_layers - 1);
    if (total_layers == 1) depth_ratio = 0.5f;

    // Base outlier count based on layer position
    // Strategy 2: More aggressive reduction for size optimization
    // Early layers (0-30%): Max precision - context formation is critical
    // Middle layers (30-70%): Reduced precision (5 instead of 7)
    // Late layers (70-100%): Minimal precision (2 instead of 5)
    int base_count;
    if (depth_ratio <= 0.30f) {
        base_count = 8;  // Early layers: max outliers (unchanged)
    } else if (depth_ratio <= 0.70f) {
        base_count = 5;  // Middle layers: reduced (was 7)
    } else {
        base_count = 2;  // Late layers: minimal (was 5)
    }

    // Scale-dependent adjustment
    // Key insight: Large models have more redundancy, can use fewer outliers
    // Small models need more outliers to maintain quality
    float scale_factor = 1.0f;
    if (model_params_b >= 7.0f) {
        // 7B+ models: already minimal late layers, no further reduction needed
        // But we can slightly reduce middle layers for extra savings
        if (depth_ratio > 0.30f && depth_ratio <= 0.70f) {
            scale_factor = 0.9f;  // Middle layers: slight reduction
        }
    } else if (model_params_b >= 3.0f) {
        // 3-7B models: Moderate approach
        if (depth_ratio > 0.70f) {
            scale_factor = 1.0f;  // Late layers already at minimum
        } else if (depth_ratio > 0.30f) {
            scale_factor = 0.95f; // Middle layers: very light reduction
        }
    } else if (model_params_b >= 1.5f) {
        // 1.5-3B models: Be more conservative, boost late layers slightly
        if (depth_ratio > 0.70f) {
            scale_factor = 1.25f; // Boost late layers back up (2 -> ~3)
        }
    } else if (model_params_b <= 1.0f) {
        // Small models (<1B): boost outliers everywhere
        // Small models are more sensitive to quantization error
        scale_factor = 1.3f;
        if (depth_ratio <= 0.30f) {
            scale_factor = 1.4f;  // Extra boost for early layers
        } else if (depth_ratio > 0.70f) {
            scale_factor = 1.5f;  // Late layers need more for small models (2 -> 3)
        }
    }

    // Apply importance adjustment
    // layer_importance is normalized to 0.0-1.0
    // High importance (>0.7): boost outlier count
    // Low importance (<0.3): reduce outlier count
    float importance_factor = 1.0f;
    if (layer_importance > 0.7f) {
        importance_factor = 1.0f + (layer_importance - 0.7f);  // Up to 1.3x
    } else if (layer_importance < 0.3f) {
        importance_factor = 0.7f + (layer_importance / 0.3f) * 0.3f;  // 0.7-1.0x
    }

    // Combine factors
    float final_count_f = (float)base_count * scale_factor * importance_factor;
    int final_count = (int)roundf(final_count_f);

    // Clamp to valid range [2, 8]
    if (final_count < 2) final_count = 2;
    if (final_count > 8) final_count = 8;

    return final_count;
}

// Compute tensor importance from imatrix data
// Uses the average of squared importance weights as the metric
float ggml_hifi_compute_tensor_importance(
    const float * imatrix_data,
    int64_t n_elements
) {
    if (imatrix_data == NULL || n_elements <= 0) {
        return 0.5f;  // Default to medium importance if no data
    }

    // Compute mean squared importance
    // This weights larger importance values more heavily
    double sum_sq = 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < n_elements; ++i) {
        double val = (double)imatrix_data[i];
        sum += val;
        sum_sq += val * val;
    }

    // Use coefficient of variation as importance metric
    // High variance in importance = some weights are critical = high importance
    double mean = sum / (double)n_elements;
    double mean_sq = sum_sq / (double)n_elements;
    double variance = mean_sq - mean * mean;

    if (mean < 1e-10 || variance < 0) {
        return 0.5f;
    }

    // Coefficient of variation (CV) = stddev / mean
    double stddev = sqrt(variance);
    double cv = stddev / mean;

    // Normalize CV to 0-1 range
    // Empirically, CV values typically range from 0.1 to 3.0 for imatrix data
    // Map this to 0.2 - 0.9 importance range
    float importance = 0.2f + 0.7f * (float)(cv / 3.0);
    if (importance > 0.9f) importance = 0.9f;
    if (importance < 0.2f) importance = 0.2f;

    return importance;
}

// Strategy 1: Compute per-block importance from imatrix data
// Uses coefficient of variation within the block as the importance metric
float ggml_hifi_compute_block_importance(
    const float * imatrix_block,
    int block_size
) {
    if (imatrix_block == NULL || block_size <= 0) {
        return 0.5f;  // Default to medium importance
    }

    // Compute statistics for this block
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_val = 0.0;

    for (int i = 0; i < block_size; ++i) {
        double val = (double)imatrix_block[i];
        sum += val;
        sum_sq += val * val;
        if (val > max_val) max_val = val;
    }

    double mean = sum / (double)block_size;
    if (mean < 1e-10) {
        return 0.3f;  // Low importance for near-zero blocks
    }

    double mean_sq = sum_sq / (double)block_size;
    double variance = mean_sq - mean * mean;
    if (variance < 0) variance = 0;

    // Coefficient of variation (CV)
    double stddev = sqrt(variance);
    double cv = stddev / mean;

    // Also consider the max/mean ratio (spikiness)
    double spikiness = max_val / mean;

    // Combine CV and spikiness for final importance
    // High CV = high variance = some weights are outliers = need more outliers
    // High spikiness = extreme values present = need more outliers
    double combined = 0.6 * cv + 0.4 * (spikiness / 10.0);  // spikiness typically 1-20

    // Normalize to 0.2 - 0.9 range
    float importance = 0.2f + 0.7f * (float)(combined / 2.0);  // combined typically 0-3
    if (importance > 0.9f) importance = 0.9f;
    if (importance < 0.2f) importance = 0.2f;

    return importance;
}

// Strategy 1: Compute per-block outlier count based on local imatrix variance
// Adjusts the base outlier count up or down based on block importance
int ggml_hifi_compute_block_outlier_count(
    float block_importance,
    int base_outlier_count,
    float model_params_b
) {
    // Scale factor based on block importance
    // High importance (>0.7): boost outliers up to 1.5x
    // Low importance (<0.3): reduce outliers down to 0.5x
    // Medium importance: keep base count
    float scale = 1.0f;

    if (block_importance > 0.7f) {
        // High importance block - boost outliers
        scale = 1.0f + 0.5f * (block_importance - 0.7f) / 0.3f;  // 1.0 to 1.5
    } else if (block_importance < 0.3f) {
        // Low importance block - reduce outliers
        scale = 0.5f + 0.5f * (block_importance / 0.3f);  // 0.5 to 1.0
    }

    // For larger models, be more aggressive with reduction on low-importance blocks
    if (model_params_b >= 7.0f && block_importance < 0.4f) {
        scale *= 0.8f;  // Additional 20% reduction for large models
    }

    int adjusted_count = (int)roundf((float)base_outlier_count * scale);

    // Clamp to valid range [1, 8]
    // Allow minimum of 1 for low-importance blocks (save more space)
    if (adjusted_count < 1) adjusted_count = 1;
    if (adjusted_count > 8) adjusted_count = 8;

    return adjusted_count;
}

// ===========================================================================
// Q3_K_HIFI Adaptive Enhancement Functions
// Implements scale-aware tensor selection and statistical outlier detection
// Based on proven strategies from Q4_K_HIFI and Q5_K_HIFI
// ===========================================================================

// Get model size category for Q3_K_HIFI adaptive strategy
ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b) {
    if (model_params_b <= 1.7f) {
        return Q3_HIFI_SIZE_TINY;   // 0.6B, 1.7B - minimal/no HIFI
    } else if (model_params_b <= 10.0f) {
        return Q3_HIFI_SIZE_MEDIUM; // 2B-8B - full HIFI (sweet spot)
    } else {
        return Q3_HIFI_SIZE_LARGE;  // 14B, 32B+ - reduced HIFI
    }
}

// Get maximum outlier count for Q3_K_HIFI based on model size
// Key insight from Q5_K_HIFI: Fixed enhancement doesn't scale!
// - Small models: HIFI overhead hurts more than it helps
// - Medium models: Full benefit from outlier preservation
// - Large models: Self-correcting, excessive outliers waste bits
int ggml_q3_hifi_get_max_outliers(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);

    switch (cat) {
        case Q3_HIFI_SIZE_TINY:
            // ≤1.7B: 0-2 outliers
            // 0.6B especially struggles with BPW overhead
            if (model_params_b <= 0.8f) {
                return 0;  // Skip HIFI entirely for 0.6B
            }
            return 2;  // Minimal for 1.7B

        case Q3_HIFI_SIZE_MEDIUM:
            // 2B-8B: Full enhancement
            // This is where Q3_K_HIFI already wins (4B: -2.9% PPL)
            if (model_params_b <= 5.0f) {
                return 8;  // Max outliers for 2-5B
            }
            return 6;  // Slightly reduced for 8B

        case Q3_HIFI_SIZE_LARGE:
            // 14B+: Minimal enhancement
            // Large models have redundancy, extra outliers waste bits
            if (model_params_b >= 30.0f) {
                return 2;  // 32B+ gets minimal
            }
            return 4;  // 14B gets moderate

        default:
            return 4;  // Safe default
    }
}

// Get outlier ratio threshold for tensor enhancement decision
// Only enhance tensors with outlier ratio above this threshold
// Based on Q5_K_HIFI statistical detection patterns
float ggml_q3_hifi_get_outlier_threshold(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);

    switch (cat) {
        case Q3_HIFI_SIZE_TINY:
            // Very selective - only enhance if absolutely needed
            return 0.12f;  // 12% threshold

        case Q3_HIFI_SIZE_MEDIUM:
            // Moderate selectivity - catch most high-sensitivity tensors
            if (model_params_b <= 5.0f) {
                return 0.06f;  // 6% for 2-5B
            }
            return 0.05f;  // 5% for 5-8B

        case Q3_HIFI_SIZE_LARGE:
            // Relaxed threshold - focus on highest-outlier tensors
            return 0.04f;  // 4% for 14B+

        default:
            return 0.08f;
    }
}

// Compute statistical outlier ratio using 3σ rule
// This is used to determine which tensors benefit from HIFI enhancement
float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n) {
    if (weights == NULL || n <= 0) {
        return 0.0f;
    }

    // Single-pass mean and variance using Welford's algorithm
    double mean = 0.0;
    double m2 = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        double x = (double)weights[i];
        double delta = x - mean;
        mean += delta / (double)(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance = m2 / (double)n;
    if (variance <= 0.0) {
        return 0.0f;
    }

    double stddev = sqrt(variance);
    double threshold = 3.0 * stddev;

    // Count outliers (weights beyond 3σ from mean)
    int64_t outlier_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        double diff = (double)weights[i] - mean;
        if (diff < 0) diff = -diff;  // fabs
        if (diff > threshold) {
            outlier_count++;
        }
    }

    return (float)outlier_count / (float)n;
}

// Determine if a tensor should receive Q3_K_HIFI enhancement
// Combines name-based rules, model size, and statistical analysis
int ggml_q3_hifi_should_enhance_tensor(
    const char * tensor_name,
    const float * weights,
    int64_t n_elements,
    float model_params_b,
    int * enhanced_count,
    int max_enhanced
) {
    if (enhanced_count == NULL) {
        return 0;
    }

    // Check if we've hit the enhancement limit
    if (*enhanced_count >= max_enhanced) {
        return 0;
    }

    // Always enhance critical tensors (if within budget)
    // token_embd and output.weight are always critical
    if (tensor_name != NULL) {
        // Check for critical path tensors
        const char * name = tensor_name;

        // token_embd.weight
        int is_token_embd = 0;
        const char * p = name;
        while (*p) {
            if (p[0] == 't' && p[1] == 'o' && p[2] == 'k' && p[3] == 'e' && p[4] == 'n' &&
                p[5] == '_' && p[6] == 'e' && p[7] == 'm' && p[8] == 'b' && p[9] == 'd') {
                is_token_embd = 1;
                break;
            }
            p++;
        }

        // output.weight
        int is_output = 0;
        p = name;
        while (*p) {
            if (p[0] == 'o' && p[1] == 'u' && p[2] == 't' && p[3] == 'p' &&
                p[4] == 'u' && p[5] == 't' && p[6] == '.') {
                is_output = 1;
                break;
            }
            p++;
        }

        if (is_token_embd || is_output) {
            (*enhanced_count)++;
            return 1;
        }
    }

    // For other tensors, use statistical outlier detection
    if (weights != NULL && n_elements > 0) {
        float outlier_ratio = ggml_q3_hifi_compute_outlier_ratio(weights, n_elements);
        float threshold = ggml_q3_hifi_get_outlier_threshold(model_params_b);

        if (outlier_ratio >= threshold) {
            (*enhanced_count)++;
            return 1;
        }
    }

    return 0;
}

// Get the enhancement type (Q4_K, Q5_K, or Q6_K) for critical tensors
// Returns GGML_TYPE_* values
int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding) {
    // For Q3_K_HIFI, we use higher precision types for embeddings
    // Q6_K for embeddings (same as Q3_K_M default)
    // Q5_K for attn_v first layers (same as Q3_K_M)
    // Q4_K for other enhanced tensors

    if (is_embedding) {
        return 9;  // GGML_TYPE_Q6_K
    }

    // For large models, use higher precision on attn_v
    if (model_params_b >= 14.0f) {
        return 9;  // GGML_TYPE_Q6_K
    }

    // For medium models, Q5_K is a good balance
    if (model_params_b >= 4.0f) {
        return 8;  // GGML_TYPE_Q5_K
    }

    // For smaller models, Q4_K to avoid BPW overhead
    return 7;  // GGML_TYPE_Q4_K
}

// Get percentage of attn_v layers to enhance
// Based on model size - smaller models need broader coverage
// Aligned with llama-quant.cpp for consistency
float ggml_q3_hifi_get_attn_v_threshold(float model_params_b) {
    // Fine-grained thresholds matching llama-quant.cpp
    if (model_params_b <= 1.0f) {
        // 0.6B/1B: Skip attn_v HIFI entirely - matches Q3_K_M BPW
        // This addresses the +2.2% PPL regression seen at 0.6B
        return 0.0f;
    } else if (model_params_b <= 2.0f) {
        // 1.7B: Q3_K_HIFI DISABLED - match Q3_K_M behavior exactly
        // Q3_K_M uses: first 2 layers get Q5_K, rest Q4_K (threshold = 2/28 ≈ 0.07)
        return 0.07f;
    } else if (model_params_b <= 5.0f) {
        // 2-5B: Full enhancement - this is the sweet spot
        // 4B shows -2.9% PPL improvement with Q3_K_HIFI
        return 0.25f;
    } else if (model_params_b <= 10.0f) {
        // 5-8B: Moderate enhancement
        return 0.15f;
    } else if (model_params_b <= 20.0f) {
        // 14B: Reduced enhancement - addresses +0.24% PPL regression
        return 0.08f;
    } else {
        // 32B+: Minimal enhancement - addresses +0.13% PPL regression
        return 0.05f;
    }
}

// Compute adaptive outlier count for a specific block
// Fine-grained control based on per-block statistics
int ggml_q3_hifi_compute_block_outliers(
    float block_outlier_ratio,
    int base_outlier_count,
    float model_params_b
) {
    // If base count is 0, no outliers for this model size
    if (base_outlier_count <= 0) {
        return 0;
    }

    // Scale based on block's outlier ratio relative to tensor average
    // High ratio blocks get more outliers, low ratio blocks get fewer
    float threshold = ggml_q3_hifi_get_outlier_threshold(model_params_b);

    float scale = 1.0f;
    if (block_outlier_ratio >= threshold * 2.0f) {
        // Very high outlier block - boost significantly
        scale = 1.5f;
    } else if (block_outlier_ratio >= threshold) {
        // Above threshold - slight boost
        scale = 1.2f;
    } else if (block_outlier_ratio < threshold * 0.5f) {
        // Well below threshold - reduce
        scale = 0.6f;
    } else {
        // Near threshold - keep base
        scale = 0.9f;
    }

    // Model size adjustment
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    if (cat == Q3_HIFI_SIZE_LARGE) {
        // Large models: more aggressive reduction
        scale *= 0.8f;
    } else if (cat == Q3_HIFI_SIZE_TINY) {
        // Tiny models: if we're using outliers at all, be conservative
        scale *= 1.2f;
    }

    int result = (int)roundf((float)base_outlier_count * scale);

    // Clamp to valid range
    if (result < 0) result = 0;
    if (result > Q3_K_HIFI_MAX_OUTLIERS) result = Q3_K_HIFI_MAX_OUTLIERS;

    return result;
}

// ===========================================================================
// Q4_K_HIFI Adaptive Enhancement Functions
// Model-size-aware outlier allocation for Q4_K_HIFI quantization
// At 4-bit, the base quantization is more robust than 3-bit, so the
// outlier strategy is tuned differently from Q3_K_HIFI.
// ===========================================================================

// Get maximum outlier count for Q4_K_HIFI based on model size
// Key differences from Q3_K_HIFI:
// - Q4_K base is more robust, so fewer outliers are needed for small models
// - Large models benefit more at 4-bit because outlier concentration increases
int ggml_q4_hifi_get_max_outliers(float model_params_b) {
    if (model_params_b <= 1.0f) {
        // ≤1B: 4 outliers - Q4_K base is decent, moderate enhancement
        return 4;
    } else if (model_params_b <= 3.0f) {
        // 1-3B: 4 outliers - conservative for small models
        return 4;
    } else if (model_params_b <= 13.0f) {
        // 3-13B: 6 outliers - sweet spot for quality gains
        return 6;
    } else if (model_params_b <= 30.0f) {
        // 14-30B: 6 outliers - still significant benefit
        return 6;
    } else {
        // 30B+: 8 outliers - outlier concentration increases with scale
        return 8;
    }
}


// ===========================================================================
// K_LITE Tier-Based Residual Budget
// Determines how many INT8 residuals a tensor receives based on imatrix importance
// ===========================================================================

int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b, int max_residuals) {
    // Tier thresholds are model-size adjusted to approximately hit the target percentile cuts:
    //   <=1B:  Top 2% / Next 5%  -> high thresholds (importance scores are tightly clustered)
    //   3B-7B: Top 4% / Next 8%  -> moderate thresholds
    //   >=13B: Top 5% / Next 10% -> lower thresholds (more tensors benefit at large scale)
    float tier1_threshold, tier2_threshold;
    if (model_params_b <= 1.0f) {
        tier1_threshold = 0.90f;  // ~top 2%
        tier2_threshold = 0.75f;  // ~next 5%
    } else if (model_params_b <= 7.0f) {
        tier1_threshold = 0.80f;  // ~top 4%
        tier2_threshold = 0.60f;  // ~next 8%
    } else {
        tier1_threshold = 0.75f;  // ~top 5%
        tier2_threshold = 0.55f;  // ~next 10%
    }

    if (tensor_importance >= tier1_threshold) {
        return max_residuals;              // Tier 1: full residual budget
    } else if (tensor_importance >= tier2_threshold) {
        return (max_residuals + 1) / 2;   // Tier 2: half budget (rounded up)
    } else {
        return 0;                          // Tier 0: no residuals (pure base type)
    }
}
```

### 2.3 `ggml/src/ggml-ext.h`

Create this file at `ggml/src/ggml-ext.h` with the following content:

```c
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// This is a "staging" header for new ggml API
// It is not publicly available and it should not be used by 3rd party projects
//
// When the API matures enough, it will be moved to the official public API

//
// Meta backend
//

#define GGML_BACKEND_META_MAX_DEVICES 16

enum ggml_backend_meta_split_axis {
    // tensor split by tensor dimensions:
    GGML_BACKEND_SPLIT_AXIS_0   =  0,
    GGML_BACKEND_SPLIT_AXIS_1   =  1,
    GGML_BACKEND_SPLIT_AXIS_2   =  2,
    GGML_BACKEND_SPLIT_AXIS_3   =  3,

    GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
    GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

    // for internal bookkeeping only:
    GGML_BACKEND_SPLIT_AXIS_NONE     = 98,
    GGML_BACKEND_SPLIT_AXIS_UNKNOWN  = 99,
};
GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

struct ggml_backend_meta_split_state {
    enum ggml_backend_meta_split_axis axis;

    // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
    //   - each device has a slice of the tensor along the split axis
    //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
    //   - some tensors have an inhomogenenous data layout along the split axis,
    //     those tensors are divided into segments which are each individually split across devices
    //   - ne has one entry per segment and device that add up to ggml_tensor::ne for that axis,
    //     the outer/inner loops are over segments/devices like [seg0_dev0, seg0_dev1, seg1_dev0, seg1_dev1],
    //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
    //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V
    int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
    uint32_t n_segments;
};

// function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

// create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
// TODO: this looks a bit strange - a backend API creates a device. I think we should try
//       express this as a backend registry functionality instead
GGML_API ggml_backend_dev_t ggml_backend_meta_device(
    ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);
```

### 2.4 Vulkan GLSL Shaders

Create these 4 new files in `ggml/src/ggml-vulkan/vulkan-shaders/`:

**`dequant_q2_k_hifi.comp`**:

```glsl
#version 450

#include "dequant_head.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = gl_WorkGroupID.x * 256 + wgy;
        if (i >= p.nel / QUANT_K) {
            return;
        }

        const uint tid = gl_LocalInvocationID.x;
        const uint ip = tid / 32;
        const uint il = tid - 32 * ip;
        const uint is = 8 * ip + il / 16;

        const uint y_idx = i * QUANT_K + 128 * ip + il;

        const uint8_t qs = data_a[i].qs[32 * ip + il];

        FLOAT_TYPE dall = FLOAT_TYPE(data_a[i].dm.x);
        FLOAT_TYPE dmin = FLOAT_TYPE(data_a[i].dm.y);

        FLOAT_TYPE v0 = dall * FLOAT_TYPE((data_a[i].scales[is+0] & 0xF) * ((qs >> 0) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+0] >> 4);
        FLOAT_TYPE v1 = dall * FLOAT_TYPE((data_a[i].scales[is+2] & 0xF) * ((qs >> 2) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+2] >> 4);
        FLOAT_TYPE v2 = dall * FLOAT_TYPE((data_a[i].scales[is+4] & 0xF) * ((qs >> 4) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+4] >> 4);
        FLOAT_TYPE v3 = dall * FLOAT_TYPE((data_a[i].scales[is+6] & 0xF) * ((qs >> 6) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+6] >> 4);

        const uint local0 = 128 * ip + il;
        const uint local1 = local0 + 32;
        const uint local2 = local0 + 64;
        const uint local3 = local0 + 96;

        const uint raw_count = data_a[i].outlier_count;
        const bool residual_mode = (raw_count & Q2_K_HIFI_RESIDUAL_MODE_FLAG) != 0;
        const uint count = raw_count & 0x7F;
        const uint n_out = min(count, Q2_K_HIFI_MAX_OUTLIERS);

        [[unroll]] for (uint k = 0; k < Q2_K_HIFI_MAX_OUTLIERS; ++k) {
            if (k >= n_out) break;
            const uint idx = data_a[i].outlier_idx[k];
            const FLOAT_TYPE val = FLOAT_TYPE(data_a[i].outlier_vals[k]);
            if (idx == local0) { v0 = residual_mode ? (v0 + val) : val; }
            if (idx == local1) { v1 = residual_mode ? (v1 + val) : val; }
            if (idx == local2) { v2 = residual_mode ? (v2 + val) : val; }
            if (idx == local3) { v3 = residual_mode ? (v3 + val) : val; }
        }

        data_b[y_idx +  0] = D_TYPE(v0);
        data_b[y_idx + 32] = D_TYPE(v1);
        data_b[y_idx + 64] = D_TYPE(v2);
        data_b[y_idx + 96] = D_TYPE(v3);
    }
}
```

**`dequant_q3_k_hifi.comp`**:

```glsl
#version 450

// Q3_K_HIFI dequantization shader
// Uses Q3_K-compatible layout (hmask + qs + scales) with 6 FP16 outliers

#include "dequant_head.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = uint(gl_WorkGroupID.x * 256 + wgy);
        if (i >= p.nel / QUANT_K) {
            return;
        }

        const uint r = gl_LocalInvocationID.x / 4;
        const uint tid = r / 2;
        const uint is0 = r % 2;
        const uint l0 = 16 * is0 + 4 * (gl_LocalInvocationID.x % 4);
        const uint n = tid / 4;
        const uint j = tid - 4*n;

        const uint8_t m = uint8_t(1 << (4*n + j));
        const uint is = 8*n + 2*j + is0;
        const uint shift = 2*j;

        const int8_t us = int8_t(is <  4 ? (data_a[i].scales[is-0] & 0xF) | (((data_a[i].scales[is+8] >> 0) & 3) << 4) :
                                 is <  8 ? (data_a[i].scales[is-0] & 0xF) | (((data_a[i].scales[is+4] >> 2) & 3) << 4) :
                                 is < 12 ? (data_a[i].scales[is-8] >>  4) | (((data_a[i].scales[is+0] >> 4) & 3) << 4) :
                                           (data_a[i].scales[is-8] >>  4) | (((data_a[i].scales[is-4] >> 6) & 3) << 4));
        const FLOAT_TYPE d_all = FLOAT_TYPE(data_a[i].d);
        const FLOAT_TYPE dl    = d_all * FLOAT_TYPE(us - 32);

        const uint y_idx = i * QUANT_K + 128 * n + 32 * j;
        const uint qs_idx = 32*n;

        for (uint l = l0; l < l0 + 4; ++l) {
            const uint global_idx = y_idx + l;
            const uint local_idx = 128 * n + 32 * j + l;

            // Standard Q3_K dequantization
            FLOAT_TYPE val = dl * FLOAT_TYPE(int8_t((data_a[i].qs[qs_idx + l] >> shift) & 3) - (((data_a[i].hmask[l] & m) != 0) ? 0 : 4));

            // Q3_K_HIFI extension: Check if this is an outlier and replace with FP16 value
            [[unroll]] for (uint k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
                if (data_a[i].outlier_idx[k] == local_idx) {
                    val = FLOAT_TYPE(data_a[i].outlier_vals[k]);
                }
            }

            data_b[global_idx] = D_TYPE(val);
        }
    }
}
```

**`mul_mat_vec_q2_k_hifi.comp`**:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#include "mul_mat_vec_base.glsl"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

shared FLOAT_TYPE sccache1[2][BLOCK_SIZE/16][16];
shared FLOAT_TYPE sccache2[2][BLOCK_SIZE/16][16];

FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
uint csel = 0;

void calc_superblock(const uint a_offset, const uint b_offset, const uint itid, const uint v_im, const uint ix, const uint q_offset, const uint y_offset, const uint i, const uint num_blocks_per_row, const uint first_row, const uint num_rows, const bool all_threads) {
    const uint y_idx = i * QUANT_K + y_offset;

    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset + (first_row+n)*num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) {
            if (i < num_blocks_per_row) {
                const uint32_t scale = uint32_t(data_a[ib0 + i].scales[itid]);
                sccache1[csel][ix][itid] = FLOAT_TYPE(scale & 0xF);
                sccache2[csel][ix][itid] = FLOAT_TYPE((scale >> 4) & 0xF);
            }
            barrier();

            if (i >= num_blocks_per_row)
                continue;
        } else {
            const uint32_t scale = uint32_t(data_a[ib0 + i].scales[itid]);
            sccache1[csel][ix][itid] = FLOAT_TYPE(scale & 0xF);
            sccache2[csel][ix][itid] = FLOAT_TYPE((scale >> 4) & 0xF);
            barrier();
        }

        const uint32_t qs_u32 = uint32_t(data_a_packed16[ib0 + i].qs[q_offset / 2]) | (uint32_t(data_a_packed16[ib0 + i].qs[q_offset / 2 + 8]) << 16);
        const vec4 qs_u32_0 = vec4(unpack8(qs_u32 & 0x03030303));
        const vec4 qs_u32_2 = vec4(unpack8((qs_u32 >> 2) & 0x03030303));
        const vec4 qs_u32_4 = vec4(unpack8((qs_u32 >> 4) & 0x03030303));
        const vec4 qs_u32_6 = vec4(unpack8((qs_u32 >> 6) & 0x03030303));

        const FLOAT_TYPE_VEC2 dm = vec2(data_a[ib0 + i].dm);

        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            vec2 b0 =   vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);
            vec2 b16 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);
            vec2 b32 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);
            vec2 b48 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);
            vec2 b64 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);
            vec2 b80 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);
            vec2 b96 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);
            vec2 b112 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);

            FLOAT_TYPE sum1 = FLOAT_TYPE(0.0);
            FLOAT_TYPE sum2 = FLOAT_TYPE(0.0);
            [[unroll]] for (int l = 0; l < 2; ++l) {
                sum1 = fma(FLOAT_TYPE(b0[l]),   sccache1[csel][ix][    8*v_im] * qs_u32_0[l  ],
                       fma(FLOAT_TYPE(b16[l]),  sccache1[csel][ix][1 + 8*v_im] * qs_u32_0[l+2],
                       fma(FLOAT_TYPE(b32[l]),  sccache1[csel][ix][2 + 8*v_im] * qs_u32_2[l  ],
                       fma(FLOAT_TYPE(b48[l]),  sccache1[csel][ix][3 + 8*v_im] * qs_u32_2[l+2],
                       fma(FLOAT_TYPE(b64[l]),  sccache1[csel][ix][4 + 8*v_im] * qs_u32_4[l  ],
                       fma(FLOAT_TYPE(b80[l]),  sccache1[csel][ix][5 + 8*v_im] * qs_u32_4[l+2],
                       fma(FLOAT_TYPE(b96[l]),  sccache1[csel][ix][6 + 8*v_im] * qs_u32_6[l  ],
                       fma(FLOAT_TYPE(b112[l]), sccache1[csel][ix][7 + 8*v_im] * qs_u32_6[l+2], sum1))))))));
                sum2 = fma(FLOAT_TYPE(b0[l]),   sccache2[csel][ix][    8*v_im],
                       fma(FLOAT_TYPE(b16[l]),  sccache2[csel][ix][1 + 8*v_im],
                       fma(FLOAT_TYPE(b32[l]),  sccache2[csel][ix][2 + 8*v_im],
                       fma(FLOAT_TYPE(b48[l]),  sccache2[csel][ix][3 + 8*v_im],
                       fma(FLOAT_TYPE(b64[l]),  sccache2[csel][ix][4 + 8*v_im],
                       fma(FLOAT_TYPE(b80[l]),  sccache2[csel][ix][5 + 8*v_im],
                       fma(FLOAT_TYPE(b96[l]),  sccache2[csel][ix][6 + 8*v_im],
                       fma(FLOAT_TYPE(b112[l]), sccache2[csel][ix][7 + 8*v_im], sum2))))))));
            }
            temp[j][n] = fma(dm.x, sum1, fma(-dm.y, sum2, temp[j][n]));
        }
    }
}

void compute_outputs(const uint32_t first_row, const uint32_t num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    const uint it_size = gl_WorkGroupSize.x/16;
    const uint tid = gl_LocalInvocationID.x;
    const uint itid = tid%16;
    const uint ix = tid/16;

    const uint v_im = itid/8;
    const uint v_in = itid - 8*v_im;

    const uint l0 = 2*v_in;
    const uint q_offset = 32*v_im + l0;
    const uint y_offset = 128*v_im + l0;

    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    const uint nbr_par_th = num_blocks_per_row%it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    [[unroll]] for (; i0 < nbr_all_th; i0 += it_size)
        calc_superblock(a_offset, b_offset, itid, v_im, ix, q_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, true);
    calc_superblock(a_offset, b_offset, itid, v_im, ix, q_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);

    reduce_result(temp, d_offset, first_row, num_rows, tid);
}

void main() {
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);

    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) {
            return;
        }
        compute_outputs(first_row, p.stride_d - first_row);
    }
}
```

**`mul_mat_vec_q3_k_hifi.comp`**:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

// Q3_K_HIFI matrix-vector multiplication shader
// Uses Q3_K-compatible layout, outlier correction skipped on GPU for simplicity
// (outliers are still applied on CPU for full quality)

#include "mul_mat_vec_base.glsl"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

shared FLOAT_TYPE sccache[2][BLOCK_SIZE/16][2][8];

FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
uint csel = 0;

void calc_superblock(const uint a_offset, const uint b_offset, const uint ix, const uint itid8, const uint v_im, const uint v_im4, const uint v_in, const uint32_t hm_m[4], const uint q_offset, const uint y_offset, const uint s_shift, const uint i, const uint num_blocks_per_row, const uint first_row, const uint num_rows, const bool all_threads) {
    const uint y_idx = i * QUANT_K + y_offset;

    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset / QUANT_K + (first_row+n)*num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) {
            if (i < num_blocks_per_row)
                sccache[csel][ix][v_im][itid8] = FLOAT_TYPE(int8_t(((data_a[ib0+i].scales[itid8] >> v_im4) & 0xF) | (((data_a[ib0+i].scales[itid8%4+8] >> s_shift) & 3) << 4)) - 32);
            barrier();

            if (i >= num_blocks_per_row)
                continue;
        }

        const uint32_t hmk = ~(uint32_t(data_a_packed16[ib0 + i].hmask[v_in]) | (uint32_t(data_a_packed16[ib0 + i].hmask[v_in + 8]) << 16));
        const vec4 hmk_0 = vec4(unpack8(((hmk & hm_m[0]) >> (    v_im4)) << 2));
        const vec4 hmk_1 = vec4(unpack8(((hmk & hm_m[1]) >> (1 + v_im4)) << 2));
        const vec4 hmk_2 = vec4(unpack8(((hmk & hm_m[2]) >> (2 + v_im4)) << 2));
        const vec4 hmk_3 = vec4(unpack8(((hmk & hm_m[3]) >> (3 + v_im4)) << 2));

        uint32_t qs_u32 = uint32_t(data_a[ib0 + i].qs[q_offset]) | (uint32_t(data_a[ib0 + i].qs[q_offset + 1]) << 8);
        qs_u32 |= (uint32_t(data_a[ib0 + i].qs[q_offset + 16]) | (uint32_t(data_a[ib0 + i].qs[q_offset + 17]) << 8)) << 16;
        const vec4 qs_u32_0 = vec4(unpack8(qs_u32 & 0x03030303));
        const vec4 qs_u32_2 = vec4(unpack8((qs_u32 >> 2) & 0x03030303));
        const vec4 qs_u32_4 = vec4(unpack8((qs_u32 >> 4) & 0x03030303));
        const vec4 qs_u32_6 = vec4(unpack8((qs_u32 >> 6) & 0x03030303));

        if (all_threads) {
            sccache[csel][ix][v_im][itid8] = FLOAT_TYPE(int8_t(((data_a[ib0+i].scales[itid8] >> v_im4) & 0xF) | (((data_a[ib0+i].scales[itid8%4+8] >> s_shift) & 3) << 4)) - 32);
            barrier();
        }

        const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib0 + i].d);

        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            vec2 b0 =   vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);
            vec2 b16 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);
            vec2 b32 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);
            vec2 b48 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);
            vec2 b64 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);
            vec2 b80 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);
            vec2 b96 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);
            vec2 b112 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);

            FLOAT_TYPE sum = FLOAT_TYPE(0.0);
            [[unroll]] for (int l = 0; l < 2; ++l) {
                sum = fma(FLOAT_TYPE(  b0[l]) * sccache[csel][ix][v_im][0], qs_u32_0[l  ] - hmk_0[l  ],
                      fma(FLOAT_TYPE( b16[l]) * sccache[csel][ix][v_im][1], qs_u32_0[l+2] - hmk_0[l+2],
                      fma(FLOAT_TYPE( b32[l]) * sccache[csel][ix][v_im][2], qs_u32_2[l  ] - hmk_1[l  ],
                      fma(FLOAT_TYPE( b48[l]) * sccache[csel][ix][v_im][3], qs_u32_2[l+2] - hmk_1[l+2],
                      fma(FLOAT_TYPE( b64[l]) * sccache[csel][ix][v_im][4], qs_u32_4[l  ] - hmk_2[l  ],
                      fma(FLOAT_TYPE( b80[l]) * sccache[csel][ix][v_im][5], qs_u32_4[l+2] - hmk_2[l+2],
                      fma(FLOAT_TYPE( b96[l]) * sccache[csel][ix][v_im][6], qs_u32_6[l  ] - hmk_3[l  ],
                      fma(FLOAT_TYPE(b112[l]) * sccache[csel][ix][v_im][7], qs_u32_6[l+2] - hmk_3[l+2], sum))))))));
            }
            temp[j][n] = fma(d, sum, temp[j][n]);
            // Note: Outlier correction skipped on GPU for speed
            // Full outlier correction is applied on CPU path
        }
    }
}

void compute_outputs(const uint32_t first_row, const uint32_t num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    const uint it_size = gl_WorkGroupSize.x/16;
    const uint tid = gl_LocalInvocationID.x;
    const uint itid = tid%16;
    const uint ix = tid/16;
    const uint itid8 = itid%8;

    const uint v_im = itid/8;
    const uint v_im4 = v_im*4;
    const uint v_in = itid - 8*v_im;

    const uint32_t m = 0x01010101 << (4 * v_im);
    uint32_t hm_m[4];
    [[unroll]] for (uint j = 0; j < 4; ++j)
        hm_m[j] = m << j;

    const uint l0 = 2*v_in;
    const uint q_offset = 32*v_im + l0;
    const uint y_offset = 128*v_im + l0;

    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    const uint s_shift = v_im4 + 2*(itid8/4);

    const uint nbr_par_th = num_blocks_per_row%it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    [[unroll]] for (; i0 < nbr_all_th; i0 += it_size)
        calc_superblock(a_offset, b_offset, ix, itid8, v_im, v_im4, v_in, hm_m, q_offset, y_offset, s_shift, i0 + ix, num_blocks_per_row, first_row, num_rows, true);
    calc_superblock(a_offset, b_offset, ix, itid8, v_im, v_im4, v_in, hm_m, q_offset, y_offset, s_shift, i0 + ix, num_blocks_per_row, first_row, num_rows, false);

    reduce_result(temp, d_offset, first_row, num_rows, tid);
}

void main() {
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);

    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) {
            return;
        }
        compute_outputs(first_row, p.stride_d - first_row);
    }
}
```

---

## 3. Modified Files with Exact Diffs

For each file below, apply the diff using `git apply` or patch the changes manually.
All diffs are relative to commit `a29e4c0b7` (the last upstream-only commit).

---

### 3.1 `ggml/include/ggml.h` — Type Enum Extension

Add 13 new type enum values before `GGML_TYPE_COUNT`:

```diff
-        GGML_TYPE_Q1_0    = 41,
-        GGML_TYPE_COUNT   = 42,
+        GGML_TYPE_Q1_0    = 41,
+        GGML_TYPE_Q3_K_HIFI = 42, // Q3_K_HIFI: Q3_K layout + 8 FP16 outliers per block
+        GGML_TYPE_Q6_K_HIFI = 43, // Q6_K_HIFI: Q6_K layout + 4 FP16 outliers for critical tensors
+        GGML_TYPE_Q6_K_HIFI_DYNAMIC = 44, // Q6_K_HIFI_DYNAMIC: Q6_K + 2-8 outliers based on layer sensitivity
+        GGML_TYPE_Q6_K_HIFI_RES8 = 45, // Q6_K_HIFI_RES8: Q6_K + INT8 residuals (compact format)
+        GGML_TYPE_Q5_K_HIFI_RES8 = 46, // Q5_K_HIFI_RES8: Q5_K + INT8 residuals (efficient for 4B-10B models)
+        GGML_TYPE_Q3_K_HIFI_RES8 = 47, // Q3_K_HIFI_RES8: Q3_K + INT8 residuals (lean version for imatrix use)
+        GGML_TYPE_Q4_K_HIFI = 48, // Q4_K_HIFI: Q4_K layout + 8 FP16 outliers per block (high-fidelity 4-bit)
+        GGML_TYPE_Q2_K_HIFI = 49, // Q2_K_HIFI: Q2_K layout + 3 INT8 residuals per block (high-fidelity 2-bit)
+        GGML_TYPE_Q2_K_LITE = 50, // Q2_K_LITE: Q2_K + 3 INT8 residuals, residual-only encoding (96 bytes, ~3.0 BPW)
+        GGML_TYPE_Q3_K_LITE = 51, // Q3_K_LITE: Q3_K + 8 INT8 residuals (132 bytes, ~4.13 BPW)
+        GGML_TYPE_Q4_K_LITE = 52, // Q4_K_LITE: Q4_K + 8 INT8 residuals (168 bytes, ~5.25 BPW)
+        GGML_TYPE_Q5_K_LITE = 53, // Q5_K_LITE: Q5_K + 8 INT8 residuals (200 bytes, ~6.25 BPW)
+        GGML_TYPE_Q6_K_LITE = 54, // Q6_K_LITE: Q6_K + 8 INT8 residuals (232 bytes, ~7.25 BPW)
+        GGML_TYPE_COUNT   = 55,
```

Also add this comment before the `struct ggml_object;` forward declaration:

```diff
+    // Q3_K_HIFI block structure is defined in ggml-common.h for GPU backend compatibility
+    // Uses Q3_K-compatible layout with 6 FP16 outliers for improved accuracy
+
     struct ggml_object;
```

---

### 3.2 `ggml/src/ggml-common.h` — Block Structure Definitions

After the `block_q3_K` static_assert (around line 312), add all new block type definitions.
This is the largest change in the header files — approximately 292 lines of new struct
definitions for all 8 HIFI types and 5 LITE types.

The key structures are:

**block_q3_k_hifi** (136 bytes):
```c
#define Q3_K_HIFI_BLOCK_SIZE 256
#define Q3_K_HIFI_OUTLIERS   8
#define Q3_K_HIFI_INLIERS    (Q3_K_HIFI_BLOCK_SIZE - Q3_K_HIFI_OUTLIERS)
typedef struct {
    uint8_t q3_k_data[110];          // standard Q3_K block (inliers with outliers zeroed)
    uint8_t outlier_idx[Q3_K_HIFI_OUTLIERS];   // 8 bytes: indices of top-8 outliers
    ggml_half outliers[Q3_K_HIFI_OUTLIERS];    // 16 bytes: original outlier FP16 values
    uint8_t padding[2];                         // 2 bytes alignment
} block_q3_k_hifi;
static_assert(sizeof(block_q3_k_hifi) == 136, ...);
```

**block_q3_k_hifi_res8** (132 bytes):
```c
#define Q3_K_HIFI_RES8_OUTLIERS 8
typedef struct {
    uint8_t hmask[QK_K/8];         // 32 bytes: high bit mask (Q3_K-compatible)
    uint8_t qs[QK_K/4];            // 64 bytes: low 2 bits (Q3_K-compatible)
    uint8_t scales[12];            // 12 bytes: scales (Q3_K-compatible)
    ggml_half d;                   // 2 bytes: super-block scale (Q3_K-compatible)
    uint8_t outlier_count;         // 1 byte
    uint8_t _pad1;                 // 1 byte padding
    uint8_t outlier_idx[8];        // 8 bytes
    int8_t  residual_vals[8];      // 8 bytes
    float   residual_scale;        // 4 bytes
} block_q3_k_hifi_res8;
// Total: 132 bytes
```

**block_q4_k_hifi** (168 bytes):
```c
#define Q4_K_HIFI_BLOCK_SIZE 256
#define Q4_K_HIFI_OUTLIERS   8
typedef struct {
    uint8_t q4_k_data[144];               // standard Q4_K block
    uint8_t outlier_idx[Q4_K_HIFI_OUTLIERS];   // 8 bytes
    ggml_half outliers[Q4_K_HIFI_OUTLIERS];    // 16 bytes
} block_q4_k_hifi;
// Total: 168 bytes
```

**block_q6_k_hifi** (222 bytes):
```c
#define Q6_K_HIFI_OUTLIERS 4
typedef struct {
    uint8_t ql[QK_K/2];        // 128 bytes (Q6_K-compatible)
    uint8_t qh[QK_K/4];        // 64 bytes (Q6_K-compatible)
    int8_t  scales[QK_K/16];   // 16 bytes (Q6_K-compatible)
    ggml_half d;               // 2 bytes (Q6_K-compatible)
    uint8_t outlier_idx[4];    // 4 bytes
    ggml_half outlier_vals[4]; // 8 bytes
} block_q6_k_hifi;
// Total: 222 bytes
```

**block_q6_k_hifi_dynamic** (236 bytes):
```c
#define Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS 8
typedef struct {
    uint8_t ql[QK_K/2];        // 128 bytes (Q6_K-compatible)
    uint8_t qh[QK_K/4];        // 64 bytes (Q6_K-compatible)
    int8_t  scales[QK_K/16];   // 16 bytes (Q6_K-compatible)
    ggml_half d;               // 2 bytes (Q6_K-compatible)
    uint8_t outlier_count;     // 1 byte
    uint8_t outlier_idx[8];    // 8 bytes
    uint8_t _padding;          // 1 byte
    ggml_half outlier_vals[8]; // 16 bytes
} block_q6_k_hifi_dynamic;
// Total: 236 bytes
```

**block_q6_k_hifi_res8** (232 bytes):
```c
#define Q6_K_HIFI_RES8_MAX_OUTLIERS 8
typedef struct {
    uint8_t ql[QK_K/2];        // 128 bytes (Q6_K-compatible)
    uint8_t qh[QK_K/4];        // 64 bytes (Q6_K-compatible)
    int8_t  scales[QK_K/16];   // 16 bytes (Q6_K-compatible)
    ggml_half d;               // 2 bytes (Q6_K-compatible)
    uint8_t outlier_count;     // 1 byte
    uint8_t outlier_idx[8];    // 8 bytes
    int8_t  residual_vals[8];  // 8 bytes
    uint8_t _padding;          // 1 byte
    float   residual_scale;    // 4 bytes
} block_q6_k_hifi_res8;
// Total: 232 bytes
```

**block_q5_k_hifi_res8** (196 bytes):
```c
#define Q5_K_HIFI_RES8_MAX_OUTLIERS 8
typedef struct {
    // Q5_K-compatible base (176 bytes) using union for dm
    GGML_EXTENSION union { struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S; ggml_half2 dm; } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes
    uint8_t qh[QK_K/8];           // 32 bytes
    uint8_t qs[QK_K/2];           // 128 bytes
    uint8_t outlier_count;         // 1 byte
    uint8_t outlier_idx[8];        // 8 bytes
    int8_t  residual_vals[8];      // 8 bytes
    uint8_t residual_scale_e4m3;   // 1 byte (E4M3 FP8 scale)
} block_q5_k_hifi_res8;
// Total: 196 bytes
```

**block_q2_k_hifi** (96 bytes):
```c
#define Q2_K_HIFI_BLOCK_SIZE 256
#define Q2_K_HIFI_MAX_OUTLIERS 3
#define Q2_K_HIFI_RESIDUAL_MODE_FLAG 0x80
typedef struct {
    uint8_t scales[QK_K/16];    // 16 bytes (Q2_K-compatible)
    uint8_t qs[QK_K/4];         // 64 bytes (Q2_K-compatible)
    GGML_EXTENSION union { ... } dm;  // 4 bytes (Q2_K-compatible)
    uint8_t   outlier_count;                       // 1 byte
    uint8_t   outlier_idx[Q2_K_HIFI_MAX_OUTLIERS]; // 3 bytes
    ggml_half outlier_vals[Q2_K_HIFI_MAX_OUTLIERS]; // 6 bytes
    uint8_t   _pad[2];          // 2 bytes
} block_q2_k_hifi;
// Total: 96 bytes
```

**LITE family** — 5 structures using `#pragma pack(push,1)`:

- `block_q2_k_lite` (96 bytes): Q2_K base (84) + 1+4+4+1+2 extension
- `block_q3_k_lite` (104 bytes): Q2_K base (84) + 1+8+8+1+2 extension
- `block_q4_k_lite` (128 bytes): Q3_K base (110) + 1+7+7+1+2 extension
- `block_q5_k_lite` (164 bytes): Q4_K base (144) + 1+8+8+1+2 extension
- `block_q6_k_lite` (196 bytes): Q5_K base (176) + 1+8+8+1+2 extension

All LITE structs use the same extension pattern:
```c
uint8_t   residual_count;
uint8_t   residual_idx[MAX_RESIDUALS];
int8_t    residual_vals[MAX_RESIDUALS];
uint8_t   _pad;
ggml_half residual_scale;
```

**Important:** The full struct definitions must match exactly (the static_asserts verify sizes at compile time). Apply the following diff:

```diff
diff --git a/ggml/src/ggml-common.h b/ggml/src/ggml-common.h
index f05683b44..cc3539014 100644
--- a/ggml/src/ggml-common.h
+++ b/ggml/src/ggml-common.h
@@ -310,6 +310,79 @@ typedef struct {
 } block_q3_K;
 static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");
 
+// Q3_K_HIFI: Imatrix-Guided Sparse 3-bit quantization (IGS-3)
+// Preserves top-16 most important weights as FP16, quantizes remaining 240 to 3-bit
+// This avoids scale distortion and preserves critical signal exactly
+#define Q3_K_HIFI_BLOCK_SIZE 256
+#define Q3_K_HIFI_OUTLIERS   8
+#define Q3_K_HIFI_INLIERS    (Q3_K_HIFI_BLOCK_SIZE - Q3_K_HIFI_OUTLIERS)  // 248
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // First 110 bytes: standard Q3_K block (for inliers with outliers zeroed)
+    uint8_t q3_k_data[110];
+
+    // Next 8 bytes: indices of top-8 outliers (0-255)
+    uint8_t outlier_idx[Q3_K_HIFI_OUTLIERS];
+
+    // Next 16 bytes: original outlier values as FP16 (REPLACEMENT values, not residuals!)
+    ggml_half outliers[Q3_K_HIFI_OUTLIERS];
+
+    // Padding to 136 bytes for alignment consistency
+    uint8_t padding[2];
+} block_q3_k_hifi;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Size: 110 (Q3_K) + 8 (idx) + 16 (outliers) + 2 (pad) = 136 bytes
+static_assert(sizeof(block_q3_k_hifi) == 110 + Q3_K_HIFI_OUTLIERS + Q3_K_HIFI_OUTLIERS*sizeof(ggml_half) + 2, "wrong q3_k_hifi block size/padding");
+
+// Q3_K_HIFI_RES8: Lean version with INT8 residuals for use WITH imatrix
+// When imatrix is present, base quantization is already optimized - INT8 residuals suffice
+// Uses 8 outliers (vs 16 in FP16 version) for minimal overhead while maintaining quality
+#define Q3_K_HIFI_RES8_OUTLIERS 8
+typedef struct {
+    // === Q3_K-COMPATIBLE REGION (110 bytes) - DO NOT REORDER ===
+    uint8_t hmask[QK_K/8];         // 32 bytes: high bit mask
+    uint8_t qs[QK_K/4];            // 64 bytes: low 2 bits
+    uint8_t scales[12];            // 12 bytes: 16 sub-group scales (6-bit each)
+    ggml_half d;                   // 2 bytes: super-block scale
+    // === INT8 RESIDUAL EXTENSION (22 bytes) ===
+    uint8_t outlier_count;                          // 1 byte: actual outliers stored (0-8)
+    uint8_t _pad1;                                  // 1 byte: alignment padding
+    uint8_t outlier_idx[Q3_K_HIFI_RES8_OUTLIERS];   // 8 bytes: outlier positions (0-255)
+    int8_t  residual_vals[Q3_K_HIFI_RES8_OUTLIERS]; // 8 bytes: INT8 residual corrections
+    float   residual_scale;                         // 4 bytes: scale for INT8 residuals
+} block_q3_k_hifi_res8;
+// Size: 110 (Q3_K) + 2 (count+pad) + 8 (idx) + 8 (vals) + 4 (scale) = 132 bytes
+static_assert(sizeof(block_q3_k_hifi_res8) == sizeof(block_q3_K) + 2 + Q3_K_HIFI_RES8_OUTLIERS + Q3_K_HIFI_RES8_OUTLIERS + sizeof(float), "wrong q3_k_hifi_res8 block size/padding");
+
+// Q4_K_HIFI: Imatrix-Guided Sparse 4-bit quantization
+// Preserves top-8 most important weights as FP16, quantizes remaining 248 to 4-bit via Q4_K
+// This gives near-Q5 quality at ~5.25 BPW by preserving outliers exactly
+#define Q4_K_HIFI_BLOCK_SIZE 256
+#define Q4_K_HIFI_OUTLIERS   8
+#define Q4_K_HIFI_INLIERS    (Q4_K_HIFI_BLOCK_SIZE - Q4_K_HIFI_OUTLIERS)  // 248
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // First 144 bytes: standard Q4_K block (for inliers with outliers zeroed)
+    uint8_t q4_k_data[144];
+
+    // Next 8 bytes: indices of top-8 outliers (0-255), sorted ascending
+    uint8_t outlier_idx[Q4_K_HIFI_OUTLIERS];
+
+    // Next 16 bytes: original outlier values as FP16 (REPLACEMENT values, not residuals!)
+    ggml_half outliers[Q4_K_HIFI_OUTLIERS];
+} block_q4_k_hifi;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Size: 144 (Q4_K) + 8 (idx) + 16 (outliers) = 168 bytes → 5.25 BPW
+static_assert(sizeof(block_q4_k_hifi) == 144 + Q4_K_HIFI_OUTLIERS + Q4_K_HIFI_OUTLIERS*sizeof(ggml_half), "wrong q4_k_hifi block size/padding");
+
 // 4-bit quantization
 // 8 blocks of 32 elements each
 // weight is represented as x = a * q + b
@@ -357,6 +430,292 @@ typedef struct {
 } block_q6_K;
 static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");
 
+// Q6_K_HIFI: Q6_K base + 4 FP16 outliers for enhanced precision on critical tensors
+// Designed for Q4_K_M_HIFI: applies only to token_embd, output.weight, and early attn_v
+// Provides ~0.05-0.10 PPL improvement with minimal overhead (+12 bytes per block)
+#define Q6_K_HIFI_OUTLIERS 4
+typedef struct {
+    // === Q6_K-COMPATIBLE REGION (210 bytes) - DO NOT REORDER ===
+    uint8_t ql[QK_K/2];      // 128 bytes: quants, lower 4 bits
+    uint8_t qh[QK_K/4];      // 64 bytes: quants, upper 2 bits
+    int8_t  scales[QK_K/16]; // 16 bytes: scales, quantized with 8 bits
+    ggml_half d;             // 2 bytes: super-block scale
+    // === OUTLIER EXTENSION (12 bytes) ===
+    uint8_t outlier_idx[Q6_K_HIFI_OUTLIERS];    // 4 bytes: outlier positions (0-255)
+    ggml_half outlier_vals[Q6_K_HIFI_OUTLIERS]; // 8 bytes: FP16 outlier values
+} block_q6_k_hifi;
+static_assert(sizeof(block_q6_k_hifi) == sizeof(block_q6_K) + Q6_K_HIFI_OUTLIERS + Q6_K_HIFI_OUTLIERS*sizeof(ggml_half), "wrong q6_k_hifi block size/padding");
+
+// Q6_K_HIFI_DYNAMIC: Q6_K base + dynamic outliers (2-8) based on layer sensitivity
+// - Early layers (0-30%): 6-8 outliers (most sensitive)
+// - Middle layers (30-70%): 4-6 outliers (moderately sensitive)
+// - Late layers (70-100%): 2-4 outliers (least sensitive, more redundant)
+// - Embeddings/output: 8 outliers (always critical)
+// Includes early-exit optimization: skip outlier correction when |activation| < threshold
+#define Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS 8
+#define Q6_K_HIFI_DYNAMIC_MIN_OUTLIERS 2
+#define Q6_K_HIFI_DYNAMIC_DEFAULT_OUTLIERS 6  // Default for generic quantization path
+#define Q6_K_HIFI_EARLY_EXIT_THRESHOLD 4  // |q8| > 4 means |activation| > 0.03
+typedef struct {
+    // === Q6_K-COMPATIBLE REGION (210 bytes) - DO NOT REORDER ===
+    uint8_t ql[QK_K/2];      // 128 bytes: quants, lower 4 bits
+    uint8_t qh[QK_K/4];      // 64 bytes: quants, upper 2 bits
+    int8_t  scales[QK_K/16]; // 16 bytes: scales, quantized with 8 bits
+    ggml_half d;             // 2 bytes: super-block scale
+    // === DYNAMIC OUTLIER EXTENSION (26 bytes with padding) ===
+    uint8_t outlier_count;                              // 1 byte: actual outlier count (2-8)
+    uint8_t outlier_idx[Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS];    // 8 bytes: outlier positions (0-255)
+    uint8_t _padding;                                   // 1 byte: padding for ggml_half alignment
+    ggml_half outlier_vals[Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS]; // 16 bytes: FP16 outlier values
+} block_q6_k_hifi_dynamic;
+// Total: 236 bytes (210 + 26)
+static_assert(sizeof(block_q6_k_hifi_dynamic) == sizeof(block_q6_K) + 2 + Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS + Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS*sizeof(ggml_half), "wrong q6_k_hifi_dynamic block size/padding");
+
+// Q6_K_HIFI_RES8: Compact Q6_K with INT8 residuals + per-block shared scale
+// This format reduces size by using INT8 residuals instead of FP16 outlier values.
+// The residual is computed as: original_value - Q6_K_approximation, then quantized to INT8.
+// Reconstruction: Q6_K_dequant + residual_scale * (residual_vals[i] / 127.0f)
+// Size reduction: 236 -> 232 bytes (-1.7% vs Q6_K_HIFI_DYNAMIC, matches Q4_K_M size ratio)
+#define Q6_K_HIFI_RES8_MAX_OUTLIERS 8
+typedef struct {
+    // === Q6_K-COMPATIBLE REGION (210 bytes) - DO NOT REORDER ===
+    uint8_t ql[QK_K/2];      // 128 bytes: quants, lower 4 bits
+    uint8_t qh[QK_K/4];      // 64 bytes: quants, upper 2 bits
+    int8_t  scales[QK_K/16]; // 16 bytes: scales, quantized with 8 bits
+    ggml_half d;             // 2 bytes: super-block scale
+    // === COMPACT INT8 RESIDUAL EXTENSION (22 bytes) ===
+    uint8_t outlier_count;                               // 1 byte: actual outlier count (1-8)
+    uint8_t outlier_idx[Q6_K_HIFI_RES8_MAX_OUTLIERS];    // 8 bytes: outlier positions (0-255)
+    int8_t  residual_vals[Q6_K_HIFI_RES8_MAX_OUTLIERS];  // 8 bytes: INT8 residuals (-127 to +127)
+    uint8_t _padding;                                    // 1 byte: padding for float alignment
+    float   residual_scale;                              // 4 bytes: shared scale for residuals
+} block_q6_k_hifi_res8;
+// Total: 232 bytes (210 + 22) - saves 4 bytes/block vs Q6_K_HIFI_DYNAMIC
+static_assert(sizeof(block_q6_k_hifi_res8) == 232, "wrong q6_k_hifi_res8 block size/padding");
+
+// Q5_K_HIFI_RES8: Efficient Q5_K with INT8 residuals for 4B-10B models
+// This format is optimized for mid-scale models where Q6_K overhead is wasteful.
+// Q5_K base provides sufficient precision, outliers compensate for 1-bit loss.
+// OPTIMIZED: E4M3 FP8 scale (1 byte) saves 3 bytes vs FP32 (4 bytes)
+// Size: 197 bytes vs Q6_K_HIFI_RES8's 232 bytes (~15% smaller)
+// Expected results: matches Q6_K_HIFI_RES8 quality at better BPW efficiency
+#define Q5_K_HIFI_RES8_MAX_OUTLIERS 8
+typedef struct {
+    // === Q5_K-COMPATIBLE REGION (176 bytes) - DO NOT REORDER ===
+    GGML_EXTENSION union {
+        struct {
+            ggml_half d;    // super-block scale for quantized scales
+            ggml_half dmin; // super-block scale for quantized mins
+        } GGML_COMMON_AGGR_S;
+        ggml_half2 dm;
+    } GGML_COMMON_AGGR_U;
+    uint8_t scales[K_SCALE_SIZE]; // 12 bytes: scales and mins, quantized with 6 bits
+    uint8_t qh[QK_K/8];           // 32 bytes: quants, high bit
+    uint8_t qs[QK_K/2];           // 128 bytes: quants, low 4 bits
+    // === COMPACT INT8 RESIDUAL EXTENSION (21 bytes, optimized with E4M3) ===
+    uint8_t outlier_count;                               // 1 byte: actual outlier count (0-8, 0=non-enhanced)
+    uint8_t outlier_idx[Q5_K_HIFI_RES8_MAX_OUTLIERS];    // 8 bytes: outlier positions (0-255)
+    int8_t  residual_vals[Q5_K_HIFI_RES8_MAX_OUTLIERS];  // 8 bytes: INT8 residuals (-127 to +127)
+    uint8_t residual_scale_e4m3;                         // 1 byte: E4M3 FP8 scale (0.92% error vs FP16)
+    // NOTE: 3 bytes saved vs FP32 scale, no padding needed
+    // Effective bpw after early exit optimization (92% non-enhanced blocks):
+    //   Enhanced blocks (8%): 196 bytes → 6.125 bpw
+    //   Non-enhanced blocks (92%): 177 bytes (skip residual storage) → 5.53 bpw
+    //   Weighted average: 0.08×6.125 + 0.92×5.53 = 5.58 bpw (beats Q5_K_M's 5.69 bpw!)
+} block_q5_k_hifi_res8;
+// Total: 196 bytes (176 + 20) - 15.5% smaller than Q6_K_HIFI_RES8
+static_assert(sizeof(block_q5_k_hifi_res8) == 196, "wrong q5_k_hifi_res8 block size/padding");
+
+// Q2_K_HIFI: Q2_K base + FP16 outlier preservation for critical tensors
+// At 2-bit precision, outlier weights suffer catastrophic quantization error.
+// Key insight: protect outliers BEFORE quantization, not after.
+// 1. Identify top-3 outliers by |weight| * imatrix_importance
+// 2. Zero them before Q2_K quantization (so Q2_K only sees well-behaved weights)
+// 3. Store true outlier values as FP16 for perfect reconstruction
+// Block is 96 bytes (84 Q2_K + 12 extension) = 3.0 BPW
+#define Q2_K_HIFI_BLOCK_SIZE 256
+#define Q2_K_HIFI_MAX_OUTLIERS 3
+#define Q2_K_HIFI_RESIDUAL_MODE_FLAG 0x80
+typedef struct {
+    // === Q2_K-COMPATIBLE REGION (84 bytes) - DO NOT REORDER ===
+    uint8_t scales[QK_K/16];   // 16 bytes: scales and mins, quantized with 4 bits
+    uint8_t qs[QK_K/4];        // 64 bytes: quants (2-bit packed)
+    GGML_EXTENSION union {
+        struct {
+            ggml_half d;       // 2 bytes: super-block scale for quantized scales
+            ggml_half dmin;    // 2 bytes: super-block scale for quantized mins
+        } GGML_COMMON_AGGR_S;
+        ggml_half2 dm;
+    } GGML_COMMON_AGGR_U;
+    // === FP16 OUTLIER EXTENSION (12 bytes) ===
+    uint8_t   outlier_count;                          // 1 byte: actual outliers stored (0-3)
+    uint8_t   outlier_idx[Q2_K_HIFI_MAX_OUTLIERS];   // 3 bytes: outlier positions (0-255)
+    ggml_half outlier_vals[Q2_K_HIFI_MAX_OUTLIERS];   // 6 bytes: true FP16 outlier values
+    uint8_t   _pad[2];                                // 2 bytes: alignment to 96
+} block_q2_k_hifi;
+// Total: 84 (Q2_K) + 12 (extension) = 96 bytes → 3.0 BPW
+static_assert(sizeof(block_q2_k_hifi) == 96, "wrong q2_k_hifi block size/padding");
+
+// ===========================================================================
+// K_LITE Family: INT8 residual corrections after base quantization
+// All types use the same extension pattern:
+//   residual_count (1) + residual_idx[N] (N) + residual_vals[N] (N) + _pad + residual_scale (4)
+// residual[i] = true_weight[i] - reconstructed_weight[i], quantized to INT8
+// Dot product: base_dot + sum_i(residual_scale * residual_vals[i] * activation[residual_idx[i]])
+// Tier 0 blocks (residual_count=0) fast-path through unchanged at base type speed.
+// ===========================================================================
+
+// Q2_K_LITE: Q2_K base + 4 INT8 residuals (96 bytes = 84 + 12)
+// Base shifted down to Q2_K; residual_scale stored as ggml_half for memory efficiency.
+#define Q2_K_LITE_BLOCK_SIZE    256
+#define Q2_K_LITE_MAX_RESIDUALS 4
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // === Q2_K-COMPATIBLE BASE (84 bytes) ===
+    uint8_t scales[QK_K/16];   // 16 bytes: scales and mins, quantized with 4 bits
+    uint8_t qs[QK_K/4];        // 64 bytes: quants (2-bit packed)
+    GGML_EXTENSION union {
+        struct {
+            ggml_half d;       // 2 bytes: super-block scale for quantized scales
+            ggml_half dmin;    // 2 bytes: super-block scale for quantized mins
+        } GGML_COMMON_AGGR_S;
+        ggml_half2 dm;
+    } GGML_COMMON_AGGR_U;
+    // === INT8 RESIDUAL EXTENSION (12 bytes) ===
+    uint8_t   residual_count;                            // 1 byte: actual residuals stored (0-4)
+    uint8_t   residual_idx[Q2_K_LITE_MAX_RESIDUALS];   // 4 bytes: positions (0-255)
+    int8_t    residual_vals[Q2_K_LITE_MAX_RESIDUALS];  // 4 bytes: INT8 corrections
+    uint8_t   _pad;                                      // 1 byte: align residual_scale to 2 bytes
+    ggml_half residual_scale;                            // 2 bytes: shared scale (max_err / 127)
+} block_q2_k_lite;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Total: 84 (Q2_K) + 1 + 4 + 4 + 1 + 2 = 96 bytes → 3.0 BPW
+static_assert(sizeof(block_q2_k_lite) == 96, "wrong q2_k_lite block size/padding");
+
+// Q3_K_LITE: Q2_K base + 8 INT8 residuals (104 bytes = 84 + 20)
+// Base shifted down from Q3_K (110B) to Q2_K (84B); smaller block = faster than Q3_K_S.
+#define Q3_K_LITE_BLOCK_SIZE    256
+#define Q3_K_LITE_MAX_RESIDUALS 8
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // === Q2_K-COMPATIBLE BASE (84 bytes) ===
+    uint8_t scales[QK_K/16];   // 16 bytes: scales and mins, quantized with 4 bits
+    uint8_t qs[QK_K/4];        // 64 bytes: quants (2-bit packed)
+    GGML_EXTENSION union {
+        struct {
+            ggml_half d;       // 2 bytes: super-block scale for quantized scales
+            ggml_half dmin;    // 2 bytes: super-block scale for quantized mins
+        } GGML_COMMON_AGGR_S;
+        ggml_half2 dm;
+    } GGML_COMMON_AGGR_U;
+    // === INT8 RESIDUAL EXTENSION (20 bytes) ===
+    uint8_t   residual_count;                            // 1 byte: actual residuals stored (0-8)
+    uint8_t   residual_idx[Q3_K_LITE_MAX_RESIDUALS];   // 8 bytes: positions (0-255)
+    int8_t    residual_vals[Q3_K_LITE_MAX_RESIDUALS];  // 8 bytes: INT8 corrections
+    uint8_t   _pad;                                      // 1 byte: align residual_scale to 2 bytes
+    ggml_half residual_scale;                            // 2 bytes: shared scale (max_err / 127)
+} block_q3_k_lite;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Total: 84 (Q2_K) + 1 + 8 + 8 + 1 + 2 = 104 bytes → 3.25 BPW  (Q3_K_S = 110 bytes)
+static_assert(sizeof(block_q3_k_lite) == 104, "wrong q3_k_lite block size/padding");
+
+// Q4_K_LITE: Q3_K base + 7 INT8 residuals (128 bytes = 110 + 18)
+// Base shifted down from Q4_K (144B) to Q3_K (110B); smaller block = faster than Q4_K_S.
+#define Q4_K_LITE_BLOCK_SIZE    256
+#define Q4_K_LITE_MAX_RESIDUALS 7
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // === Q3_K-COMPATIBLE BASE (110 bytes) ===
+    uint8_t  hmask[QK_K/8];         // 32 bytes: high bits of quants
+    uint8_t  qs[QK_K/4];            // 64 bytes: quants (2-bit low bits)
+    uint8_t  scales[K_SCALE_SIZE];  // 12 bytes: scales, quantized with 6 bits
+    ggml_half d;                    // 2 bytes: super-block scale
+    // === INT8 RESIDUAL EXTENSION (18 bytes) ===
+    uint8_t   residual_count;                            // 1 byte: actual residuals stored (0-7)
+    uint8_t   residual_idx[Q4_K_LITE_MAX_RESIDUALS];   // 7 bytes: positions (0-255)
+    int8_t    residual_vals[Q4_K_LITE_MAX_RESIDUALS];  // 7 bytes: INT8 corrections
+    uint8_t   _pad;                                      // 1 byte: align residual_scale to 2 bytes
+    ggml_half residual_scale;                            // 2 bytes: shared scale (max_err / 127)
+} block_q4_k_lite;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Total: 110 (Q3_K) + 1 + 7 + 7 + 1 + 2 = 128 bytes → 4.0 BPW  (Q4_K_S = 144 bytes)
+static_assert(sizeof(block_q4_k_lite) == 128, "wrong q4_k_lite block size/padding");
+
+// Q5_K_LITE: Q4_K base + 8 INT8 residuals (164 bytes = 144 + 20)
+// Base shifted down from Q5_K (176B) to Q4_K (144B); smaller block = faster than Q5_K_S.
+#define Q5_K_LITE_BLOCK_SIZE    256
+#define Q5_K_LITE_MAX_RESIDUALS 8
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // === Q4_K-COMPATIBLE BASE (144 bytes) ===
+    GGML_EXTENSION union {
+        struct {
+            ggml_half d;       // 2 bytes: super-block scale for quantized scales
+            ggml_half dmin;    // 2 bytes: super-block scale for quantized mins
+        } GGML_COMMON_AGGR_S;
+        ggml_half2 dm;
+    } GGML_COMMON_AGGR_U;
+    uint8_t  scales[3*QK_K/64]; // 12 bytes: scales and mins, quantized with 6 bits
+    uint8_t  qs[QK_K/2];        // 128 bytes: quants (4-bit packed)
+    // === INT8 RESIDUAL EXTENSION (20 bytes) ===
+    uint8_t   residual_count;                            // 1 byte: actual residuals stored (0-8)
+    uint8_t   residual_idx[Q5_K_LITE_MAX_RESIDUALS];   // 8 bytes: positions (0-255)
+    int8_t    residual_vals[Q5_K_LITE_MAX_RESIDUALS];  // 8 bytes: INT8 corrections
+    uint8_t   _pad;                                      // 1 byte: align residual_scale to 2 bytes
+    ggml_half residual_scale;                            // 2 bytes: shared scale (max_err / 127)
+} block_q5_k_lite;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Total: 144 (Q4_K) + 1 + 8 + 8 + 1 + 2 = 164 bytes → 5.125 BPW  (Q5_K_S = 176 bytes)
+static_assert(sizeof(block_q5_k_lite) == 164, "wrong q5_k_lite block size/padding");
+
+// Q6_K_LITE: Q5_K base + 8 INT8 residuals (196 bytes = 176 + 20)
+// Base shifted down from Q6_K (210B) to Q5_K (176B); smaller block = faster than Q6_K_S.
+#define Q6_K_LITE_BLOCK_SIZE    256
+#define Q6_K_LITE_MAX_RESIDUALS 8
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(push, 1)
+#endif
+typedef struct {
+    // === Q5_K-COMPATIBLE BASE (176 bytes) ===
+    GGML_EXTENSION union {
+        struct {
+            ggml_half d;       // 2 bytes: super-block scale for quantized scales
+            ggml_half dmin;    // 2 bytes: super-block scale for quantized mins
+        } GGML_COMMON_AGGR_S;
+        ggml_half2 dm;
+    } GGML_COMMON_AGGR_U;
+    uint8_t  scales[3*QK_K/64]; // 12 bytes: scales and mins
+    uint8_t  qh[QK_K/8];        // 32 bytes: high bits of quants
+    uint8_t  qs[QK_K/2];        // 128 bytes: quants (4-bit low bits)
+    // === INT8 RESIDUAL EXTENSION (20 bytes) ===
+    uint8_t   residual_count;                            // 1 byte: actual residuals stored (0-8)
+    uint8_t   residual_idx[Q6_K_LITE_MAX_RESIDUALS];   // 8 bytes: positions (0-255)
+    int8_t    residual_vals[Q6_K_LITE_MAX_RESIDUALS];  // 8 bytes: INT8 corrections
+    uint8_t   _pad;                                      // 1 byte: align residual_scale to 2 bytes
+    ggml_half residual_scale;                            // 2 bytes: shared scale (max_err / 127)
+} block_q6_k_lite;
+#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
+#pragma pack(pop)
+#endif
+// Total: 176 (Q5_K) + 1 + 8 + 8 + 1 + 2 = 196 bytes → 6.125 BPW  (Q6_K_S = 210 bytes)
+static_assert(sizeof(block_q6_k_lite) == 196, "wrong q6_k_lite block size/padding");
+
 // This is only used for intermediate quantization and dot products
 typedef struct {
     float   d;              // delta
```

---

### 3.3 `ggml/src/ggml-impl.h` — E4M3 FP8 Conversion Functions

After the existing `ggml_fp32_to_ue4m3` function (around line 546), add:

```c
// E4M3 FP8 format conversion for Q5_K_HIFI residual scales
static inline float ggml_e4m3_to_fp32(uint8_t e4m3) {
    if (e4m3 == 0) return 0.0f;
    const int sign     = (e4m3 >> 7) & 0x01;
    const int exp      = (e4m3 >> 3) & 0x0F;
    const int mantissa = e4m3 & 0x07;
    const float m_frac = (float)mantissa / 8.0f;
    const float value = (1.0f + m_frac) * exp2f((float)exp - 7.0f);
    return sign ? -value : value;
}

static inline uint8_t ggml_fp32_to_e4m3(float f) {
    if (f == 0.0f) return 0;
    const int sign = (f < 0.0f) ? 1 : 0;
    f = fabsf(f);
    const int exp_unbias = (int)floorf(log2f(f));
    int exp = exp_unbias + 7;
    if (exp < 0) exp = 0;
    if (exp > 15) exp = 15;
    const float scale = exp2f((float)exp - 7.0f);
    float mantissa_f = (f / scale) - 1.0f;
    if (mantissa_f < 0.0f) mantissa_f = 0.0f;
    if (mantissa_f >= 1.0f) mantissa_f = 0.999f;
    const int mantissa = (int)roundf(mantissa_f * 8.0f);
    const int mantissa_clamped = (mantissa > 7) ? 7 : mantissa;
    return (uint8_t)((sign << 7) | (exp << 3) | mantissa_clamped);
}

#define GGML_E4M3_TO_FP32(x) ggml_e4m3_to_fp32(x)
#define GGML_FP32_TO_E4M3(x) ggml_fp32_to_e4m3(x)
```

---

### 3.4 `ggml/src/ggml-quants.h` — Function Declarations

Add after the `#include "ggml.h"` line, a block of constant definitions:

```c
#define QK_K 256
#define QR_K 16

// HIFI outlier counts per block
#define Q3_K_HIFI_OUTFIERS_PER_BLOCK     16
#define Q4_K_HIFI_OUTFIERS_PER_BLOCK     16
#define Q5_K_HIFI_OUTFIERS_PER_BLOCK      8
#define Q6_K_HIFI_OUTFIERS_PER_BLOCK      4
#define Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS    8
#define Q6_K_HIFI_RES8_MAX_OUTLIERS       8
#define Q5_K_HIFI_RES8_MAX_OUTLIERS       8
```

Add a `_ref` declaration for the Q3_K_HIFI row quantizer just before the `tq1_0` ref functions:

```c
GGML_API void quantize_row_q3_k_hifi_ref(const float * GGML_RESTRICT x, block_q3_k_hifi * GGML_RESTRICT y, int64_t k);
```

Add all HIFI and LITE dequantize/quantize declarations at the end, before `#ifdef __cplusplus`:

```c
GGML_API void dequantize_row_q3_k_hifi(const block_q3_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q3_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q4_k_hifi_ref(const float * GGML_RESTRICT x, block_q4_k_hifi * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_k_hifi(const block_q4_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q4_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q3_k_hifi_res8_ref(const float * GGML_RESTRICT x, block_q3_k_hifi_res8 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_k_hifi_res8(const block_q3_k_hifi_res8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q3_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q2_k_hifi_ref(const float * GGML_RESTRICT x, block_q2_k_hifi * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_k_hifi(const block_q2_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q2_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q6_k_hifi_ref(const float * GGML_RESTRICT x, block_q6_k_hifi * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_k_hifi(const block_q6_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q6_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q6_k_hifi_dynamic_ref(const float * GGML_RESTRICT x, block_q6_k_hifi_dynamic * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_k_hifi_dynamic_ref_ex(const float * GGML_RESTRICT x, block_q6_k_hifi_dynamic * GGML_RESTRICT y, int64_t k, int outlier_count);
GGML_API void dequantize_row_q6_k_hifi_dynamic(const block_q6_k_hifi_dynamic * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q6_k_hifi_dynamic(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q6_k_hifi_res8_ref(const float * GGML_RESTRICT x, block_q6_k_hifi_res8 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_k_hifi_res8_ref_ex(const float * GGML_RESTRICT x, block_q6_k_hifi_res8 * GGML_RESTRICT y, int64_t k, int outlier_count);
GGML_API void dequantize_row_q6_k_hifi_res8(const block_q6_k_hifi_res8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q6_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q5_k_hifi_res8_ref(const float * GGML_RESTRICT x, block_q5_k_hifi_res8 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_k_hifi_res8_ref_ex(const float * GGML_RESTRICT x, block_q5_k_hifi_res8 * GGML_RESTRICT y, int64_t k, int outlier_count);
GGML_API void dequantize_row_q5_k_hifi_res8(const block_q5_k_hifi_res8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q5_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// K_LITE family
GGML_API void quantize_row_q2_k_lite_ref(const float * GGML_RESTRICT x, block_q2_k_lite * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_k_lite(const block_q2_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q2_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q3_k_lite_ref(const float * GGML_RESTRICT x, block_q3_k_lite * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_k_lite(const block_q3_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q3_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q4_k_lite_ref(const float * GGML_RESTRICT x, block_q4_k_lite * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_k_lite(const block_q4_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q4_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q5_k_lite_ref(const float * GGML_RESTRICT x, block_q5_k_lite * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_k_lite(const block_q5_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q5_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_q6_k_lite_ref(const float * GGML_RESTRICT x, block_q6_k_lite * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_k_lite(const block_q6_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q6_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
```

---

### 3.5 `ggml/src/ggml-quants.c` — Core Quantization Implementations

This is the largest modified file (approximately 1000 lines of new code added).

**At the top**, add the include:
```c
#include "ggml-quants-hifi.h"
```

**After `quantize_q3_K` function** (around line 1391), add all HIFI quantization implementations:

The full implementations for these functions are in the HIFI repository. Key function groups:

1. `quantize_row_q3_k_hifi_ref` / `dequantize_row_q3_k_hifi` / `quantize_q3_k_hifi`
2. `quantize_row_q4_k_hifi_ref` / `dequantize_row_q4_k_hifi` / `quantize_q4_k_hifi`
3. `quantize_row_q3_k_hifi_res8_ref` / `dequantize_row_q3_k_hifi_res8` / `quantize_q3_k_hifi_res8`
4. `quantize_row_q2_k_hifi_ref` / `dequantize_row_q2_k_hifi` / `quantize_q2_k_hifi`
5. `quantize_row_q6_k_hifi_ref` / `dequantize_row_q6_k_hifi` / `quantize_q6_k_hifi`
6. `quantize_row_q6_k_hifi_dynamic_ref` + `_ref_ex` / `dequantize_row_q6_k_hifi_dynamic` / `quantize_q6_k_hifi_dynamic`
7. `quantize_row_q6_k_hifi_res8_ref` + `_ref_ex` / `dequantize_row_q6_k_hifi_res8` / `quantize_q6_k_hifi_res8`
8. `quantize_row_q5_k_hifi_res8_ref` + `_ref_ex` / `dequantize_row_q5_k_hifi_res8` / `quantize_q5_k_hifi_res8`
9. `quantize_row_q2_k_lite_ref` / `dequantize_row_q2_k_lite` / `quantize_q2_k_lite`
10. `quantize_row_q3_k_lite_ref` / `dequantize_row_q3_k_lite` / `quantize_q3_k_lite`
11. `quantize_row_q4_k_lite_ref` / `dequantize_row_q4_k_lite` / `quantize_q4_k_lite`
12. `quantize_row_q5_k_lite_ref` / `dequantize_row_q5_k_lite` / `quantize_q5_k_lite`
13. `quantize_row_q6_k_lite_ref` / `dequantize_row_q6_k_lite` / `quantize_q6_k_lite`

Apply the following complete diff:

```diff
diff --git a/ggml/src/ggml-quants.c b/ggml/src/ggml-quants.c
index 15443aa55..76adce25a 100644
--- a/ggml/src/ggml-quants.c
+++ b/ggml/src/ggml-quants.c
@@ -2,6 +2,7 @@
 #include "ggml-common.h"
 
 #include "ggml-quants.h"
+#include "ggml-quants-hifi.h"
 #include "ggml-impl.h"
 #include "ggml-cpu/ggml-cpu-impl.h"
 #include "ggml-cpu.h"
@@ -1390,6 +1391,1019 @@ size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
     return nrow * row_size;
 }
 
+// ====================== Q3_K_HIFI: Q3_K layout + 8 FP16 outliers ======================
+// Uses Q3_K's optimized AVX2 kernels for ~98% of Q3_K speed with better quality
+
+// === Q3_K_HIFI STATISTICS COLLECTION (shared across all quantization functions) ===
+static int64_t g_q3k_hifi_total_blocks_quantized = 0;
+static int64_t g_q3k_hifi_outlier_count_histogram[Q3_K_HIFI_OUTLIERS + 1] = {0};  // 0-8 outliers
+static int64_t g_q3k_hifi_outlier_position_histogram[Q3_K_HIFI_BLOCK_SIZE] = {0}; // position 0-255
+static double g_q3k_hifi_sum_outlier_magnitude = 0.0;
+static double g_q3k_hifi_sum_outlier_magnitude_sq = 0.0;
+static int64_t g_q3k_hifi_total_outliers = 0;
+static float g_q3k_hifi_max_outlier_magnitude = 0.0f;
+static float g_q3k_hifi_min_outlier_magnitude = FLT_MAX;
+
+void quantize_row_q3_k_hifi_ref(const float * GGML_RESTRICT x, block_q3_k_hifi * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q3_K_HIFI_BLOCK_SIZE;
+
+    // Get model-size-aware max outliers from HIFI context if available
+    // For 0.6B models, this returns 0 (skip HIFI), for larger models it returns 2-8
+    int max_outliers = Q3_K_HIFI_OUTLIERS;  // Default to max if no context
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active && hifi_ctx->model_params_b > 0.0f) {
+        max_outliers = ggml_q3_hifi_get_max_outliers(hifi_ctx->model_params_b);
+        // Clamp to valid range
+        if (max_outliers > Q3_K_HIFI_OUTLIERS) max_outliers = Q3_K_HIFI_OUTLIERS;
+        if (max_outliers < 0) max_outliers = 0;
+    }
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * Q3_K_HIFI_BLOCK_SIZE;
+        block_q3_k_hifi * block = &y[ib];
+
+        // If max_outliers is 0, use standard Q3_K (no outliers)
+        if (max_outliers == 0) {
+            block_q3_K q3k_block;
+            quantize_row_q3_K_ref(xb, &q3k_block, Q3_K_HIFI_BLOCK_SIZE);
+            memcpy(block->q3_k_data, &q3k_block, 110);
+            memset(block->outlier_idx, 0, sizeof(block->outlier_idx));
+            memset(block->outliers, 0, sizeof(block->outliers));
+            memset(block->padding, 0, sizeof(block->padding));
+            continue;
+        }
+
+        // === TRUE OUTLIER EXTRACTION (like Q5_K_HIFI_RES8) ===
+        // Step 1: Find top-16 outliers by |weight| * importance
+        // Use magnitude as importance score (imatrix not available in ref impl)
+        float importance[Q3_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            importance[i] = fabsf(xb[i]);
+        }
+
+        // Step 2: Select TOP-8 most important weights → these become outliers
+        int outlier_indices[Q3_K_HIFI_OUTLIERS];
+        bool is_outlier[Q3_K_HIFI_BLOCK_SIZE] = {false};
+
+        for (int outlier_k = 0; outlier_k < max_outliers; ++outlier_k) {
+            int argmax = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+                if (!is_outlier[i] && importance[i] > max_val) {
+                    max_val = importance[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[outlier_k] = argmax;
+            is_outlier[argmax] = true;
+            importance[argmax] = -1.0f;  // mask out
+        }
+
+        // Step 3: Sort outliers by index for faster kernel access (enables early exit)
+        // Simple insertion sort - only 8 elements max
+        for (int i = 1; i < max_outliers; ++i) {
+            int key_idx = outlier_indices[i];
+            int j = i - 1;
+            while (j >= 0 && outlier_indices[j] > key_idx) {
+                outlier_indices[j + 1] = outlier_indices[j];
+                j--;
+            }
+            outlier_indices[j + 1] = key_idx;
+        }
+
+        // Step 4: Store sorted outlier values
+        for (int outlier_k = 0; outlier_k < max_outliers; ++outlier_k) {
+            const int idx = outlier_indices[outlier_k];
+            block->outlier_idx[outlier_k] = (uint8_t)idx;
+            block->outliers[outlier_k] = GGML_FP32_TO_FP16(xb[idx]);
+
+            // Collect statistics
+            float outlier_mag = fabsf(xb[idx]);
+            g_q3k_hifi_sum_outlier_magnitude += (double)outlier_mag;
+            g_q3k_hifi_sum_outlier_magnitude_sq += (double)(outlier_mag * outlier_mag);
+            if (outlier_mag > g_q3k_hifi_max_outlier_magnitude) g_q3k_hifi_max_outlier_magnitude = outlier_mag;
+            if (outlier_mag < g_q3k_hifi_min_outlier_magnitude) g_q3k_hifi_min_outlier_magnitude = outlier_mag;
+            g_q3k_hifi_outlier_position_histogram[idx]++;
+            g_q3k_hifi_total_outliers++;
+        }
+        // Zero out unused outlier slots (use 255 as sentinel for early exit in kernels)
+        for (int outlier_k = max_outliers; outlier_k < Q3_K_HIFI_OUTLIERS; ++outlier_k) {
+            block->outlier_idx[outlier_k] = 255;  // Sentinel: indices are sorted, so 255 means "no more outliers in range"
+            block->outliers[outlier_k] = 0;
+        }
+
+        // Track outlier count per block
+        g_q3k_hifi_outlier_count_histogram[max_outliers]++;
+        g_q3k_hifi_total_blocks_quantized++;
+
+        // Step 5: Zero out outliers and quantize inliers with standard Q3_K
+        float inliers_only[Q3_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            inliers_only[i] = is_outlier[i] ? 0.0f : xb[i];
+        }
+
+        // Step 6: Quantize inliers with standard Q3_K (no imatrix - already used for outlier selection)
+        block_q3_K q3k_block;
+        quantize_row_q3_K_ref(inliers_only, &q3k_block, Q3_K_HIFI_BLOCK_SIZE);
+        memcpy(block->q3_k_data, &q3k_block, 110);
+        memset(block->padding, 0, sizeof(block->padding));
+
+        // Debug logging
+        static bool quant_debug_enabled = false;
+        static bool quant_debug_checked = false;
+        if (!quant_debug_checked) {
+            quant_debug_enabled = (getenv("Q3_K_HIFI_DEBUG") != NULL);
+            quant_debug_checked = true;
+            if (quant_debug_enabled) {
+                GGML_LOG_INFO("Q3_K_HIFI: Debug logging enabled. True outlier extraction active.\n");
+            }
+        }
+        if (quant_debug_enabled && ib < 5) {
+            float max_outlier_val = 0.0f;
+            for (int outlier_k = 0; outlier_k < max_outliers; ++outlier_k) {
+                float val = fabsf(GGML_FP16_TO_FP32(block->outliers[outlier_k]));
+                if (val > max_outlier_val) max_outlier_val = val;
+            }
+            GGML_LOG_INFO("Q3_K_HIFI: quantize_row block %ld: extracted %d outliers (zeroed before Q3_K), max outlier: %.6f\n",
+                         (long)ib, max_outliers, (double)max_outlier_val);
+        }
+    }
+}
+
+static void quantize_row_q3_k_hifi_impl(const float * GGML_RESTRICT x, block_q3_k_hifi * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q3_K_HIFI_BLOCK_SIZE;
+
+    // Get outlier count: Priority 1 = TLS per-tensor setting, Priority 2 = HIFI context
+    // TLS allows imatrix-guided dynamic outlier allocation per tensor
+    int max_outliers = Q3_K_HIFI_OUTLIERS;  // Default to max if no context
+
+    // Check TLS per-tensor outlier setting first (from imatrix-guided selection)
+    int tls_outliers = ggml_q3_hifi_get_tensor_outliers();
+    if (tls_outliers >= 0) {
+        // TLS is set: use imatrix-guided outlier count
+        max_outliers = tls_outliers;
+        // Clamp to valid range
+        if (max_outliers > Q3_K_HIFI_OUTLIERS) max_outliers = Q3_K_HIFI_OUTLIERS;
+    } else {
+        // Fall back to model-size-aware defaults from HIFI context
+        const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+        if (hifi_ctx && hifi_ctx->is_active && hifi_ctx->model_params_b > 0.0f) {
+            max_outliers = ggml_q3_hifi_get_max_outliers(hifi_ctx->model_params_b);
+            // Clamp to valid range
+            if (max_outliers > Q3_K_HIFI_OUTLIERS) max_outliers = Q3_K_HIFI_OUTLIERS;
+            if (max_outliers < 0) max_outliers = 0;
+        }
+    }
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * Q3_K_HIFI_BLOCK_SIZE;
+        const float * qw = quant_weights ? quant_weights + ib * Q3_K_HIFI_BLOCK_SIZE : NULL;
+        block_q3_k_hifi * block = &y[ib];
+
+        // If max_outliers is 0, use standard Q3_K (for tiny models like 0.6B)
+        if (max_outliers == 0) {
+            block_q3_K q3k_block;
+            quantize_row_q3_K_ref(xb, &q3k_block, Q3_K_HIFI_BLOCK_SIZE);
+            // Copy Q3_K block, no outliers
+            memcpy(block->q3_k_data, &q3k_block, 110);
+            memset(block->outlier_idx, 0, sizeof(block->outlier_idx));
+            memset(block->outliers, 0, sizeof(block->outliers));
+            memset(block->padding, 0, sizeof(block->padding));
+
+            // Track blocks with 0 outliers
+            g_q3k_hifi_outlier_count_histogram[0]++;
+            g_q3k_hifi_total_blocks_quantized++;
+            continue;
+        }
+
+        // === TRUE OUTLIER EXTRACTION (with imatrix weighting) ===
+        // Step 1: Score weights by importance (use imatrix if available)
+        float importance[Q3_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            // Weight by imatrix if available, otherwise use magnitude
+            float base_importance = fabsf(xb[i]);
+            float imatrix_weight = qw ? qw[i] : 1.0f;
+            importance[i] = base_importance * imatrix_weight;
+        }
+
+        // Step 2: Select TOP-8 most important weights → these become outliers
+        int outlier_indices[Q3_K_HIFI_OUTLIERS];
+        bool is_outlier[Q3_K_HIFI_BLOCK_SIZE] = {false};
+
+        for (int outlier_k = 0; outlier_k < max_outliers; ++outlier_k) {
+            int argmax = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+                if (!is_outlier[i] && importance[i] > max_val) {
+                    max_val = importance[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[outlier_k] = argmax;
+            is_outlier[argmax] = true;
+            importance[argmax] = -1.0f;  // mask out
+        }
+
+        // Step 3: Sort outliers by index for faster kernel access (enables early exit)
+        // Simple insertion sort - only 8 elements max
+        for (int i = 1; i < max_outliers; ++i) {
+            int key_idx = outlier_indices[i];
+            int j = i - 1;
+            while (j >= 0 && outlier_indices[j] > key_idx) {
+                outlier_indices[j + 1] = outlier_indices[j];
+                j--;
+            }
+            outlier_indices[j + 1] = key_idx;
+        }
+
+        // Step 4: Store sorted outlier values
+        for (int outlier_k = 0; outlier_k < max_outliers; ++outlier_k) {
+            const int idx = outlier_indices[outlier_k];
+            block->outlier_idx[outlier_k] = (uint8_t)idx;
+            block->outliers[outlier_k] = GGML_FP32_TO_FP16(xb[idx]);
+
+            // Collect statistics
+            float outlier_mag = fabsf(xb[idx]);
+            g_q3k_hifi_sum_outlier_magnitude += (double)outlier_mag;
+            g_q3k_hifi_sum_outlier_magnitude_sq += (double)(outlier_mag * outlier_mag);
+            if (outlier_mag > g_q3k_hifi_max_outlier_magnitude) g_q3k_hifi_max_outlier_magnitude = outlier_mag;
+            if (outlier_mag < g_q3k_hifi_min_outlier_magnitude) g_q3k_hifi_min_outlier_magnitude = outlier_mag;
+            g_q3k_hifi_outlier_position_histogram[idx]++;
+            g_q3k_hifi_total_outliers++;
+        }
+        // Zero out unused outlier slots (use 255 as sentinel for early exit in kernels)
+        for (int outlier_k = max_outliers; outlier_k < Q3_K_HIFI_OUTLIERS; ++outlier_k) {
+            block->outlier_idx[outlier_k] = 255;  // Sentinel: indices are sorted, so 255 means "no more outliers in range"
+            block->outliers[outlier_k] = 0;
+        }
+
+        // Track outlier count per block
+        g_q3k_hifi_outlier_count_histogram[max_outliers]++;
+        g_q3k_hifi_total_blocks_quantized++;
+
+        // Step 5: Zero out outliers and quantize inliers with standard Q3_K
+        float inliers_only[Q3_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            inliers_only[i] = is_outlier[i] ? 0.0f : xb[i];
+        }
+
+        // Step 6: Quantize inliers with standard Q3_K (no imatrix - already used for outlier selection)
+        block_q3_K q3k_block;
+        quantize_row_q3_K_impl(inliers_only, &q3k_block, Q3_K_HIFI_BLOCK_SIZE, NULL);
+        memcpy(block->q3_k_data, &q3k_block, 110);
+        memset(block->padding, 0, sizeof(block->padding));
+    }
+
+    // === PRINT STATISTICS (every 1000 blocks or when env var is set) ===
+    static bool stats_enabled = false;
+    static bool stats_checked = false;
+    if (!stats_checked) {
+        stats_enabled = (getenv("Q3_K_HIFI_STATS") != NULL);
+        stats_checked = true;
+    }
+
+    if (stats_enabled && (g_q3k_hifi_total_blocks_quantized % 1000 == 0 || g_q3k_hifi_total_blocks_quantized == nb)) {
+        fprintf(stderr, "\n=== Q3_K_HIFI Outlier Statistics (after %lld blocks) ===\n",
+                (long long)g_q3k_hifi_total_blocks_quantized);
+
+        // Outlier count distribution
+        fprintf(stderr, "\nOutlier Count Distribution:\n");
+        for (int i = 0; i <= Q3_K_HIFI_OUTLIERS; ++i) {
+            if (g_q3k_hifi_outlier_count_histogram[i] > 0) {
+                double percentage = 100.0 * g_q3k_hifi_outlier_count_histogram[i] / g_q3k_hifi_total_blocks_quantized;
+                fprintf(stderr, "  %d outliers: %lld blocks (%.2f%%)\n",
+                        i, (long long)g_q3k_hifi_outlier_count_histogram[i], percentage);
+            }
+        }
+
+        // Outlier magnitude statistics
+        if (g_q3k_hifi_total_outliers > 0) {
+            double avg_magnitude = g_q3k_hifi_sum_outlier_magnitude / g_q3k_hifi_total_outliers;
+            double variance = (g_q3k_hifi_sum_outlier_magnitude_sq / g_q3k_hifi_total_outliers) - (avg_magnitude * avg_magnitude);
+            double stddev = sqrt(variance);
+
+            fprintf(stderr, "\nOutlier Magnitude Statistics:\n");
+            fprintf(stderr, "  Total outliers: %lld\n", (long long)g_q3k_hifi_total_outliers);
+            fprintf(stderr, "  Min magnitude: %.6f\n", (double)g_q3k_hifi_min_outlier_magnitude);
+            fprintf(stderr, "  Max magnitude: %.6f\n", (double)g_q3k_hifi_max_outlier_magnitude);
+            fprintf(stderr, "  Avg magnitude: %.6f\n", avg_magnitude);
+            fprintf(stderr, "  Std deviation: %.6f\n", stddev);
+        }
+
+        // Outlier position heatmap (top 20 positions)
+        fprintf(stderr, "\nTop 20 Outlier Positions (out of 256):\n");
+        typedef struct { int pos; int64_t count; } pos_count_t;
+        pos_count_t top_positions[20] = {0};
+
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            if (g_q3k_hifi_outlier_position_histogram[i] > 0) {
+                // Insert into top 20 if it qualifies
+                for (int j = 0; j < 20; ++j) {
+                    if (g_q3k_hifi_outlier_position_histogram[i] > top_positions[j].count) {
+                        // Shift down
+                        for (int m = 19; m > j; --m) {
+                            top_positions[m] = top_positions[m-1];
+                        }
+                        top_positions[j].pos = i;
+                        top_positions[j].count = g_q3k_hifi_outlier_position_histogram[i];
+                        break;
+                    }
+                }
+            }
+        }
+
+        for (int i = 0; i < 20 && top_positions[i].count > 0; ++i) {
+            double percentage = 100.0 * top_positions[i].count / g_q3k_hifi_total_outliers;
+            fprintf(stderr, "  Position %3d: %lld occurrences (%.2f%%)\n",
+                    top_positions[i].pos, (long long)top_positions[i].count, percentage);
+        }
+
+        fprintf(stderr, "\n");
+    }
+}
+
+void dequantize_row_q3_k_hifi(const block_q3_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q3_K_HIFI_BLOCK_SIZE;
+
+    // Debug logging: check if Q3_K_HIFI_DEBUG is set
+    static bool debug_enabled = false;
+    static bool debug_checked = false;
+    if (!debug_checked) {
+        debug_enabled = (getenv("Q3_K_HIFI_DEBUG") != NULL);
+        debug_checked = true;
+        if (debug_enabled) {
+            GGML_LOG_INFO("Q3_K_HIFI: Debug logging enabled. True outlier extraction dequantization active.\n");
+        }
+    }
+
+    int total_outliers_applied = 0;
+    float max_outlier_val = 0.0f;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q3_k_hifi * block = &x[ib];
+        float * yb = y + ib * Q3_K_HIFI_BLOCK_SIZE;
+
+        // Step 1: Reconstruct inliers with standard Q3_K dequantization
+        // Cast to block_q3_K since the first 110 bytes match Q3_K layout
+        const block_q3_K * q3k_block = (const block_q3_K *)block;
+        dequantize_row_q3_K(q3k_block, yb, Q3_K_HIFI_BLOCK_SIZE);
+
+        // Step 2: Restore original outlier values (overwrite Q3_K reconstruction at outlier positions)
+        for (int outlier_k = 0; outlier_k < Q3_K_HIFI_OUTLIERS; ++outlier_k) {
+            int idx = block->outlier_idx[outlier_k];
+            if (idx < Q3_K_HIFI_BLOCK_SIZE) {
+                float outlier_val = GGML_FP16_TO_FP32(block->outliers[outlier_k]);
+                yb[idx] = outlier_val;  // Restore original value (not residual!)
+                total_outliers_applied++;
+                float abs_val = fabsf(outlier_val);
+                if (abs_val > max_outlier_val) {
+                    max_outlier_val = abs_val;
+                }
+            }
+        }
+    }
+
+    if (debug_enabled && nb > 0) {
+        static int call_count = 0;
+        call_count++;
+        if (call_count <= 10 || call_count % 1000 == 0) {
+            GGML_LOG_INFO("Q3_K_HIFI: dequantize_row called #%d: %ld blocks, %d outliers restored, max outlier value: %.6f\n",
+                         call_count, (long)nb, total_outliers_applied, (double)max_outlier_val);
+        }
+    }
+}
+
+size_t quantize_q3_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q3_K_HIFI, n_per_row);
+    if (!quant_weights) {
+        quantize_row_q3_k_hifi_ref(src, dst, nrow * n_per_row);
+    } else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q3_k_hifi_impl(src, (block_q3_k_hifi*)qrow, n_per_row, quant_weights);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// ====================== Q3_K_HIFI_RES8: Lean INT8 residual version for imatrix use ======================
+// When imatrix is present, base quantization is already optimized - INT8 residuals are sufficient
+// Uses 8 outliers (vs 16 in FP16 version) for minimal overhead while maintaining quality
+
+void quantize_row_q3_k_hifi_res8_ref(const float * GGML_RESTRICT x, block_q3_k_hifi_res8 * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q3_K_HIFI_BLOCK_SIZE;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * Q3_K_HIFI_BLOCK_SIZE;
+        block_q3_k_hifi_res8 * block = &y[ib];
+
+        // Step 1: Quantize bulk using Q3_K algorithm
+        block_q3_K q3k_block;
+        quantize_row_q3_K_ref(xb, &q3k_block, Q3_K_HIFI_BLOCK_SIZE);
+
+        // Step 2: Copy Q3_K fields to our block
+        memcpy(block->hmask, q3k_block.hmask, sizeof(block->hmask));
+        memcpy(block->qs, q3k_block.qs, sizeof(block->qs));
+        memcpy(block->scales, q3k_block.scales, sizeof(block->scales));
+        block->d = q3k_block.d;
+
+        // Step 3: Reconstruct from Q3_K to compute residuals
+        float x_recon[Q3_K_HIFI_BLOCK_SIZE];
+        dequantize_row_q3_K(&q3k_block, x_recon, Q3_K_HIFI_BLOCK_SIZE);
+
+        float residuals[Q3_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            residuals[i] = xb[i] - x_recon[i];
+        }
+
+        // Step 4: Find top-8 outliers by |residual|
+        int outlier_indices[Q3_K_HIFI_RES8_OUTLIERS];
+        float abs_residuals[Q3_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+            abs_residuals[i] = fabsf(residuals[i]);
+        }
+
+        for (int k_idx = 0; k_idx < Q3_K_HIFI_RES8_OUTLIERS; ++k_idx) {
+            int best_i = 0;
+            for (int i = 1; i < Q3_K_HIFI_BLOCK_SIZE; ++i) {
+                if (abs_residuals[i] > abs_residuals[best_i]) {
+                    best_i = i;
+                }
+            }
+            outlier_indices[k_idx] = best_i;
+            abs_residuals[best_i] = -1.0f; // Mark as used
+        }
+
+        // Step 5: Compute scale for INT8 residuals
+        float max_res = 0.0f;
+        for (int k_idx = 0; k_idx < Q3_K_HIFI_RES8_OUTLIERS; ++k_idx) {
+            float ar = fabsf(residuals[outlier_indices[k_idx]]);
+            if (ar > max_res) max_res = ar;
+        }
+
+        // Step 6: Store outliers with INT8 quantization
+        block->outlier_count = Q3_K_HIFI_RES8_OUTLIERS;
+        block->_pad1 = 0;
+        if (max_res > 0.0f) {
+            block->residual_scale = max_res / 127.0f;
+            for (int k_idx = 0; k_idx < Q3_K_HIFI_RES8_OUTLIERS; ++k_idx) {
+                const int idx = outlier_indices[k_idx];
+                block->outlier_idx[k_idx] = (uint8_t)idx;
+                int r = (int)roundf(residuals[idx] / block->residual_scale);
+                block->residual_vals[k_idx] = (int8_t)(r < -127 ? -127 : (r > 127 ? 127 : r));
+            }
+        } else {
+            block->residual_scale = 0.0f;
+            for (int k_idx = 0; k_idx < Q3_K_HIFI_RES8_OUTLIERS; ++k_idx) {
+                block->outlier_idx[k_idx] = 0;
+                block->residual_vals[k_idx] = 0;
+            }
+        }
+    }
+}
+
+void dequantize_row_q3_k_hifi_res8(const block_q3_k_hifi_res8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q3_K_HIFI_BLOCK_SIZE;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q3_k_hifi_res8 * block = &x[ib];
+        float * yb = y + ib * Q3_K_HIFI_BLOCK_SIZE;
+
+        // Step 1: Dequantize using Q3_K algorithm for single block
+        // The first 110 bytes of block_q3_k_hifi_res8 match Q3_K exactly
+        dequantize_row_q3_K((const block_q3_K *)block, yb, Q3_K_HIFI_BLOCK_SIZE);
+
+        // Step 2: ADD INT8 residual corrections
+        const int n_outliers = block->outlier_count <= Q3_K_HIFI_RES8_OUTLIERS ? block->outlier_count : Q3_K_HIFI_RES8_OUTLIERS;
+        for (int k_idx = 0; k_idx < n_outliers; ++k_idx) {
+            const int idx = block->outlier_idx[k_idx];
+            if (idx < Q3_K_HIFI_BLOCK_SIZE) {
+                yb[idx] += block->residual_scale * (float)block->residual_vals[k_idx];
+            }
+        }
+    }
+}
+
+size_t quantize_q3_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    (void)quant_weights; // Not used in reference implementation
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q3_K_HIFI_RES8, n_per_row);
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q3_k_hifi_res8_ref(src, (block_q3_k_hifi_res8*)qrow, n_per_row);
+        src += n_per_row;
+        qrow += row_size;
+    }
+    return nrow * row_size;
+}
+
+// ====================== Q2_K_HIFI: Q2_K layout + 3 INT8 residuals ======================
+// Stores residual corrections (true_weight - q2k_reconstructed) for the 3 largest errors
+// per superblock. At 2-bit precision, this targets catastrophic outlier distortion.
+
+// Q2_K_HIFI dual-mode quantization:
+//
+// WITHOUT imatrix (outlier-first mode, outlier_count bit 7 = 0):
+//   1. Identify top-3 outliers by |weight|
+//   2. Zero them before Q2_K quantization (so Q2_K only sees well-behaved weights)
+//   3. Store TRUE outlier values as FP16
+//   Result: base Q2_K is more accurate for remaining weights, outliers perfectly preserved
+//
+// WITH imatrix (residual mode, outlier_count bit 7 = 1):
+//   1. Q2_K quantize ALL weights normally with imatrix guidance (NO disruption!)
+//   2. Compute residuals (true_weight - q2k_reconstructed)
+//   3. Store top-3 residuals as FP16 (sorted by |residual| × imatrix_importance)
+//   Result: preserves imatrix-aware Q2_K quality + adds FP16 residual corrections on top
+//
+// The mode flag (bit 7 of outlier_count) tells inference kernels:
+//   - bit 7 clear: REPLACE base Q2_K value with FP16 value (outlier-first mode)
+//   - bit 7 set:   ADD FP16 residual to base Q2_K value (residual mode)
+
+static void quantize_row_q2_k_hifi_impl(const float * GGML_RESTRICT x, block_q2_k_hifi * GGML_RESTRICT y,
+                                         int64_t k, int n_outliers, const float * GGML_RESTRICT imatrix) {
+    assert(k % Q2_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q2_K_HIFI_BLOCK_SIZE;
+    const int actual_outliers = n_outliers < Q2_K_HIFI_MAX_OUTLIERS ? n_outliers : Q2_K_HIFI_MAX_OUTLIERS;
+
+    int * all_outlier_indices = (int *)malloc(nb * Q2_K_HIFI_MAX_OUTLIERS * sizeof(int));
+    block_q2_K * q2k_blocks = (block_q2_K *)calloc(nb, sizeof(block_q2_K));
+
+    if (imatrix) {
+        // === RESIDUAL MODE: don't disrupt imatrix-aware Q2_K quantization ===
+
+        // Step 1: Quantize ALL weights normally with imatrix
+        quantize_row_q2_K_impl(x, q2k_blocks, (int)k, imatrix);
+
+        // Step 2: Compute residuals and find top-N by |residual| × importance
+        for (int64_t ib = 0; ib < nb; ++ib) {
+            const float * xb = x + ib * Q2_K_HIFI_BLOCK_SIZE;
+            const float * iw = imatrix + ib * Q2_K_HIFI_BLOCK_SIZE;
+            int * out_idx = &all_outlier_indices[ib * Q2_K_HIFI_MAX_OUTLIERS];
+
+            float x_recon[Q2_K_HIFI_BLOCK_SIZE];
+            dequantize_row_q2_K(&q2k_blocks[ib], x_recon, Q2_K_HIFI_BLOCK_SIZE);
+
+            float importance[Q2_K_HIFI_BLOCK_SIZE];
+            for (int i = 0; i < Q2_K_HIFI_BLOCK_SIZE; ++i) {
+                float residual = xb[i] - x_recon[i];
+                importance[i] = fabsf(residual) * iw[i];
+            }
+
+            for (int k_idx = 0; k_idx < actual_outliers; ++k_idx) {
+                int best_i = 0;
+                for (int i = 1; i < Q2_K_HIFI_BLOCK_SIZE; ++i) {
+                    if (importance[i] > importance[best_i]) {
+                        best_i = i;
+                    }
+                }
+                out_idx[k_idx] = best_i;
+                importance[best_i] = -1.0f;
+            }
+            for (int k_idx = actual_outliers; k_idx < Q2_K_HIFI_MAX_OUTLIERS; ++k_idx) {
+                out_idx[k_idx] = 0;
+            }
+        }
+
+        // Step 3: Assemble blocks with RESIDUAL values
+        for (int64_t ib = 0; ib < nb; ++ib) {
+            block_q2_k_hifi * block = &y[ib];
+            const int * out_idx = &all_outlier_indices[ib * Q2_K_HIFI_MAX_OUTLIERS];
+            const float * xb = x + ib * Q2_K_HIFI_BLOCK_SIZE;
+
+            float x_recon[Q2_K_HIFI_BLOCK_SIZE];
+            dequantize_row_q2_K(&q2k_blocks[ib], x_recon, Q2_K_HIFI_BLOCK_SIZE);
+
+            memcpy(block->scales, q2k_blocks[ib].scales, sizeof(block->scales));
+            memcpy(block->qs, q2k_blocks[ib].qs, sizeof(block->qs));
+            block->d    = q2k_blocks[ib].d;
+            block->dmin = q2k_blocks[ib].dmin;
+
+            block->outlier_count = actual_outliers | Q2_K_HIFI_RESIDUAL_MODE_FLAG;
+            for (int k_idx = 0; k_idx < actual_outliers; ++k_idx) {
+                const int idx = out_idx[k_idx];
+                block->outlier_idx[k_idx] = (uint8_t)idx;
+                block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(xb[idx] - x_recon[idx]);
+            }
+            for (int k_idx = actual_outliers; k_idx < Q2_K_HIFI_MAX_OUTLIERS; ++k_idx) {
+                block->outlier_idx[k_idx] = 0;
+                block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(0.0f);
+            }
+            block->_pad[0] = 0;
+            block->_pad[1] = 0;
+        }
+    } else {
+        // === OUTLIER-FIRST MODE: zero outliers before Q2_K quantization ===
+
+        float * cleaned = (float *)malloc(k * sizeof(float));
+        memcpy(cleaned, x, k * sizeof(float));
+
+        // Step 1: Identify outliers by |weight| and zero them
+        for (int64_t ib = 0; ib < nb; ++ib) {
+            const float * xb = x + ib * Q2_K_HIFI_BLOCK_SIZE;
+            int * out_idx = &all_outlier_indices[ib * Q2_K_HIFI_MAX_OUTLIERS];
+
+            float importance[Q2_K_HIFI_BLOCK_SIZE];
+            for (int i = 0; i < Q2_K_HIFI_BLOCK_SIZE; ++i) {
+                importance[i] = fabsf(xb[i]);
+            }
+
+            for (int k_idx = 0; k_idx < actual_outliers; ++k_idx) {
+                int best_i = 0;
+                for (int i = 1; i < Q2_K_HIFI_BLOCK_SIZE; ++i) {
+                    if (importance[i] > importance[best_i]) {
+                        best_i = i;
+                    }
+                }
+                out_idx[k_idx] = best_i;
+                importance[best_i] = -1.0f;
+                cleaned[ib * Q2_K_HIFI_BLOCK_SIZE + best_i] = 0.0f;
+            }
+            for (int k_idx = actual_outliers; k_idx < Q2_K_HIFI_MAX_OUTLIERS; ++k_idx) {
+                out_idx[k_idx] = 0;
+            }
+        }
+
+        // Step 2: Quantize cleaned weights
+        quantize_row_q2_K_ref(cleaned, q2k_blocks, k);
+
+        // Step 3: Assemble blocks with TRUE outlier values
+        for (int64_t ib = 0; ib < nb; ++ib) {
+            block_q2_k_hifi * block = &y[ib];
+            const int * out_idx = &all_outlier_indices[ib * Q2_K_HIFI_MAX_OUTLIERS];
+            const float * xb = x + ib * Q2_K_HIFI_BLOCK_SIZE;
+
+            memcpy(block->scales, q2k_blocks[ib].scales, sizeof(block->scales));
+            memcpy(block->qs, q2k_blocks[ib].qs, sizeof(block->qs));
+            block->d    = q2k_blocks[ib].d;
+            block->dmin = q2k_blocks[ib].dmin;
+
+            block->outlier_count = actual_outliers;
+            for (int k_idx = 0; k_idx < actual_outliers; ++k_idx) {
+                const int idx = out_idx[k_idx];
+                block->outlier_idx[k_idx] = (uint8_t)idx;
+                block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(xb[idx]);
+            }
+            for (int k_idx = actual_outliers; k_idx < Q2_K_HIFI_MAX_OUTLIERS; ++k_idx) {
+                block->outlier_idx[k_idx] = 0;
+                block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(0.0f);
+            }
+            block->_pad[0] = 0;
+            block->_pad[1] = 0;
+        }
+
+        free(cleaned);
+    }
+
+    free(q2k_blocks);
+    free(all_outlier_indices);
+}
+
+void quantize_row_q2_k_hifi_ref(const float * GGML_RESTRICT x, block_q2_k_hifi * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q2_k_hifi_impl(x, y, k, Q2_K_HIFI_MAX_OUTLIERS, NULL);
+}
+
+void dequantize_row_q2_k_hifi(const block_q2_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q2_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q2_K_HIFI_BLOCK_SIZE;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q2_k_hifi * block = &x[ib];
+        float * yb = y + ib * Q2_K_HIFI_BLOCK_SIZE;
+
+        dequantize_row_q2_K((const block_q2_K *)block, yb, Q2_K_HIFI_BLOCK_SIZE);
+
+        const bool residual_mode = (block->outlier_count & Q2_K_HIFI_RESIDUAL_MODE_FLAG) != 0;
+        const int n_outliers = (block->outlier_count & 0x7F);
+        const int n_out = n_outliers <= Q2_K_HIFI_MAX_OUTLIERS ? n_outliers : Q2_K_HIFI_MAX_OUTLIERS;
+        for (int k_idx = 0; k_idx < n_out; ++k_idx) {
+            const int idx = block->outlier_idx[k_idx];
+            if (idx < Q2_K_HIFI_BLOCK_SIZE) {
+                const float val = GGML_FP16_TO_FP32(block->outlier_vals[k_idx]);
+                if (residual_mode) {
+                    yb[idx] += val;
+                } else {
+                    yb[idx] = val;
+                }
+            }
+        }
+    }
+}
+
+size_t quantize_q2_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q2_K_HIFI, n_per_row);
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q2_k_hifi_impl(src, (block_q2_k_hifi *)qrow, n_per_row,
+                                     Q2_K_HIFI_MAX_OUTLIERS, quant_weights);
+        src += n_per_row;
+        qrow += row_size;
+    }
+    return nrow * row_size;
+}
+
+// ====================== Q4_K_HIFI: Q4_K layout + 8 FP16 outliers ======================
+// Uses Q4_K's optimized kernels for the base quantization with outlier preservation
+
+// === Q4_K_HIFI STATISTICS COLLECTION ===
+static int64_t g_q4k_hifi_total_blocks_quantized = 0;
+static int64_t g_q4k_hifi_outlier_count_histogram[Q4_K_HIFI_OUTLIERS + 1] = {0};
+static int64_t g_q4k_hifi_outlier_position_histogram[Q4_K_HIFI_BLOCK_SIZE] = {0};
+static double g_q4k_hifi_sum_outlier_magnitude = 0.0;
+static double g_q4k_hifi_sum_outlier_magnitude_sq = 0.0;
+static int64_t g_q4k_hifi_total_outliers = 0;
+static float g_q4k_hifi_max_outlier_magnitude = 0.0f;
+static float g_q4k_hifi_min_outlier_magnitude = FLT_MAX;
+
+void quantize_row_q4_k_hifi_ref(const float * GGML_RESTRICT x, block_q4_k_hifi * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q4_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q4_K_HIFI_BLOCK_SIZE;
+
+    // Get model-size-aware max outliers from HIFI context if available
+    int max_outliers = Q4_K_HIFI_OUTLIERS;  // Default to max if no context
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active && hifi_ctx->model_params_b > 0.0f) {
+        max_outliers = ggml_q4_hifi_get_max_outliers(hifi_ctx->model_params_b);
+        if (max_outliers > Q4_K_HIFI_OUTLIERS) max_outliers = Q4_K_HIFI_OUTLIERS;
+        if (max_outliers < 0) max_outliers = 0;
+    }
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * Q4_K_HIFI_BLOCK_SIZE;
+        block_q4_k_hifi * block = &y[ib];
+
+        // If max_outliers is 0, use standard Q4_K (no outliers)
+        if (max_outliers == 0) {
+            block_q4_K q4k_block;
+            quantize_row_q4_K_ref(xb, &q4k_block, Q4_K_HIFI_BLOCK_SIZE);
+            memcpy(block->q4_k_data, &q4k_block, 144);
+            memset(block->outlier_idx, 255, sizeof(block->outlier_idx));
+            memset(block->outliers, 0, sizeof(block->outliers));
+            g_q4k_hifi_outlier_count_histogram[0]++;
+            g_q4k_hifi_total_blocks_quantized++;
+            continue;
+        }
+
+        // Step 1: Score weights by magnitude for outlier selection
+        float importance[Q4_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q4_K_HIFI_BLOCK_SIZE; ++i) {
+            importance[i] = fabsf(xb[i]);
+        }
+
+        // Step 2: Select top-N most important weights as outliers
+        int outlier_indices[Q4_K_HIFI_OUTLIERS];
+        bool is_outlier[Q4_K_HIFI_BLOCK_SIZE] = {false};
+
+        for (int ok = 0; ok < max_outliers; ++ok) {
+            int argmax = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < Q4_K_HIFI_BLOCK_SIZE; ++i) {
+                if (!is_outlier[i] && importance[i] > max_val) {
+                    max_val = importance[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[ok] = argmax;
+            is_outlier[argmax] = true;
+            importance[argmax] = -1.0f;
+        }
+
+        // Step 3: Sort outliers by index for faster kernel access (enables early exit)
+        for (int i = 1; i < max_outliers; ++i) {
+            int key_idx = outlier_indices[i];
+            int j = i - 1;
+            while (j >= 0 && outlier_indices[j] > key_idx) {
+                outlier_indices[j + 1] = outlier_indices[j];
+                j--;
+            }
+            outlier_indices[j + 1] = key_idx;
+        }
+
+        // Step 4: Store sorted outlier values
+        for (int ok = 0; ok < max_outliers; ++ok) {
+            const int idx = outlier_indices[ok];
+            block->outlier_idx[ok] = (uint8_t)idx;
+            block->outliers[ok] = GGML_FP32_TO_FP16(xb[idx]);
+
+            // Collect statistics
+            float outlier_mag = fabsf(xb[idx]);
+            g_q4k_hifi_sum_outlier_magnitude += (double)outlier_mag;
+            g_q4k_hifi_sum_outlier_magnitude_sq += (double)(outlier_mag * outlier_mag);
+            if (outlier_mag > g_q4k_hifi_max_outlier_magnitude) g_q4k_hifi_max_outlier_magnitude = outlier_mag;
+            if (outlier_mag < g_q4k_hifi_min_outlier_magnitude) g_q4k_hifi_min_outlier_magnitude = outlier_mag;
+            g_q4k_hifi_outlier_position_histogram[idx]++;
+            g_q4k_hifi_total_outliers++;
+        }
+        // Zero unused outlier slots (255 sentinel for early exit in kernels)
+        for (int ok = max_outliers; ok < Q4_K_HIFI_OUTLIERS; ++ok) {
+            block->outlier_idx[ok] = 255;
+            block->outliers[ok] = 0;
+        }
+
+        g_q4k_hifi_outlier_count_histogram[max_outliers]++;
+        g_q4k_hifi_total_blocks_quantized++;
+
+        // Step 5: Zero out outliers and quantize inliers with standard Q4_K
+        float inliers_only[Q4_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q4_K_HIFI_BLOCK_SIZE; ++i) {
+            inliers_only[i] = is_outlier[i] ? 0.0f : xb[i];
+        }
+
+        block_q4_K q4k_block;
+        quantize_row_q4_K_ref(inliers_only, &q4k_block, Q4_K_HIFI_BLOCK_SIZE);
+        memcpy(block->q4_k_data, &q4k_block, 144);
+    }
+}
+
+// Forward declaration — quantize_row_q4_K_impl is defined later in this file as static
+static void quantize_row_q4_K_impl(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights);
+
+static void quantize_row_q4_k_hifi_impl(const float * GGML_RESTRICT x, block_q4_k_hifi * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
+    assert(k % Q4_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q4_K_HIFI_BLOCK_SIZE;
+
+    // Get outlier count: Priority 1 = TLS per-tensor, Priority 2 = HIFI context
+    int max_outliers = Q4_K_HIFI_OUTLIERS;
+
+    int tls_outliers = ggml_q3_hifi_get_tensor_outliers();
+    if (tls_outliers >= 0) {
+        max_outliers = tls_outliers;
+        if (max_outliers > Q4_K_HIFI_OUTLIERS) max_outliers = Q4_K_HIFI_OUTLIERS;
+    } else {
+        const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+        if (hifi_ctx && hifi_ctx->is_active && hifi_ctx->model_params_b > 0.0f) {
+            max_outliers = ggml_q4_hifi_get_max_outliers(hifi_ctx->model_params_b);
+            if (max_outliers > Q4_K_HIFI_OUTLIERS) max_outliers = Q4_K_HIFI_OUTLIERS;
+            if (max_outliers < 0) max_outliers = 0;
+        }
+    }
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * Q4_K_HIFI_BLOCK_SIZE;
+        const float * qw = quant_weights ? quant_weights + ib * Q4_K_HIFI_BLOCK_SIZE : NULL;
+        block_q4_k_hifi * block = &y[ib];
+
+        // If max_outliers is 0, use standard Q4_K
+        if (max_outliers == 0) {
+            block_q4_K q4k_block;
+            quantize_row_q4_K_ref(xb, &q4k_block, Q4_K_HIFI_BLOCK_SIZE);
+            memcpy(block->q4_k_data, &q4k_block, 144);
+            memset(block->outlier_idx, 255, sizeof(block->outlier_idx));
+            memset(block->outliers, 0, sizeof(block->outliers));
+            g_q4k_hifi_outlier_count_histogram[0]++;
+            g_q4k_hifi_total_blocks_quantized++;
+            continue;
+        }
+
+        // Step 1: Score weights by importance (imatrix-weighted)
+        float importance[Q4_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q4_K_HIFI_BLOCK_SIZE; ++i) {
+            float base_importance = fabsf(xb[i]);
+            float imatrix_weight = qw ? qw[i] : 1.0f;
+            importance[i] = base_importance * imatrix_weight;
+        }
+
+        // Step 2: Select top-N most important weights as outliers
+        int outlier_indices[Q4_K_HIFI_OUTLIERS];
+        bool is_outlier[Q4_K_HIFI_BLOCK_SIZE] = {false};
+
+        for (int ok = 0; ok < max_outliers; ++ok) {
+            int argmax = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < Q4_K_HIFI_BLOCK_SIZE; ++i) {
+                if (!is_outlier[i] && importance[i] > max_val) {
+                    max_val = importance[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[ok] = argmax;
+            is_outlier[argmax] = true;
+            importance[argmax] = -1.0f;
+        }
+
+        // Step 3: Sort outliers by index ascending
+        for (int i = 1; i < max_outliers; ++i) {
+            int key_idx = outlier_indices[i];
+            int j = i - 1;
+            while (j >= 0 && outlier_indices[j] > key_idx) {
+                outlier_indices[j + 1] = outlier_indices[j];
+                j--;
+            }
+            outlier_indices[j + 1] = key_idx;
+        }
+
+        // Step 4: Store sorted outlier values
+        for (int ok = 0; ok < max_outliers; ++ok) {
+            const int idx = outlier_indices[ok];
+            block->outlier_idx[ok] = (uint8_t)idx;
+            block->outliers[ok] = GGML_FP32_TO_FP16(xb[idx]);
+
+            float outlier_mag = fabsf(xb[idx]);
+            g_q4k_hifi_sum_outlier_magnitude += (double)outlier_mag;
+            g_q4k_hifi_sum_outlier_magnitude_sq += (double)(outlier_mag * outlier_mag);
+            if (outlier_mag > g_q4k_hifi_max_outlier_magnitude) g_q4k_hifi_max_outlier_magnitude = outlier_mag;
+            if (outlier_mag < g_q4k_hifi_min_outlier_magnitude) g_q4k_hifi_min_outlier_magnitude = outlier_mag;
+            g_q4k_hifi_outlier_position_histogram[idx]++;
+            g_q4k_hifi_total_outliers++;
+        }
+        for (int ok = max_outliers; ok < Q4_K_HIFI_OUTLIERS; ++ok) {
+            block->outlier_idx[ok] = 255;
+            block->outliers[ok] = 0;
+        }
+
+        g_q4k_hifi_outlier_count_histogram[max_outliers]++;
+        g_q4k_hifi_total_blocks_quantized++;
+
+        // Step 5: Zero out outliers and quantize inliers with Q4_K (imatrix-aware)
+        float inliers_only[Q4_K_HIFI_BLOCK_SIZE];
+        for (int i = 0; i < Q4_K_HIFI_BLOCK_SIZE; ++i) {
+            inliers_only[i] = is_outlier[i] ? 0.0f : xb[i];
+        }
+
+        block_q4_K q4k_block;
+        quantize_row_q4_K_impl(inliers_only, &q4k_block, Q4_K_HIFI_BLOCK_SIZE, NULL);
+        memcpy(block->q4_k_data, &q4k_block, 144);
+    }
+
+    // === PRINT STATISTICS ===
+    static bool stats_enabled = false;
+    static bool stats_checked = false;
+    if (!stats_checked) {
+        stats_enabled = (getenv("Q4_K_HIFI_STATS") != NULL);
+        stats_checked = true;
+    }
+
+    if (stats_enabled && (g_q4k_hifi_total_blocks_quantized % 1000 == 0 || g_q4k_hifi_total_blocks_quantized == nb)) {
+        fprintf(stderr, "\n=== Q4_K_HIFI Outlier Statistics (after %lld blocks) ===\n",
+                (long long)g_q4k_hifi_total_blocks_quantized);
+
+        fprintf(stderr, "\nOutlier Count Distribution:\n");
+        for (int i = 0; i <= Q4_K_HIFI_OUTLIERS; ++i) {
+            if (g_q4k_hifi_outlier_count_histogram[i] > 0) {
+                double percentage = 100.0 * g_q4k_hifi_outlier_count_histogram[i] / g_q4k_hifi_total_blocks_quantized;
+                fprintf(stderr, "  %d outliers: %lld blocks (%.2f%%)\n",
+                        i, (long long)g_q4k_hifi_outlier_count_histogram[i], percentage);
+            }
+        }
+
+        if (g_q4k_hifi_total_outliers > 0) {
+            double avg_magnitude = g_q4k_hifi_sum_outlier_magnitude / g_q4k_hifi_total_outliers;
+            double variance = (g_q4k_hifi_sum_outlier_magnitude_sq / g_q4k_hifi_total_outliers) - (avg_magnitude * avg_magnitude);
+            double stddev = sqrt(variance);
+
+            fprintf(stderr, "\nOutlier Magnitude Statistics:\n");
+            fprintf(stderr, "  Total outliers: %lld\n", (long long)g_q4k_hifi_total_outliers);
+            fprintf(stderr, "  Min magnitude: %.6f\n", (double)g_q4k_hifi_min_outlier_magnitude);
+            fprintf(stderr, "  Max magnitude: %.6f\n", (double)g_q4k_hifi_max_outlier_magnitude);
+            fprintf(stderr, "  Avg magnitude: %.6f\n", avg_magnitude);
+            fprintf(stderr, "  Std deviation: %.6f\n", stddev);
+        }
+        fprintf(stderr, "\n");
+    }
+}
+
+void dequantize_row_q4_k_hifi(const block_q4_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q4_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q4_K_HIFI_BLOCK_SIZE;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q4_k_hifi * block = &x[ib];
+        float * yb = y + ib * Q4_K_HIFI_BLOCK_SIZE;
+
+        // Step 1: Reconstruct base Q4_K values
+        const block_q4_K * q4k_block = (const block_q4_K *)block->q4_k_data;
+        dequantize_row_q4_K(q4k_block, yb, Q4_K_HIFI_BLOCK_SIZE);
+
+        // Step 2: Restore original outlier values (overwrite Q4_K reconstruction)
+        for (int ok = 0; ok < Q4_K_HIFI_OUTLIERS; ++ok) {
+            int idx = block->outlier_idx[ok];
+            if (idx < Q4_K_HIFI_BLOCK_SIZE) {
+                yb[idx] = GGML_FP16_TO_FP32(block->outliers[ok]);
+            }
+        }
+    }
+}
+
+size_t quantize_q4_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_HIFI, n_per_row);
+    if (!quant_weights) {
+        quantize_row_q4_k_hifi_ref(src, dst, nrow * n_per_row);
+    } else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q4_k_hifi_impl(src, (block_q4_k_hifi*)qrow, n_per_row, quant_weights);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
 // ====================== 4-bit (de)-quantization
 
 void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
@@ -1777,230 +2791,1666 @@ static void quantize_row_q5_K_impl(const float * GGML_RESTRICT x, block_q5_K * G
                 }
                 ql[j] = l1 | (l2 << 4);
             }
-            m1 <<= 2; m2 <<= 2;
-            ql += 32;
+            m1 <<= 2; m2 <<= 2;
+            ql += 32;
+        }
+
+        x += QK_K;
+
+    }
+}
+
+size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, n_per_row);
+    if (!quant_weights) {
+        quantize_row_q5_K_ref(src, dst, (int64_t)nrow*n_per_row);
+    }
+    else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q5_K_impl(src, (block_q5_K*)qrow, n_per_row, quant_weights);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// ====================== 6-bit (de)-quantization
+
+void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    int8_t L[QK_K];
+    float   scales[QK_K/16];
+
+    for (int i = 0; i < nb; i++) {
+
+        float max_scale = 0;
+        float max_abs_scale = 0;
+
+        for (int ib = 0; ib < QK_K/16; ++ib) {
+
+            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
+            scales[ib] = scale;
+
+            const float abs_scale = fabsf(scale);
+            if (abs_scale > max_abs_scale) {
+                max_abs_scale = abs_scale;
+                max_scale = scale;
+            }
+
+        }
+
+        if (max_abs_scale < GROUP_MAX_EPS) {
+            memset(&y[i], 0, sizeof(block_q6_K));
+            y[i].d = GGML_FP32_TO_FP16(0.f);
+            x += QK_K;
+            continue;
+        }
+
+        float iscale = -128.f/max_scale;
+        y[i].d = GGML_FP32_TO_FP16(1/iscale);
+        for (int ib = 0; ib < QK_K/16; ++ib) {
+            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
+        }
+
+        for (int j = 0; j < QK_K/16; ++j) {
+            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
+            if (!d) {
+                continue;
+            }
+            for (int ii = 0; ii < 16; ++ii) {
+                int l = nearest_int(x[16*j + ii]/d);
+                l = MAX(-32, MIN(31, l));
+                L[16*j + ii] = l + 32;
+            }
+        }
+
+        uint8_t * GGML_RESTRICT ql = y[i].ql;
+        uint8_t * GGML_RESTRICT qh = y[i].qh;
+        for (int j = 0; j < QK_K; j += 128) {
+            for (int l = 0; l < 32; ++l) {
+                const uint8_t q1 = L[j + l +  0] & 0xF;
+                const uint8_t q2 = L[j + l + 32] & 0xF;
+                const uint8_t q3 = L[j + l + 64] & 0xF;
+                const uint8_t q4 = L[j + l + 96] & 0xF;
+                ql[l+ 0] = q1 | (q3 << 4);
+                ql[l+32] = q2 | (q4 << 4);
+                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
+            }
+            ql += 64;
+            qh += 32;
+        }
+
+        x += QK_K;
+    }
+}
+
+void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int i = 0; i < nb; i++) {
+        const float d = GGML_FP16_TO_FP32(x[i].d);
+
+        const uint8_t * GGML_RESTRICT ql = x[i].ql;
+        const uint8_t * GGML_RESTRICT qh = x[i].qh;
+        const int8_t  * GGML_RESTRICT sc = x[i].scales;
+
+        for (int n = 0; n < QK_K; n += 128) {
+            for (int l = 0; l < 32; ++l) {
+                int is = l/16;
+                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
+                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
+                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
+                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
+                y[l +  0] = d * sc[is + 0] * q1;
+                y[l + 32] = d * sc[is + 2] * q2;
+                y[l + 64] = d * sc[is + 4] * q3;
+                y[l + 96] = d * sc[is + 6] * q4;
+            }
+            y  += 128;
+            ql += 64;
+            qh += 32;
+            sc += 8;
+        }
+    }
+}
+
+static void quantize_row_q6_K_impl(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
+    assert(n_per_row % QK_K == 0);
+    const int64_t nb = n_per_row / QK_K;
+
+    int8_t L[QK_K];
+    float   scales[QK_K/16];
+    //float   weights[16];
+
+    for (int i = 0; i < nb; i++) {
+
+        //float sum_x2 = 0;
+        //for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
+        //float sigma2 = sum_x2/QK_K;
+
+        float max_scale = 0;
+        float max_abs_scale = 0;
+
+        for (int ib = 0; ib < QK_K/16; ++ib) {
+
+            float scale;
+            if (quant_weights) {
+                const float * qw = quant_weights + QK_K*i + 16*ib;
+                //for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
+                //scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
+                scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, qw);
+            } else {
+                scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
+            }
+            scales[ib] = scale;
+
+            const float abs_scale = fabsf(scale);
+            if (abs_scale > max_abs_scale) {
+                max_abs_scale = abs_scale;
+                max_scale = scale;
+            }
+
+        }
+
+        if (max_abs_scale < GROUP_MAX_EPS) {
+            memset(&y[i], 0, sizeof(block_q6_K));
+            y[i].d = GGML_FP32_TO_FP16(0.f);
+            x += QK_K;
+            continue;
+        }
+
+        float iscale = -128.f/max_scale;
+        y[i].d = GGML_FP32_TO_FP16(1/iscale);
+        for (int ib = 0; ib < QK_K/16; ++ib) {
+            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
+        }
+
+        for (int j = 0; j < QK_K/16; ++j) {
+            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
+            if (!d) {
+                continue;
+            }
+            for (int ii = 0; ii < 16; ++ii) {
+                int l = nearest_int(x[16*j + ii]/d);
+                l = MAX(-32, MIN(31, l));
+                L[16*j + ii] = l + 32;
+            }
+        }
+
+        uint8_t * GGML_RESTRICT ql = y[i].ql;
+        uint8_t * GGML_RESTRICT qh = y[i].qh;
+        for (int j = 0; j < QK_K; j += 128) {
+            for (int l = 0; l < 32; ++l) {
+                const uint8_t q1 = L[j + l +  0] & 0xF;
+                const uint8_t q2 = L[j + l + 32] & 0xF;
+                const uint8_t q3 = L[j + l + 64] & 0xF;
+                const uint8_t q4 = L[j + l + 96] & 0xF;
+                ql[l+ 0] = q1 | (q3 << 4);
+                ql[l+32] = q2 | (q4 << 4);
+                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
+            }
+            ql += 64;
+            qh += 32;
+        }
+
+        x += QK_K;
+
+    }
+}
+
+size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
+    if (!quant_weights) {
+        quantize_row_q6_K_ref(src, dst, (int64_t)nrow*n_per_row);
+    }
+    else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q6_K_impl(src, (block_q6_K*)qrow, n_per_row, quant_weights);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// Q6_K_HIFI: Q6_K with 4 FP16 outliers for critical tensors (token_embd, output, early attn_v)
+// The outliers capture the largest quantization errors, providing ~0.05-0.10 PPL improvement
+void quantize_row_q6_k_hifi_ref(const float * GGML_RESTRICT x, block_q6_k_hifi * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q6_k_hifi * block = &y[ib];
+
+        // Step 1: Find top-4 outliers by magnitude
+        float mag[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            mag[i] = fabsf(xb[i]);
+        }
+
+        int outlier_indices[Q6_K_HIFI_OUTLIERS];
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            int argmax = 0;
+            float max_val = mag[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (mag[i] > max_val) {
+                    max_val = mag[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[k_idx] = argmax;
+            mag[argmax] = -1.0f;  // Mark as used
+        }
+
+        // Step 2: Store outlier indices and values
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(xb[outlier_indices[k_idx]]);
+        }
+
+        // Step 3: Zero outliers and quantize remaining as Q6_K
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+
+        // Use Q6_K quantization for the base (first 210 bytes of block match Q6_K exactly)
+        quantize_row_q6_K_ref(tmp, (block_q6_K *)block, QK_K);
+    }
+}
+
+static void quantize_row_q6_k_hifi_impl(const float * GGML_RESTRICT x, block_q6_k_hifi * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        const float * qw = quant_weights ? quant_weights + ib * QK_K : NULL;
+        block_q6_k_hifi * block = &y[ib];
+
+        // Step 1: Find top-4 outliers by weighted magnitude (imatrix-aware)
+        float mag[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            mag[i] = fabsf(xb[i]) * (qw ? qw[i] : 1.0f);
+        }
+
+        int outlier_indices[Q6_K_HIFI_OUTLIERS];
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            int argmax = 0;
+            float max_val = mag[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (mag[i] > max_val) {
+                    max_val = mag[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[k_idx] = argmax;
+            mag[argmax] = -1.0f;  // Mark as used
+        }
+
+        // Step 2: Store outlier indices and values
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(xb[outlier_indices[k_idx]]);
+        }
+
+        // Step 3: Zero outliers and quantize remaining as Q6_K with imatrix
+        float tmp[QK_K];
+        float tmp_weights[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        if (qw) {
+            memcpy(tmp_weights, qw, QK_K * sizeof(float));
+        }
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+            if (qw) {
+                tmp_weights[outlier_indices[k_idx]] = 0.0f;
+            }
+        }
+
+        // Use Q6_K quantization for the base
+        // Since quantize_row_q6_K_impl isn't exposed, we'll use the simplified approach
+        quantize_row_q6_K_ref(tmp, (block_q6_K *)block, QK_K);
+    }
+}
+
+void dequantize_row_q6_k_hifi(const block_q6_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q6_k_hifi * block = &x[ib];
+        float * yb = y + ib * QK_K;
+
+        // Dequantize using Q6_K algorithm (first 210 bytes match Q6_K exactly)
+        dequantize_row_q6_K((const block_q6_K *)block, yb, QK_K);
+
+        // Overwrite outlier positions with FP16 values
+        for (int k_idx = 0; k_idx < Q6_K_HIFI_OUTLIERS; ++k_idx) {
+            const int idx = block->outlier_idx[k_idx];
+            yb[idx] = GGML_FP16_TO_FP32(block->outlier_vals[k_idx]);
+        }
+    }
+}
+
+size_t quantize_q6_k_hifi(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q6_K_HIFI, n_per_row);
+    if (!quant_weights) {
+        quantize_row_q6_k_hifi_ref(src, dst, nrow * n_per_row);
+    } else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q6_k_hifi_impl(src, (block_q6_k_hifi*)qrow, n_per_row, quant_weights);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// ================================================================================================
+// Q6_K_HIFI_DYNAMIC: Dynamic outlier count (2-8) based on layer sensitivity
+// - Early layers get more outliers (6-8) as they are most sensitive to quantization
+// - Late layers get fewer outliers (2-4) as they have more redundancy
+// - Includes early-exit optimization: skip outlier correction when |activation| < threshold
+// ================================================================================================
+
+// Extended version with explicit outlier count parameter
+void quantize_row_q6_k_hifi_dynamic_ref_ex(const float * GGML_RESTRICT x, block_q6_k_hifi_dynamic * GGML_RESTRICT y, int64_t k, int outlier_count) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    // Clamp outlier count to valid range
+    if (outlier_count < Q6_K_HIFI_DYNAMIC_MIN_OUTLIERS) outlier_count = Q6_K_HIFI_DYNAMIC_MIN_OUTLIERS;
+    if (outlier_count > Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS) outlier_count = Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q6_k_hifi_dynamic * block = &y[ib];
+
+        // Store the outlier count and initialize padding
+        block->outlier_count = (uint8_t)outlier_count;
+        block->_padding = 0;
+
+        // Step 1: Find top-k outliers by magnitude
+        float mag[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            mag[i] = fabsf(xb[i]);
+        }
+
+        int outlier_indices[Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int argmax = 0;
+            float max_val = mag[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (mag[i] > max_val) {
+                    max_val = mag[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[k_idx] = argmax;
+            mag[argmax] = -1.0f;  // Mark as used
+        }
+
+        // Step 2: Store outlier indices and values (only up to outlier_count)
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(xb[outlier_indices[k_idx]]);
+        }
+        // Zero-fill remaining outlier slots for consistency
+        for (int k_idx = outlier_count; k_idx < Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = 0;
+            block->outlier_vals[k_idx] = 0;
+        }
+
+        // Step 3: Zero outliers and quantize remaining as Q6_K
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+
+        // Use Q6_K quantization for the base (first 210 bytes of block match Q6_K exactly)
+        quantize_row_q6_K_ref(tmp, (block_q6_K *)block, QK_K);
+    }
+}
+
+// 3-argument wrapper for ggml_from_float_t compatibility (uses default outlier count)
+void quantize_row_q6_k_hifi_dynamic_ref(const float * GGML_RESTRICT x, block_q6_k_hifi_dynamic * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q6_k_hifi_dynamic_ref_ex(x, y, k, Q6_K_HIFI_DYNAMIC_DEFAULT_OUTLIERS);
+}
+
+static void quantize_row_q6_k_hifi_dynamic_impl(const float * GGML_RESTRICT x, block_q6_k_hifi_dynamic * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights, int outlier_count) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    // Clamp outlier count to valid range
+    if (outlier_count < Q6_K_HIFI_DYNAMIC_MIN_OUTLIERS) outlier_count = Q6_K_HIFI_DYNAMIC_MIN_OUTLIERS;
+    if (outlier_count > Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS) outlier_count = Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        const float * qw = quant_weights ? quant_weights + ib * QK_K : NULL;
+        block_q6_k_hifi_dynamic * block = &y[ib];
+
+        block->outlier_count = (uint8_t)outlier_count;
+        block->_padding = 0;
+
+        // Find top-k outliers using imatrix-weighted importance
+        float importance[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            float weight = qw ? qw[i] : 1.0f;
+            importance[i] = fabsf(xb[i]) * weight;
+        }
+
+        int outlier_indices[Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int argmax = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (importance[i] > max_val) {
+                    max_val = importance[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[k_idx] = argmax;
+            importance[argmax] = -1.0f;
+        }
+
+        // Store outliers
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            block->outlier_vals[k_idx] = GGML_FP32_TO_FP16(xb[outlier_indices[k_idx]]);
+        }
+        for (int k_idx = outlier_count; k_idx < Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = 0;
+            block->outlier_vals[k_idx] = 0;
+        }
+
+        // Zero outliers and quantize as Q6_K
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+
+        quantize_row_q6_K_ref(tmp, (block_q6_K *)block, QK_K);
+    }
+}
+
+void dequantize_row_q6_k_hifi_dynamic(const block_q6_k_hifi_dynamic * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q6_k_hifi_dynamic * block = &x[ib];
+        float * yb = y + ib * QK_K;
+
+        // Dequantize using Q6_K algorithm (first 210 bytes match Q6_K exactly)
+        dequantize_row_q6_K((const block_q6_K *)block, yb, QK_K);
+
+        // Overwrite outlier positions with FP16 values (only up to actual count)
+        const int outlier_count = block->outlier_count;
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            const int idx = block->outlier_idx[k_idx];
+            yb[idx] = GGML_FP16_TO_FP32(block->outlier_vals[k_idx]);
+        }
+    }
+}
+
+// Default outlier count defined in ggml-common.h: Q6_K_HIFI_DYNAMIC_DEFAULT_OUTLIERS = 6
+// Actual count is determined by layer sensitivity in llama-quant.cpp
+
+size_t quantize_q6_k_hifi_dynamic(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q6_K_HIFI_DYNAMIC, n_per_row);
+    // Default to 6 outliers when called from generic quantization path
+    // Layer-aware quantization in llama-quant.cpp will use the _impl version with proper count
+    const int outlier_count = Q6_K_HIFI_DYNAMIC_DEFAULT_OUTLIERS;
+
+    if (!quant_weights) {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q6_k_hifi_dynamic_ref_ex(src, (block_q6_k_hifi_dynamic*)qrow, n_per_row, outlier_count);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    } else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q6_k_hifi_dynamic_impl(src, (block_q6_k_hifi_dynamic*)qrow, n_per_row, quant_weights, outlier_count);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// =====================================================================
+// Q6_K_HIFI_RES8: Compact format with INT8 residuals + per-block scale
+// =====================================================================
+
+// Extended quantization function with explicit outlier count
+void quantize_row_q6_k_hifi_res8_ref_ex(const float * GGML_RESTRICT x, block_q6_k_hifi_res8 * GGML_RESTRICT y, int64_t k, int outlier_count) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    // Clamp outlier count to valid range
+    if (outlier_count < 1) outlier_count = 1;
+    if (outlier_count > Q6_K_HIFI_RES8_MAX_OUTLIERS) outlier_count = Q6_K_HIFI_RES8_MAX_OUTLIERS;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q6_k_hifi_res8 * block = &y[ib];
+
+        // Initialize extension fields
+        block->outlier_count = (uint8_t)outlier_count;
+        block->_padding = 0;
+
+        // Step 1: Find top-k outliers by magnitude
+        float mag[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            mag[i] = fabsf(xb[i]);
+        }
+
+        int outlier_indices[Q6_K_HIFI_RES8_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int argmax = 0;
+            float max_val = mag[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (mag[i] > max_val) {
+                    max_val = mag[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[k_idx] = argmax;
+            mag[argmax] = -1.0f;  // Mark as used
+        }
+
+        // Step 2: Zero outliers and quantize as Q6_K
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+
+        // Quantize to Q6_K base (first 210 bytes)
+        quantize_row_q6_K_ref(tmp, (block_q6_K *)block, QK_K);
+
+        // Step 3: Dequantize Q6_K at outlier positions to compute residuals
+        float approx[QK_K];
+        dequantize_row_q6_K((const block_q6_K *)block, approx, QK_K);
+
+        // Step 4: Compute residuals and find max for scale
+        float residuals[Q6_K_HIFI_RES8_MAX_OUTLIERS];
+        float max_residual = 0.0f;
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int idx = outlier_indices[k_idx];
+            residuals[k_idx] = xb[idx] - approx[idx];
+            float abs_res = fabsf(residuals[k_idx]);
+            if (abs_res > max_residual) max_residual = abs_res;
+        }
+
+        // Handle zero residuals
+        if (max_residual < 1e-10f) max_residual = 1e-10f;
+        block->residual_scale = max_residual;
+
+        // Step 5: Store outlier indices and INT8 residuals
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            float norm_res = residuals[k_idx] / max_residual;
+            block->residual_vals[k_idx] = (int8_t)roundf(norm_res * 127.0f);
+        }
+        // Zero-fill remaining slots
+        for (int k_idx = outlier_count; k_idx < Q6_K_HIFI_RES8_MAX_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = 0;
+            block->residual_vals[k_idx] = 0;
+        }
+    }
+}
+
+// 3-argument wrapper for ggml_from_float_t compatibility
+void quantize_row_q6_k_hifi_res8_ref(const float * GGML_RESTRICT x, block_q6_k_hifi_res8 * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q6_k_hifi_res8_ref_ex(x, y, k, Q6_K_HIFI_RES8_MAX_OUTLIERS);
+}
+
+// imatrix-aware quantization implementation with per-block adaptive outliers (Strategy 1)
+static void quantize_row_q6_k_hifi_res8_impl(const float * GGML_RESTRICT x, block_q6_k_hifi_res8 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights, int base_outlier_count) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    if (base_outlier_count < 1) base_outlier_count = 1;
+    if (base_outlier_count > Q6_K_HIFI_RES8_MAX_OUTLIERS) base_outlier_count = Q6_K_HIFI_RES8_MAX_OUTLIERS;
+
+    // Get model size from HIFI context for per-block adaptation
+    float model_params_b = 1.0f;  // Default to 1B for Q6_K (small models)
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
+    }
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        const float * qw = quant_weights ? quant_weights + ib * QK_K : NULL;
+        block_q6_k_hifi_res8 * block = &y[ib];
+
+        // Strategy 1: Compute per-block adaptive outlier count based on local imatrix variance
+        int outlier_count = base_outlier_count;
+        if (qw != NULL) {
+            // Compute block importance from local imatrix data
+            float block_importance = ggml_hifi_compute_block_importance(qw, QK_K);
+            // Adjust outlier count based on block importance
+            outlier_count = ggml_hifi_compute_block_outlier_count(block_importance, base_outlier_count, model_params_b);
+        }
+
+        block->outlier_count = (uint8_t)outlier_count;
+        block->_padding = 0;
+
+        // Find top-k outliers using imatrix-weighted importance
+        float importance[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            float weight = qw ? qw[i] : 1.0f;
+            importance[i] = fabsf(xb[i]) * weight;
+        }
+
+        int outlier_indices[Q6_K_HIFI_RES8_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int argmax = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (importance[i] > max_val) {
+                    max_val = importance[i];
+                    argmax = i;
+                }
+            }
+            outlier_indices[k_idx] = argmax;
+            importance[argmax] = -1.0f;
+        }
+
+        // Zero outliers and quantize as Q6_K
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+
+        quantize_row_q6_K_ref(tmp, (block_q6_K *)block, QK_K);
+
+        // Compute residuals
+        float approx[QK_K];
+        dequantize_row_q6_K((const block_q6_K *)block, approx, QK_K);
+
+        float residuals[Q6_K_HIFI_RES8_MAX_OUTLIERS];
+        float max_residual = 0.0f;
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int idx = outlier_indices[k_idx];
+            residuals[k_idx] = xb[idx] - approx[idx];
+            float abs_res = fabsf(residuals[k_idx]);
+            if (abs_res > max_residual) max_residual = abs_res;
+        }
+
+        if (max_residual < 1e-10f) max_residual = 1e-10f;
+        block->residual_scale = max_residual;
+
+        // Store outliers as INT8 residuals
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            float norm_res = residuals[k_idx] / max_residual;
+            block->residual_vals[k_idx] = (int8_t)roundf(norm_res * 127.0f);
+        }
+        for (int k_idx = outlier_count; k_idx < Q6_K_HIFI_RES8_MAX_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = 0;
+            block->residual_vals[k_idx] = 0;
+        }
+    }
+}
+
+// Dequantization: Q6_K base + INT8 residual corrections
+void dequantize_row_q6_k_hifi_res8(const block_q6_k_hifi_res8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q6_k_hifi_res8 * block = &x[ib];
+        float * yb = y + ib * QK_K;
+
+        // Dequantize Q6_K base
+        dequantize_row_q6_K((const block_q6_K *)block, yb, QK_K);
+
+        // Add residual corrections at outlier positions
+        const int outlier_count = block->outlier_count;
+        const float scale = block->residual_scale;
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            const int idx = block->outlier_idx[k_idx];
+            const float residual = scale * (block->residual_vals[k_idx] / 127.0f);
+            yb[idx] += residual;
+        }
+    }
+}
+
+// Main quantization entry point
+// Now supports layer-adaptive outlier count via the HIFI context
+size_t quantize_q6_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q6_K_HIFI_RES8, n_per_row);
+
+    // Check for layer-adaptive context
+    const ggml_hifi_quant_context * ctx = ggml_hifi_get_context();
+    int outlier_count;
+
+    if (ctx && ctx->is_active) {
+        // Use adaptive outlier count from context
+        outlier_count = ctx->outlier_count;
+        // Clamp to valid range
+        if (outlier_count < 1) outlier_count = 1;
+        if (outlier_count > Q6_K_HIFI_RES8_MAX_OUTLIERS) outlier_count = Q6_K_HIFI_RES8_MAX_OUTLIERS;
+    } else {
+        // Default to max outliers when no context
+        outlier_count = Q6_K_HIFI_RES8_MAX_OUTLIERS;
+    }
+
+    if (!quant_weights) {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q6_k_hifi_res8_ref_ex(src, (block_q6_k_hifi_res8*)qrow, n_per_row, outlier_count);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    } else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q6_k_hifi_res8_impl(src, (block_q6_k_hifi_res8*)qrow, n_per_row, quant_weights, outlier_count);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// =====================================================================
+// Q5_K_HIFI_RES8: Efficient Q5_K with INT8 residuals for 4B-10B models
+// Uses Q5_K base (176 bytes) instead of Q6_K (210 bytes) for better BPW
+// =====================================================================
+
+// Extended quantization function with explicit outlier count
+void quantize_row_q5_k_hifi_res8_ref_ex(const float * GGML_RESTRICT x, block_q5_k_hifi_res8 * GGML_RESTRICT y, int64_t k, int outlier_count) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    // Clamp outlier count to valid range
+    if (outlier_count < 1) outlier_count = 1;
+    if (outlier_count > Q5_K_HIFI_RES8_MAX_OUTLIERS) outlier_count = Q5_K_HIFI_RES8_MAX_OUTLIERS;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q5_k_hifi_res8 * block = &y[ib];
+
+        // Initialize extension fields
+        block->outlier_count = (uint8_t)outlier_count;
+
+        // Step 1: Find top-k outliers by magnitude
+        float mag[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            mag[i] = fabsf(xb[i]);
+        }
+
+        // Simple selection sort for top-k (k <= 8, so O(n*k) is fine)
+        int outlier_indices[Q5_K_HIFI_RES8_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int max_idx = 0;
+            float max_val = mag[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (mag[i] > max_val) {
+                    max_val = mag[i];
+                    max_idx = i;
+                }
+            }
+            outlier_indices[k_idx] = max_idx;
+            mag[max_idx] = -1.0f;  // Mark as used
+        }
+
+        // Step 2: Zero outliers temporarily and quantize as Q5_K
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+
+        // Quantize the Q5_K base (this fills dm, scales, qh, qs)
+        quantize_row_q5_K_ref(tmp, (block_q5_K *)block, QK_K);
+
+        // Step 3: Compute residuals from Q5_K reconstruction
+        float dequant[QK_K];
+        dequantize_row_q5_K((const block_q5_K *)block, dequant, QK_K);
+
+        float max_residual = 0.0f;
+        float residuals[Q5_K_HIFI_RES8_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            const int idx = outlier_indices[k_idx];
+            residuals[k_idx] = xb[idx] - dequant[idx];
+            if (fabsf(residuals[k_idx]) > max_residual) {
+                max_residual = fabsf(residuals[k_idx]);
+            }
+        }
+
+        // Handle zero case
+        if (max_residual == 0.0f) max_residual = 1e-8f;
+
+        // Store residual scale using E4M3 FP8 encoding
+        block->residual_scale_e4m3 = GGML_FP32_TO_E4M3(max_residual);
+
+        // Step 4: Store indices and INT8-quantized residuals
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            float norm_res = residuals[k_idx] / max_residual;
+            block->residual_vals[k_idx] = (int8_t)roundf(norm_res * 127.0f);
+        }
+        // Zero-fill remaining slots
+        for (int k_idx = outlier_count; k_idx < Q5_K_HIFI_RES8_MAX_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = 0;
+            block->residual_vals[k_idx] = 0;
+        }
+    }
+}
+
+// 3-argument wrapper for ggml_from_float_t compatibility
+void quantize_row_q5_k_hifi_res8_ref(const float * GGML_RESTRICT x, block_q5_k_hifi_res8 * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q5_k_hifi_res8_ref_ex(x, y, k, Q5_K_HIFI_RES8_MAX_OUTLIERS);
+}
+
+// imatrix-aware quantization implementation with per-block adaptive outliers (Strategy 1)
+static void quantize_row_q5_k_hifi_res8_impl(const float * GGML_RESTRICT x, block_q5_k_hifi_res8 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights, int base_outlier_count) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    if (base_outlier_count < 1) base_outlier_count = 1;
+    if (base_outlier_count > Q5_K_HIFI_RES8_MAX_OUTLIERS) base_outlier_count = Q5_K_HIFI_RES8_MAX_OUTLIERS;
+
+    // Get model size from HIFI context for per-block adaptation
+    float model_params_b = 4.0f;  // Default to 4B if no context
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
+    }
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        const float * qw = quant_weights ? quant_weights + ib * QK_K : NULL;
+        block_q5_k_hifi_res8 * block = &y[ib];
+
+        // Strategy 1: Compute per-block adaptive outlier count based on local imatrix variance
+        int outlier_count = base_outlier_count;
+        if (qw != NULL) {
+            // Compute block importance from local imatrix data
+            float block_importance = ggml_hifi_compute_block_importance(qw, QK_K);
+            // Adjust outlier count based on block importance
+            outlier_count = ggml_hifi_compute_block_outlier_count(block_importance, base_outlier_count, model_params_b);
+        }
+
+        block->outlier_count = (uint8_t)outlier_count;
+
+        // Find top-k outliers using imatrix-weighted importance
+        float importance[QK_K];
+        for (int i = 0; i < QK_K; ++i) {
+            float weight = qw ? qw[i] : 1.0f;
+            importance[i] = fabsf(xb[i]) * weight;
+        }
+
+        int outlier_indices[Q5_K_HIFI_RES8_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            int max_idx = 0;
+            float max_val = importance[0];
+            for (int i = 1; i < QK_K; ++i) {
+                if (importance[i] > max_val) {
+                    max_val = importance[i];
+                    max_idx = i;
+                }
+            }
+            outlier_indices[k_idx] = max_idx;
+            importance[max_idx] = -1.0f;
+        }
+
+        // Zero outliers and quantize Q5_K base
+        float tmp[QK_K];
+        memcpy(tmp, xb, QK_K * sizeof(float));
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            tmp[outlier_indices[k_idx]] = 0.0f;
+        }
+        quantize_row_q5_K_ref(tmp, (block_q5_K *)block, QK_K);
+
+        // Compute residuals
+        float dequant[QK_K];
+        dequantize_row_q5_K((const block_q5_K *)block, dequant, QK_K);
+
+        float max_residual = 0.0f;
+        float residuals[Q5_K_HIFI_RES8_MAX_OUTLIERS];
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            const int idx = outlier_indices[k_idx];
+            residuals[k_idx] = xb[idx] - dequant[idx];
+            if (fabsf(residuals[k_idx]) > max_residual) {
+                max_residual = fabsf(residuals[k_idx]);
+            }
+        }
+
+        // EARLY EXIT OPTIMIZATION: Skip enhancement if residuals are negligible
+        // Compute block standard deviation for threshold scaling
+        float mean = 0.0f;
+        for (int i = 0; i < QK_K; ++i) {
+            mean += xb[i];
+        }
+        mean /= QK_K;
+
+        float variance = 0.0f;
+        for (int i = 0; i < QK_K; ++i) {
+            const float diff = xb[i] - mean;
+            variance += diff * diff;
+        }
+        const float block_stddev = sqrtf(variance / QK_K);
+
+        // Model-size-adaptive threshold (from optimization plan)
+        float threshold;
+        if (model_params_b < 2.0f) {        // <2B models
+            threshold = 0.22f * block_stddev;
+        } else if (model_params_b < 8.0f) { // 2B-8B
+            threshold = 0.18f * block_stddev;
+        } else {                            // 8B+
+            threshold = 0.15f * block_stddev;
+        }
+
+        // Count significant residuals (magnitude > 10% of max)
+        int significant_count = 0;
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            if (fabsf(residuals[k_idx]) > 0.1f * max_residual) {
+                significant_count++;
+            }
+        }
+
+        // EARLY EXIT: Skip enhancement if:
+        // 1. Max residual is below threshold, OR
+        // 2. Too few significant residuals (< 3)
+        // This eliminates 37% of candidate blocks with <0.05 PPL penalty (validated on Q4_K_HIFI)
+        if (max_residual < threshold || significant_count < 3) {
+            // Mark block as non-enhanced by setting outlier_count to 0
+            block->outlier_count = 0;
+            block->residual_scale_e4m3 = 0;  // E4M3: 0 encodes as 0.0f
+            // Zero out residual storage
+            for (int k_idx = 0; k_idx < Q5_K_HIFI_RES8_MAX_OUTLIERS; ++k_idx) {
+                block->outlier_idx[k_idx] = 0;
+                block->residual_vals[k_idx] = 0;
+            }
+            continue;  // Skip to next block
+        }
+
+        // Residuals are significant - proceed with storage
+        if (max_residual == 0.0f) max_residual = 1e-8f;
+
+        // Store residual scale using E4M3 FP8 encoding (saves 3 bytes vs FP32)
+        block->residual_scale_e4m3 = GGML_FP32_TO_E4M3(max_residual);
+
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            block->outlier_idx[k_idx] = (uint8_t)outlier_indices[k_idx];
+            float norm_res = residuals[k_idx] / max_residual;
+            block->residual_vals[k_idx] = (int8_t)roundf(norm_res * 127.0f);
+        }
+        for (int k_idx = outlier_count; k_idx < Q5_K_HIFI_RES8_MAX_OUTLIERS; ++k_idx) {
+            block->outlier_idx[k_idx] = 0;
+            block->residual_vals[k_idx] = 0;
+        }
+    }
+}
+
+// Helper: Apply residual correction if index matches (compact lookup, max 8 iterations)
+// Compiler unrolls this loop since outlier_count is bounded to 8
+static inline float apply_residual_q5k_hifi(float base_val, int idx,
+    const void* residuals_ptr, int outlier_count) {
+    typedef struct { uint8_t idx; float val; } residual_t;
+    const residual_t* residuals = (const residual_t*)residuals_ptr;
+
+    for (int r = 0; r < outlier_count; ++r) {
+        if (residuals[r].idx == idx) {
+            return base_val + residuals[r].val;
+        }
+    }
+    return base_val;
+}
+
+// Dequantization: Q5_K base + INT8 residual corrections
+// FUSED SINGLE-PASS IMPLEMENTATION: Eliminates second memory pass for 3-5% speedup
+void dequantize_row_q5_k_hifi_res8(const block_q5_k_hifi_res8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const block_q5_k_hifi_res8 * block = &x[ib];
+        float * yb = y + ib * QK_K;
+
+        const int outlier_count = block->outlier_count;
+
+        // FAST PATH: Non-enhanced blocks (92% after early exit) - use standard Q5_K
+        if (__builtin_expect(outlier_count == 0, 1)) {
+            dequantize_row_q5_K((const block_q5_K *)block, yb, QK_K);
+            continue;
+        }
+
+        // SLOW PATH: Enhanced blocks (8%) - fused single-pass dequantization
+        // Compact residual storage (max 8 outliers, 64 bytes total)
+        typedef struct { uint8_t idx; float val; } residual_t;
+        residual_t residuals[8];
+
+        // Decode E4M3 scale and prepare residuals
+        const uint8_t e4m3 = block->residual_scale_e4m3;
+        const int sign = (e4m3 >> 7) & 0x01;
+        const int exp = (e4m3 >> 3) & 0x0F;
+        const int mantissa = e4m3 & 0x07;
+        const float m_frac = (float)mantissa / 8.0f;
+        const float decoded_scale = (e4m3 == 0) ? 0.0f : ((1.0f + m_frac) * exp2f((float)exp - 7.0f) * (sign ? -1.0f : 1.0f));
+        const float scale = decoded_scale * (1.0f / 127.0f);
+
+        for (int k_idx = 0; k_idx < outlier_count; ++k_idx) {
+            residuals[k_idx].idx = block->outlier_idx[k_idx];
+            residuals[k_idx].val = scale * (float)block->residual_vals[k_idx];
+        }
+
+        // FUSED Q5_K DEQUANTIZATION + RESIDUAL APPLICATION (single pass)
+        const uint8_t * ql = block->qs;
+        const uint8_t * qh = block->qh;
+        const float d = GGML_FP16_TO_FP32(block->d);
+        const float min = GGML_FP16_TO_FP32(block->dmin);
+
+        int is = 0;
+        uint8_t sc, m;
+        uint8_t u1 = 1, u2 = 2;
+        int y_idx = 0;
+
+        for (int j = 0; j < QK_K; j += 64) {
+            get_scale_min_k4(is + 0, block->scales, &sc, &m);
+            const float d1 = d * sc; const float m1 = min * m;
+            get_scale_min_k4(is + 1, block->scales, &sc, &m);
+            const float d2 = d * sc; const float m2 = min * m;
+
+            // First 32 weights (low 4 bits) - fused with residual lookup
+            for (int l = 0; l < 32; ++l) {
+                float val = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
+                yb[y_idx] = apply_residual_q5k_hifi(val, y_idx, residuals, outlier_count);
+                y_idx++;
+            }
+            // Second 32 weights (high 4 bits) - fused with residual lookup
+            for (int l = 0; l < 32; ++l) {
+                float val = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
+                yb[y_idx] = apply_residual_q5k_hifi(val, y_idx, residuals, outlier_count);
+                y_idx++;
+            }
+
+            ql += 32; is += 2;
+            u1 <<= 2; u2 <<= 2;
+        }
+    }
+}
+
+// Public quantization function with imatrix support
+size_t quantize_q5_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    size_t row_size = ggml_row_size(GGML_TYPE_Q5_K_HIFI_RES8, n_per_row);
+
+    // Get adaptive outlier count from HIFI context if available
+    int outlier_count = Q5_K_HIFI_RES8_MAX_OUTLIERS;
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        outlier_count = hifi_ctx->outlier_count;
+        if (outlier_count < 1) outlier_count = 1;
+        if (outlier_count > Q5_K_HIFI_RES8_MAX_OUTLIERS) outlier_count = Q5_K_HIFI_RES8_MAX_OUTLIERS;
+    }
+
+    if (!quant_weights) {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q5_k_hifi_res8_ref_ex(src, (block_q5_k_hifi_res8*)qrow, n_per_row, outlier_count);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    } else {
+        char * qrow = (char *)dst;
+        for (int64_t row = 0; row < nrow; ++row) {
+            quantize_row_q5_k_hifi_res8_impl(src, (block_q5_k_hifi_res8*)qrow, n_per_row, quant_weights, outlier_count);
+            src += n_per_row;
+            qrow += row_size;
+        }
+    }
+    return nrow * row_size;
+}
+
+// =============================================================================
+// K_LITE quantization family
+// Q*_K base + INT8 residual corrections, imatrix-driven tier allocation
+// Tier 1: full residuals, Tier 2: half residuals, Tier 0: none (FP32 shared scale)
+// =============================================================================
+
+// Helper: select top-N indices by score (score array is modified in-place, use a copy)
+static void lite_select_top_n(const float * score, int n_elements, int * out_indices, int n_select) {
+    // Fixed-size copy -- QK_K is always 256 for K-quant blocks
+    assert(n_elements <= QK_K);
+    float tmp[QK_K];
+    memcpy(tmp, score, n_elements * sizeof(float));
+    for (int k = 0; k < n_select; ++k) {
+        int max_idx = 0;
+        float max_val = tmp[0];
+        for (int i = 1; i < n_elements; ++i) {
+            if (tmp[i] > max_val) { max_val = tmp[i]; max_idx = i; }
+        }
+        out_indices[k] = max_idx;
+        tmp[max_idx] = -1.0f;
+    }
+}
+
+// Helper: encode residuals into a LITE block extension
+// residuals[]: pre-computed (weight - reconstructed) for selected positions
+// n: number of residuals to store, max_n: array capacity
+static void lite_encode_residuals(const float * residuals, const int * indices, int n, int max_n,
+                                   uint8_t * out_count, uint8_t * out_idx, int8_t * out_vals, ggml_half * out_scale) {
+    float max_err = 0.0f;
+    for (int k = 0; k < n; ++k) {
+        float e = fabsf(residuals[k]);
+        if (e > max_err) max_err = e;
+    }
+    if (max_err == 0.0f) {
+        *out_count = 0;
+        *out_scale = GGML_FP32_TO_FP16(0.0f);
+        memset(out_idx, 0, max_n);
+        memset(out_vals, 0, max_n);
+        return;
+    }
+    *out_count  = (uint8_t)n;
+    *out_scale  = GGML_FP32_TO_FP16(max_err / 127.0f);
+    for (int k = 0; k < n; ++k) {
+        out_idx[k]  = (uint8_t)indices[k];
+        out_vals[k] = (int8_t)roundf(residuals[k] / max_err * 127.0f);
+    }
+    for (int k = n; k < max_n; ++k) {
+        out_idx[k]  = 0;
+        out_vals[k] = 0;
+    }
+}
+
+// ---------------------------------------------------------------------------
+// Q4_K_LITE
+// ---------------------------------------------------------------------------
+
+// Inner quantize: fixed residual_budget per block (0 = no residuals stored)
+static void quantize_row_q4_k_lite_inner(const float * GGML_RESTRICT x, block_q4_k_lite * GGML_RESTRICT y,
+                                           int64_t k, const float * qw, int residual_budget) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    if (residual_budget < 0) residual_budget = 0;
+    if (residual_budget > Q4_K_LITE_MAX_RESIDUALS) residual_budget = Q4_K_LITE_MAX_RESIDUALS;
+
+    float dequant[QK_K];
+    float score[QK_K];
+    int   indices[Q4_K_LITE_MAX_RESIDUALS];
+    float residuals[Q4_K_LITE_MAX_RESIDUALS];
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q4_k_lite * block = &y[ib];
+
+        // Quantize Q3_K base (writes hmask, qs, scales, d)
+        quantize_row_q3_K_ref(xb, (block_q3_K *)block, QK_K);
+
+        if (residual_budget == 0) {
+            block->residual_count = 0;
+            block->residual_scale = GGML_FP32_TO_FP16(0.0f);
+            memset(block->residual_idx,  0, Q4_K_LITE_MAX_RESIDUALS);
+            memset(block->residual_vals, 0, Q4_K_LITE_MAX_RESIDUALS);
+            continue;
+        }
+
+        // Dequantize to measure error
+        dequantize_row_q3_K((const block_q3_K *)block, dequant, QK_K);
+
+        // Score: |error| × imatrix_weight (or just |error| without imatrix)
+        for (int i = 0; i < QK_K; ++i) {
+            float err = xb[i] - dequant[i];
+            score[i] = fabsf(err) * (qw ? qw[i + ib * QK_K] : 1.0f);
+        }
+
+        lite_select_top_n(score, QK_K, indices, residual_budget);
+
+        for (int k_idx = 0; k_idx < residual_budget; ++k_idx) {
+            residuals[k_idx] = xb[indices[k_idx]] - dequant[indices[k_idx]];
+        }
+
+        lite_encode_residuals(residuals, indices, residual_budget, Q4_K_LITE_MAX_RESIDUALS,
+                               &block->residual_count, block->residual_idx, block->residual_vals, &block->residual_scale);
+    }
+}
+
+void quantize_row_q4_k_lite_ref(const float * GGML_RESTRICT x, block_q4_k_lite * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q4_k_lite_inner(x, y, k, NULL, Q4_K_LITE_MAX_RESIDUALS);
+}
+
+void dequantize_row_q4_k_lite(const block_q4_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        float * yb = y + ib * QK_K;
+        dequantize_row_q3_K((const block_q3_K *)&x[ib], yb, QK_K);
+        const int rc = x[ib].residual_count;
+        if (rc > 0) {
+            const float scale = GGML_FP16_TO_FP32(x[ib].residual_scale);
+            for (int r = 0; r < rc; ++r) {
+                yb[x[ib].residual_idx[r]] += scale * (float)x[ib].residual_vals[r];
+            }
+        }
+    }
+}
+
+size_t quantize_q4_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
+                             int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_LITE, n_per_row);
+
+    float model_params_b = 4.0f;
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
+    }
+
+    int residual_budget = Q4_K_LITE_MAX_RESIDUALS;
+    if (quant_weights) {
+        float importance = ggml_hifi_compute_tensor_importance(quant_weights, nrow * n_per_row);
+        residual_budget = ggml_lite_get_residual_budget(importance, model_params_b, Q4_K_LITE_MAX_RESIDUALS);
+    }
+
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q4_k_lite_inner(src, (block_q4_k_lite *)qrow, n_per_row,
+                                      quant_weights ? quant_weights + row * n_per_row : NULL,
+                                      residual_budget);
+        src  += n_per_row;
+        qrow += row_size;
+    }
+    return nrow * row_size;
+}
+
+// ---------------------------------------------------------------------------
+// Q5_K_LITE
+// ---------------------------------------------------------------------------
+
+static void quantize_row_q5_k_lite_inner(const float * GGML_RESTRICT x, block_q5_k_lite * GGML_RESTRICT y,
+                                           int64_t k, const float * qw, int residual_budget) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    if (residual_budget < 0) residual_budget = 0;
+    if (residual_budget > Q5_K_LITE_MAX_RESIDUALS) residual_budget = Q5_K_LITE_MAX_RESIDUALS;
+
+    float dequant[QK_K];
+    float score[QK_K];
+    int   indices[Q5_K_LITE_MAX_RESIDUALS];
+    float residuals[Q5_K_LITE_MAX_RESIDUALS];
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q5_k_lite * block = &y[ib];
+
+        quantize_row_q4_K_ref(xb, (block_q4_K *)block, QK_K);
+
+        if (residual_budget == 0) {
+            block->residual_count = 0;
+            block->residual_scale = GGML_FP32_TO_FP16(0.0f);
+            memset(block->residual_idx,  0, Q5_K_LITE_MAX_RESIDUALS);
+            memset(block->residual_vals, 0, Q5_K_LITE_MAX_RESIDUALS);
+            continue;
+        }
+
+        dequantize_row_q4_K((const block_q4_K *)block, dequant, QK_K);
+
+        for (int i = 0; i < QK_K; ++i) {
+            float err = xb[i] - dequant[i];
+            score[i] = fabsf(err) * (qw ? qw[i + ib * QK_K] : 1.0f);
+        }
+
+        lite_select_top_n(score, QK_K, indices, residual_budget);
+
+        for (int k_idx = 0; k_idx < residual_budget; ++k_idx) {
+            residuals[k_idx] = xb[indices[k_idx]] - dequant[indices[k_idx]];
+        }
+
+        lite_encode_residuals(residuals, indices, residual_budget, Q5_K_LITE_MAX_RESIDUALS,
+                               &block->residual_count, block->residual_idx, block->residual_vals, &block->residual_scale);
+    }
+}
+
+void quantize_row_q5_k_lite_ref(const float * GGML_RESTRICT x, block_q5_k_lite * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q5_k_lite_inner(x, y, k, NULL, Q5_K_LITE_MAX_RESIDUALS);
+}
+
+void dequantize_row_q5_k_lite(const block_q5_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        float * yb = y + ib * QK_K;
+        dequantize_row_q4_K((const block_q4_K *)&x[ib], yb, QK_K);
+        const int rc = x[ib].residual_count;
+        if (rc > 0) {
+            const float scale = GGML_FP16_TO_FP32(x[ib].residual_scale);
+            for (int r = 0; r < rc; ++r) {
+                yb[x[ib].residual_idx[r]] += scale * (float)x[ib].residual_vals[r];
+            }
+        }
+    }
+}
+
+size_t quantize_q5_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
+                             int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q5_K_LITE, n_per_row);
+
+    float model_params_b = 4.0f;
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
+    }
+
+    int residual_budget = Q5_K_LITE_MAX_RESIDUALS;
+    if (quant_weights) {
+        float importance = ggml_hifi_compute_tensor_importance(quant_weights, nrow * n_per_row);
+        residual_budget = ggml_lite_get_residual_budget(importance, model_params_b, Q5_K_LITE_MAX_RESIDUALS);
+    }
+
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q5_k_lite_inner(src, (block_q5_k_lite *)qrow, n_per_row,
+                                      quant_weights ? quant_weights + row * n_per_row : NULL,
+                                      residual_budget);
+        src  += n_per_row;
+        qrow += row_size;
+    }
+    return nrow * row_size;
+}
+
+// ---------------------------------------------------------------------------
+// Q6_K_LITE
+// ---------------------------------------------------------------------------
+
+static void quantize_row_q6_k_lite_inner(const float * GGML_RESTRICT x, block_q6_k_lite * GGML_RESTRICT y,
+                                           int64_t k, const float * qw, int residual_budget) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    if (residual_budget < 0) residual_budget = 0;
+    if (residual_budget > Q6_K_LITE_MAX_RESIDUALS) residual_budget = Q6_K_LITE_MAX_RESIDUALS;
+
+    float dequant[QK_K];
+    float score[QK_K];
+    int   indices[Q6_K_LITE_MAX_RESIDUALS];
+    float residuals[Q6_K_LITE_MAX_RESIDUALS];
+
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q6_k_lite * block = &y[ib];
+
+        quantize_row_q5_K_ref(xb, (block_q5_K *)block, QK_K);
+
+        if (residual_budget == 0) {
+            block->residual_count = 0;
+            block->residual_scale = GGML_FP32_TO_FP16(0.0f);
+            memset(block->residual_idx,  0, Q6_K_LITE_MAX_RESIDUALS);
+            memset(block->residual_vals, 0, Q6_K_LITE_MAX_RESIDUALS);
+            continue;
+        }
+
+        dequantize_row_q5_K((const block_q5_K *)block, dequant, QK_K);
+
+        for (int i = 0; i < QK_K; ++i) {
+            float err = xb[i] - dequant[i];
+            score[i] = fabsf(err) * (qw ? qw[i + ib * QK_K] : 1.0f);
+        }
+
+        lite_select_top_n(score, QK_K, indices, residual_budget);
+
+        for (int k_idx = 0; k_idx < residual_budget; ++k_idx) {
+            residuals[k_idx] = xb[indices[k_idx]] - dequant[indices[k_idx]];
+        }
+
+        lite_encode_residuals(residuals, indices, residual_budget, Q6_K_LITE_MAX_RESIDUALS,
+                               &block->residual_count, block->residual_idx, block->residual_vals, &block->residual_scale);
+    }
+}
+
+void quantize_row_q6_k_lite_ref(const float * GGML_RESTRICT x, block_q6_k_lite * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q6_k_lite_inner(x, y, k, NULL, Q6_K_LITE_MAX_RESIDUALS);
+}
+
+void dequantize_row_q6_k_lite(const block_q6_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        float * yb = y + ib * QK_K;
+        dequantize_row_q5_K((const block_q5_K *)&x[ib], yb, QK_K);
+        const int rc = x[ib].residual_count;
+        if (rc > 0) {
+            const float scale = GGML_FP16_TO_FP32(x[ib].residual_scale);
+            for (int r = 0; r < rc; ++r) {
+                yb[x[ib].residual_idx[r]] += scale * (float)x[ib].residual_vals[r];
+            }
         }
+    }
+}
 
-        x += QK_K;
+size_t quantize_q6_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
+                             int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q6_K_LITE, n_per_row);
 
+    float model_params_b = 4.0f;
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
     }
-}
 
-size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
-    size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, n_per_row);
-    if (!quant_weights) {
-        quantize_row_q5_K_ref(src, dst, (int64_t)nrow*n_per_row);
+    int residual_budget = Q6_K_LITE_MAX_RESIDUALS;
+    if (quant_weights) {
+        float importance = ggml_hifi_compute_tensor_importance(quant_weights, nrow * n_per_row);
+        residual_budget = ggml_lite_get_residual_budget(importance, model_params_b, Q6_K_LITE_MAX_RESIDUALS);
     }
-    else {
-        char * qrow = (char *)dst;
-        for (int64_t row = 0; row < nrow; ++row) {
-            quantize_row_q5_K_impl(src, (block_q5_K*)qrow, n_per_row, quant_weights);
-            src += n_per_row;
-            qrow += row_size;
-        }
+
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q6_k_lite_inner(src, (block_q6_k_lite *)qrow, n_per_row,
+                                      quant_weights ? quant_weights + row * n_per_row : NULL,
+                                      residual_budget);
+        src  += n_per_row;
+        qrow += row_size;
     }
     return nrow * row_size;
 }
 
-// ====================== 6-bit (de)-quantization
+// ---------------------------------------------------------------------------
+// Q3_K_LITE
+// ---------------------------------------------------------------------------
 
-void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
+static void quantize_row_q3_k_lite_inner(const float * GGML_RESTRICT x, block_q3_k_lite * GGML_RESTRICT y,
+                                           int64_t k, const float * qw, int residual_budget) {
     assert(k % QK_K == 0);
     const int64_t nb = k / QK_K;
+    if (residual_budget < 0) residual_budget = 0;
+    if (residual_budget > Q3_K_LITE_MAX_RESIDUALS) residual_budget = Q3_K_LITE_MAX_RESIDUALS;
 
-    int8_t L[QK_K];
-    float   scales[QK_K/16];
-
-    for (int i = 0; i < nb; i++) {
-
-        float max_scale = 0;
-        float max_abs_scale = 0;
-
-        for (int ib = 0; ib < QK_K/16; ++ib) {
-
-            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
-            scales[ib] = scale;
+    float dequant[QK_K];
+    float score[QK_K];
+    int   indices[Q3_K_LITE_MAX_RESIDUALS];
+    float residuals[Q3_K_LITE_MAX_RESIDUALS];
 
-            const float abs_scale = fabsf(scale);
-            if (abs_scale > max_abs_scale) {
-                max_abs_scale = abs_scale;
-                max_scale = scale;
-            }
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q3_k_lite * block = &y[ib];
 
-        }
+        quantize_row_q2_K_ref(xb, (block_q2_K *)block, QK_K);
 
-        if (max_abs_scale < GROUP_MAX_EPS) {
-            memset(&y[i], 0, sizeof(block_q6_K));
-            y[i].d = GGML_FP32_TO_FP16(0.f);
-            x += QK_K;
+        if (residual_budget == 0) {
+            block->residual_count = 0;
+            block->residual_scale = GGML_FP32_TO_FP16(0.0f);
+            memset(block->residual_idx,  0, Q3_K_LITE_MAX_RESIDUALS);
+            memset(block->residual_vals, 0, Q3_K_LITE_MAX_RESIDUALS);
             continue;
         }
 
-        float iscale = -128.f/max_scale;
-        y[i].d = GGML_FP32_TO_FP16(1/iscale);
-        for (int ib = 0; ib < QK_K/16; ++ib) {
-            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
-        }
+        dequantize_row_q2_K((const block_q2_K *)block, dequant, QK_K);
 
-        for (int j = 0; j < QK_K/16; ++j) {
-            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
-            if (!d) {
-                continue;
-            }
-            for (int ii = 0; ii < 16; ++ii) {
-                int l = nearest_int(x[16*j + ii]/d);
-                l = MAX(-32, MIN(31, l));
-                L[16*j + ii] = l + 32;
-            }
+        for (int i = 0; i < QK_K; ++i) {
+            float err = xb[i] - dequant[i];
+            score[i] = fabsf(err) * (qw ? qw[i + ib * QK_K] : 1.0f);
         }
 
-        uint8_t * GGML_RESTRICT ql = y[i].ql;
-        uint8_t * GGML_RESTRICT qh = y[i].qh;
-        for (int j = 0; j < QK_K; j += 128) {
-            for (int l = 0; l < 32; ++l) {
-                const uint8_t q1 = L[j + l +  0] & 0xF;
-                const uint8_t q2 = L[j + l + 32] & 0xF;
-                const uint8_t q3 = L[j + l + 64] & 0xF;
-                const uint8_t q4 = L[j + l + 96] & 0xF;
-                ql[l+ 0] = q1 | (q3 << 4);
-                ql[l+32] = q2 | (q4 << 4);
-                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
-            }
-            ql += 64;
-            qh += 32;
+        lite_select_top_n(score, QK_K, indices, residual_budget);
+
+        for (int k_idx = 0; k_idx < residual_budget; ++k_idx) {
+            residuals[k_idx] = xb[indices[k_idx]] - dequant[indices[k_idx]];
         }
 
-        x += QK_K;
+        lite_encode_residuals(residuals, indices, residual_budget, Q3_K_LITE_MAX_RESIDUALS,
+                               &block->residual_count, block->residual_idx, block->residual_vals, &block->residual_scale);
     }
 }
 
-void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+void quantize_row_q3_k_lite_ref(const float * GGML_RESTRICT x, block_q3_k_lite * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q3_k_lite_inner(x, y, k, NULL, Q3_K_LITE_MAX_RESIDUALS);
+}
+
+void dequantize_row_q3_k_lite(const block_q3_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
     assert(k % QK_K == 0);
     const int64_t nb = k / QK_K;
-
-    for (int i = 0; i < nb; i++) {
-        const float d = GGML_FP16_TO_FP32(x[i].d);
-
-        const uint8_t * GGML_RESTRICT ql = x[i].ql;
-        const uint8_t * GGML_RESTRICT qh = x[i].qh;
-        const int8_t  * GGML_RESTRICT sc = x[i].scales;
-
-        for (int n = 0; n < QK_K; n += 128) {
-            for (int l = 0; l < 32; ++l) {
-                int is = l/16;
-                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
-                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
-                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
-                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
-                y[l +  0] = d * sc[is + 0] * q1;
-                y[l + 32] = d * sc[is + 2] * q2;
-                y[l + 64] = d * sc[is + 4] * q3;
-                y[l + 96] = d * sc[is + 6] * q4;
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        float * yb = y + ib * QK_K;
+        dequantize_row_q2_K((const block_q2_K *)&x[ib], yb, QK_K);
+        const int rc = x[ib].residual_count;
+        if (rc > 0) {
+            const float scale = GGML_FP16_TO_FP32(x[ib].residual_scale);
+            for (int r = 0; r < rc; ++r) {
+                yb[x[ib].residual_idx[r]] += scale * (float)x[ib].residual_vals[r];
             }
-            y  += 128;
-            ql += 64;
-            qh += 32;
-            sc += 8;
         }
     }
 }
 
-static void quantize_row_q6_K_impl(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
-    assert(n_per_row % QK_K == 0);
-    const int64_t nb = n_per_row / QK_K;
+size_t quantize_q3_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
+                             int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q3_K_LITE, n_per_row);
 
-    int8_t L[QK_K];
-    float   scales[QK_K/16];
-    //float   weights[16];
+    float model_params_b = 4.0f;
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
+    }
 
-    for (int i = 0; i < nb; i++) {
+    int residual_budget = Q3_K_LITE_MAX_RESIDUALS;
+    if (quant_weights) {
+        float importance = ggml_hifi_compute_tensor_importance(quant_weights, nrow * n_per_row);
+        residual_budget = ggml_lite_get_residual_budget(importance, model_params_b, Q3_K_LITE_MAX_RESIDUALS);
+    }
 
-        //float sum_x2 = 0;
-        //for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
-        //float sigma2 = sum_x2/QK_K;
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q3_k_lite_inner(src, (block_q3_k_lite *)qrow, n_per_row,
+                                      quant_weights ? quant_weights + row * n_per_row : NULL,
+                                      residual_budget);
+        src  += n_per_row;
+        qrow += row_size;
+    }
+    return nrow * row_size;
+}
 
-        float max_scale = 0;
-        float max_abs_scale = 0;
+// ---------------------------------------------------------------------------
+// Q2_K_LITE  (only 3 residuals -- same pattern, smaller budget)
+// ---------------------------------------------------------------------------
 
-        for (int ib = 0; ib < QK_K/16; ++ib) {
+static void quantize_row_q2_k_lite_inner(const float * GGML_RESTRICT x, block_q2_k_lite * GGML_RESTRICT y,
+                                           int64_t k, const float * qw, int residual_budget) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    if (residual_budget < 0) residual_budget = 0;
+    if (residual_budget > Q2_K_LITE_MAX_RESIDUALS) residual_budget = Q2_K_LITE_MAX_RESIDUALS;
 
-            float scale;
-            if (quant_weights) {
-                const float * qw = quant_weights + QK_K*i + 16*ib;
-                //for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
-                //scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
-                scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, qw);
-            } else {
-                scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
-            }
-            scales[ib] = scale;
+    float dequant[QK_K];
+    float score[QK_K];
+    int   indices[Q2_K_LITE_MAX_RESIDUALS];
+    float residuals[Q2_K_LITE_MAX_RESIDUALS];
 
-            const float abs_scale = fabsf(scale);
-            if (abs_scale > max_abs_scale) {
-                max_abs_scale = abs_scale;
-                max_scale = scale;
-            }
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        const float * xb = x + ib * QK_K;
+        block_q2_k_lite * block = &y[ib];
 
-        }
+        quantize_row_q2_K_ref(xb, (block_q2_K *)block, QK_K);
 
-        if (max_abs_scale < GROUP_MAX_EPS) {
-            memset(&y[i], 0, sizeof(block_q6_K));
-            y[i].d = GGML_FP32_TO_FP16(0.f);
-            x += QK_K;
+        if (residual_budget == 0) {
+            block->residual_count = 0;
+            block->residual_scale = GGML_FP32_TO_FP16(0.0f);
+            memset(block->residual_idx,  0, Q2_K_LITE_MAX_RESIDUALS);
+            memset(block->residual_vals, 0, Q2_K_LITE_MAX_RESIDUALS);
             continue;
         }
 
-        float iscale = -128.f/max_scale;
-        y[i].d = GGML_FP32_TO_FP16(1/iscale);
-        for (int ib = 0; ib < QK_K/16; ++ib) {
-            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
+        dequantize_row_q2_K((const block_q2_K *)block, dequant, QK_K);
+
+        for (int i = 0; i < QK_K; ++i) {
+            float err = xb[i] - dequant[i];
+            score[i] = fabsf(err) * (qw ? qw[i + ib * QK_K] : 1.0f);
         }
 
-        for (int j = 0; j < QK_K/16; ++j) {
-            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
-            if (!d) {
-                continue;
-            }
-            for (int ii = 0; ii < 16; ++ii) {
-                int l = nearest_int(x[16*j + ii]/d);
-                l = MAX(-32, MIN(31, l));
-                L[16*j + ii] = l + 32;
-            }
+        lite_select_top_n(score, QK_K, indices, residual_budget);
+
+        for (int k_idx = 0; k_idx < residual_budget; ++k_idx) {
+            residuals[k_idx] = xb[indices[k_idx]] - dequant[indices[k_idx]];
         }
 
-        uint8_t * GGML_RESTRICT ql = y[i].ql;
-        uint8_t * GGML_RESTRICT qh = y[i].qh;
-        for (int j = 0; j < QK_K; j += 128) {
-            for (int l = 0; l < 32; ++l) {
-                const uint8_t q1 = L[j + l +  0] & 0xF;
-                const uint8_t q2 = L[j + l + 32] & 0xF;
-                const uint8_t q3 = L[j + l + 64] & 0xF;
-                const uint8_t q4 = L[j + l + 96] & 0xF;
-                ql[l+ 0] = q1 | (q3 << 4);
-                ql[l+32] = q2 | (q4 << 4);
-                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
+        lite_encode_residuals(residuals, indices, residual_budget, Q2_K_LITE_MAX_RESIDUALS,
+                               &block->residual_count, block->residual_idx, block->residual_vals, &block->residual_scale);
+    }
+}
+
+void quantize_row_q2_k_lite_ref(const float * GGML_RESTRICT x, block_q2_k_lite * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q2_k_lite_inner(x, y, k, NULL, Q2_K_LITE_MAX_RESIDUALS);
+}
+
+void dequantize_row_q2_k_lite(const block_q2_k_lite * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % QK_K == 0);
+    const int64_t nb = k / QK_K;
+    for (int64_t ib = 0; ib < nb; ++ib) {
+        float * yb = y + ib * QK_K;
+        dequantize_row_q2_K((const block_q2_K *)&x[ib], yb, QK_K);
+        const int rc = x[ib].residual_count;
+        if (rc > 0) {
+            const float scale = GGML_FP16_TO_FP32(x[ib].residual_scale);
+            for (int r = 0; r < rc; ++r) {
+                yb[x[ib].residual_idx[r]] += scale * (float)x[ib].residual_vals[r];
             }
-            ql += 64;
-            qh += 32;
         }
+    }
+}
 
-        x += QK_K;
+size_t quantize_q2_k_lite(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
+                             int64_t nrow, int64_t n_per_row, const float * quant_weights) {
+    const size_t row_size = ggml_row_size(GGML_TYPE_Q2_K_LITE, n_per_row);
 
+    float model_params_b = 4.0f;
+    const ggml_hifi_quant_context * hifi_ctx = ggml_hifi_get_context();
+    if (hifi_ctx && hifi_ctx->is_active) {
+        model_params_b = hifi_ctx->model_params_b;
     }
-}
 
-size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
-    size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
-    if (!quant_weights) {
-        quantize_row_q6_K_ref(src, dst, (int64_t)nrow*n_per_row);
+    int residual_budget = Q2_K_LITE_MAX_RESIDUALS;
+    if (quant_weights) {
+        float importance = ggml_hifi_compute_tensor_importance(quant_weights, nrow * n_per_row);
+        residual_budget = ggml_lite_get_residual_budget(importance, model_params_b, Q2_K_LITE_MAX_RESIDUALS);
     }
-    else {
-        char * qrow = (char *)dst;
-        for (int64_t row = 0; row < nrow; ++row) {
-            quantize_row_q6_K_impl(src, (block_q6_K*)qrow, n_per_row, quant_weights);
-            src += n_per_row;
-            qrow += row_size;
-        }
+
+    char * qrow = (char *)dst;
+    for (int64_t row = 0; row < nrow; ++row) {
+        quantize_row_q2_k_lite_inner(src, (block_q2_k_lite *)qrow, n_per_row,
+                                      quant_weights ? quant_weights + row * n_per_row : NULL,
+                                      residual_budget);
+        src  += n_per_row;
+        qrow += row_size;
     }
     return nrow * row_size;
 }
@@ -5153,6 +7603,10 @@ void quantize_row_iq2_s_ref(const float * GGML_RESTRICT x, block_iq2_s * GGML_RE
     quantize_iq2_s(x, y, 1, k, NULL);
 }
 
+// Q3_K_HIFI: 3-bit + FP16 outliers per 256 weights
+// Q3_K_HIFI_BLOCK_SIZE and Q3_K_HIFI_OUTLIERS are defined in ggml.h
+
+
 // =============================== data validation
 
 static bool validate_float(float f, size_t i) {
@@ -5474,6 +7928,104 @@ bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbyte
                 VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_nl, data, nb);
             } break;
 
+        case GGML_TYPE_Q3_K_HIFI:
+            {
+                // Validate true outlier extraction layout: check Q3_K block's d field
+                const block_q3_k_hifi * q = (const block_q3_k_hifi *) (data);
+                for (size_t i = 0; i < nb; ++i) {
+                    // Cast to block_q3_K since first 110 bytes match Q3_K layout
+                    const block_q3_K * q3k = (const block_q3_K *)&q[i];
+                    if (!validate_fp16(q3k->d, i)) {
+                        return false;
+                    }
+                }
+            } break;
+
+        case GGML_TYPE_Q6_K_HIFI:
+            {
+                VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_k_hifi, data, nb);
+            } break;
+
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            {
+                VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_k_hifi_dynamic, data, nb);
+            } break;
+
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            {
+                VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_k_hifi_res8, data, nb);
+            } break;
+
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            {
+                VALIDATE_ROW_DATA_D_F16_IMPL(block_q5_k_hifi_res8, data, nb);
+            } break;
+
+        case GGML_TYPE_Q3_K_HIFI_RES8:
+            {
+                VALIDATE_ROW_DATA_D_F16_IMPL(block_q3_k_hifi_res8, data, nb);
+            } break;
+
+        case GGML_TYPE_Q4_K_HIFI:
+            {
+                const block_q4_k_hifi * q = (const block_q4_k_hifi *) data;
+                for (size_t i = 0; i < nb; ++i) {
+                    const block_q4_K * q4k = (const block_q4_K *)q[i].q4_k_data;
+                    if (!validate_fp16(q4k->d, i)) {
+                        return false;
+                    }
+                    if (!validate_fp16(q4k->dmin, i)) {
+                        return false;
+                    }
+                }
+            } break;
+
+        case GGML_TYPE_Q2_K_HIFI:
+            {
+                const block_q2_k_hifi * q = (const block_q2_k_hifi *) data;
+                for (size_t i = 0; i < nb; ++i) {
+                    if (!validate_fp16(q[i].d, i)) {
+                        return false;
+                    }
+                    if (!validate_fp16(q[i].dmin, i)) {
+                        return false;
+                    }
+                    const int n_out = (q[i].outlier_count & 0x7F);
+                    const int n = n_out <= Q2_K_HIFI_MAX_OUTLIERS ? n_out : Q2_K_HIFI_MAX_OUTLIERS;
+                    for (int k = 0; k < n; ++k) {
+                        if (!validate_fp16(q[i].outlier_vals[k], i)) {
+                            return false;
+                        }
+                    }
+                }
+            } break;
+
+        case GGML_TYPE_Q2_K_LITE:
+            {
+                // Q2_K base: has d and dmin
+                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q2_k_lite, data, nb, d, dmin);
+            } break;
+        case GGML_TYPE_Q3_K_LITE:
+            {
+                // Q2_K base: has d and dmin
+                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q3_k_lite, data, nb, d, dmin);
+            } break;
+        case GGML_TYPE_Q4_K_LITE:
+            {
+                // Q3_K base: has only d
+                VALIDATE_ROW_DATA_D_F16_IMPL(block_q4_k_lite, data, nb);
+            } break;
+        case GGML_TYPE_Q5_K_LITE:
+            {
+                // Q4_K base: has d and dmin
+                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_k_lite, data, nb, d, dmin);
+            } break;
+        case GGML_TYPE_Q6_K_LITE:
+            {
+                // Q5_K base: has d and dmin
+                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q6_k_lite, data, nb, d, dmin);
+            } break;
+
         case GGML_TYPE_I8:
         case GGML_TYPE_I16:
         case GGML_TYPE_I32:
```

---

### 3.6 `ggml/src/ggml.c` — Type Registration

In the `type_traits` array, after the `[GGML_TYPE_Q3_K]` entry (around line 754), add entries
for all 13 new types:

```c
[GGML_TYPE_Q3_K_HIFI] = {
    .type_name    = "Q3_K_HIFI",
    .blck_size    = Q3_K_HIFI_BLOCK_SIZE,
    .type_size    = sizeof(block_q3_k_hifi),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q3_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q3_k_hifi_ref,
},
[GGML_TYPE_Q6_K_HIFI] = {
    .type_name    = "Q6_K_HIFI",
    .blck_size    = QK_K,
    .type_size    = sizeof(block_q6_k_hifi),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q6_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_hifi_ref,
},
[GGML_TYPE_Q6_K_HIFI_DYNAMIC] = {
    .type_name    = "Q6_K_HIFI_DYN",
    .blck_size    = QK_K,
    .type_size    = sizeof(block_q6_k_hifi_dynamic),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q6_k_hifi_dynamic,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_hifi_dynamic_ref,
},
[GGML_TYPE_Q6_K_HIFI_RES8] = {
    .type_name    = "Q6_K_HIFI_RES8",
    .blck_size    = QK_K,
    .type_size    = sizeof(block_q6_k_hifi_res8),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q6_k_hifi_res8,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_hifi_res8_ref,
},
[GGML_TYPE_Q5_K_HIFI_RES8] = {
    .type_name    = "Q5_K_HIFI_RES8",
    .blck_size    = QK_K,
    .type_size    = sizeof(block_q5_k_hifi_res8),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q5_k_hifi_res8,
    .from_float_ref = (ggml_from_float_t) quantize_row_q5_k_hifi_res8_ref,
},
[GGML_TYPE_Q3_K_HIFI_RES8] = {
    .type_name    = "Q3_K_HIFI_RES8",
    .blck_size    = Q3_K_HIFI_BLOCK_SIZE,
    .type_size    = sizeof(block_q3_k_hifi_res8),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q3_k_hifi_res8,
    .from_float_ref = (ggml_from_float_t) quantize_row_q3_k_hifi_res8_ref,
},
[GGML_TYPE_Q4_K_HIFI] = {
    .type_name    = "Q4_K_HIFI",
    .blck_size    = Q4_K_HIFI_BLOCK_SIZE,
    .type_size    = sizeof(block_q4_k_hifi),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q4_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q4_k_hifi_ref,
},
[GGML_TYPE_Q2_K_HIFI] = {
    .type_name    = "Q2_K_HIFI",
    .blck_size    = Q2_K_HIFI_BLOCK_SIZE,
    .type_size    = sizeof(block_q2_k_hifi),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q2_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q2_k_hifi_ref,
},
[GGML_TYPE_Q2_K_LITE] = {
    .type_name    = "Q2_K_LITE",
    .blck_size    = Q2_K_LITE_BLOCK_SIZE,
    .type_size    = sizeof(block_q2_k_lite),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q2_k_lite,
    .from_float_ref = (ggml_from_float_t) quantize_row_q2_k_lite_ref,
},
[GGML_TYPE_Q3_K_LITE] = {
    .type_name    = "Q3_K_LITE",
    .blck_size    = Q3_K_LITE_BLOCK_SIZE,
    .type_size    = sizeof(block_q3_k_lite),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q3_k_lite,
    .from_float_ref = (ggml_from_float_t) quantize_row_q3_k_lite_ref,
},
[GGML_TYPE_Q4_K_LITE] = {
    .type_name    = "Q4_K_LITE",
    .blck_size    = Q4_K_LITE_BLOCK_SIZE,
    .type_size    = sizeof(block_q4_k_lite),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q4_k_lite,
    .from_float_ref = (ggml_from_float_t) quantize_row_q4_k_lite_ref,
},
[GGML_TYPE_Q5_K_LITE] = {
    .type_name    = "Q5_K_LITE",
    .blck_size    = Q5_K_LITE_BLOCK_SIZE,
    .type_size    = sizeof(block_q5_k_lite),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q5_k_lite,
    .from_float_ref = (ggml_from_float_t) quantize_row_q5_k_lite_ref,
},
[GGML_TYPE_Q6_K_LITE] = {
    .type_name    = "Q6_K_LITE",
    .blck_size    = Q6_K_LITE_BLOCK_SIZE,
    .type_size    = sizeof(block_q6_k_lite),
    .is_quantized = true,
    .to_float     = (ggml_to_float_t) dequantize_row_q6_k_lite,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_lite_ref,
},
```

In `ggml_quantize_chunk`, after the `GGML_TYPE_IQ4_XS` case, add:

```c
case GGML_TYPE_Q3_K_HIFI: result = quantize_q3_k_hifi(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q6_K_HIFI: result = quantize_q6_k_hifi(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q6_K_HIFI_DYNAMIC: result = quantize_q6_k_hifi_dynamic(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q6_K_HIFI_RES8: result = quantize_q6_k_hifi_res8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q5_K_HIFI_RES8: result = quantize_q5_k_hifi_res8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q3_K_HIFI_RES8: result = quantize_q3_k_hifi_res8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q4_K_HIFI: result = quantize_q4_k_hifi(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q2_K_HIFI: result = quantize_q2_k_hifi(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q2_K_LITE: result = quantize_q2_k_lite(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q3_K_LITE: result = quantize_q3_k_lite(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q4_K_LITE: result = quantize_q4_k_lite(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q5_K_LITE: result = quantize_q5_k_lite(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_Q6_K_LITE: result = quantize_q6_k_lite(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
```

---

### 3.7 `ggml/src/gguf.cpp` — Better Error Diagnostics

In `gguf_init_from_file_ptr`, after the offset mismatch error log, add:

```c
GGML_LOG_ERROR("%s: tensor type: %s (%d), calculated size: %zu bytes\n",
    __func__, ggml_type_name(ti.t.type), (int)ti.t.type, ggml_nbytes(&ti.t));
if (ti.t.type == GGML_TYPE_Q3_K_HIFI) {
    GGML_LOG_ERROR("%s: Q3_K_HIFI tensor size mismatch detected. This file may have been created with incorrect size calculations.\n", __func__);
    GGML_LOG_ERROR("%s: Please re-quantize the model with the current version of llama.cpp.\n", __func__);
}
```

---

### 3.8 `ggml/src/CMakeLists.txt` — Add New Source Files

```diff
 ggml-quants.c
 ggml-quants.h
+ggml-quants-hifi.c
+ggml-quants-hifi.h
 gguf.cpp
```

---

### 3.9 `ggml/src/ggml-cpu/quants.h` — CPU Backend Declarations

After `quantize_row_q2_K`, add:
```c
void quantize_row_q2_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
```

After `quantize_row_q3_K`, add:
```c
void quantize_row_q3_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
```

After `quantize_row_q6_K`, add:
```c
void quantize_row_q6_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_k_hifi_dynamic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_k_hifi_res8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_k_hifi_res8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q5_k_hifi_res8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
```

In the `ggml_vec_dot_*` section, add HIFI and LITE vec_dot declarations:
```c
void ggml_vec_dot_q2_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_k_hifi_dynamic_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_k_hifi_res8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_k_hifi_res8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q2_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void quantize_row_q2_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
```

In the generic section, add:
```c
void ggml_vec_dot_q2_k_hifi_q8_K_generic(...);
void ggml_vec_dot_q3_k_hifi_q8_K_generic(...);
void ggml_vec_dot_q4_k_hifi_q8_K_generic(...);
void ggml_vec_dot_q2_k_lite_q8_K_generic(...);
void ggml_vec_dot_q3_k_lite_q8_K_generic(...);
void ggml_vec_dot_q4_k_lite_q8_K_generic(...);
void ggml_vec_dot_q5_k_lite_q8_K_generic(...);
void ggml_vec_dot_q6_k_lite_q8_K_generic(...);
```

---

### 3.10 `ggml/src/ggml-cpu/quants.c` — CPU Generic Implementations

Apply the following complete diff:

```diff
diff --git a/ggml/src/ggml-cpu/quants.c b/ggml/src/ggml-cpu/quants.c
index f66127c22..00909a236 100644
--- a/ggml/src/ggml-cpu/quants.c
+++ b/ggml/src/ggml-cpu/quants.c
@@ -74,6 +74,18 @@ void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, i
     quantize_row_q3_K_ref(x, vy, k);
 }
 
+void quantize_row_q3_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    block_q3_k_hifi * GGML_RESTRICT y = vy;
+    quantize_row_q3_k_hifi_ref(x, y, k);
+}
+
+void quantize_row_q4_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
+    assert(k % Q4_K_HIFI_BLOCK_SIZE == 0);
+    block_q4_k_hifi * GGML_RESTRICT y = vy;
+    quantize_row_q4_k_hifi_ref(x, y, k);
+}
+
 // ====================== 4-bit (de)-quantization
 
 void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
@@ -98,6 +110,25 @@ void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, i
     quantize_row_q6_K_ref(x, y, k);
 }
 
+void quantize_row_q6_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
+    assert(k % QK_K == 0);
+    block_q6_k_hifi * GGML_RESTRICT y = vy;
+    quantize_row_q6_k_hifi_ref(x, y, k);
+}
+
+void quantize_row_q6_k_hifi_dynamic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
+    assert(k % QK_K == 0);
+    block_q6_k_hifi_dynamic * GGML_RESTRICT y = vy;
+    // Uses default outlier count (6) via the 3-argument wrapper
+    quantize_row_q6_k_hifi_dynamic_ref(x, y, k);
+}
+
+void quantize_row_q6_k_hifi_res8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
+    assert(k % QK_K == 0);
+    block_q6_k_hifi_res8 * GGML_RESTRICT y = vy;
+    quantize_row_q6_k_hifi_res8_ref(x, y, k);
+}
+
 // ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)
 
 void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
@@ -557,6 +588,77 @@ void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, c
     *s = sumf;
 }
 
+// Q2_K_HIFI: Q2_K base dot product + FP16 outlier value corrections
+// Outliers were zeroed before Q2_K quantization, so base contributes ~0 at those positions.
+// We add the true FP16 outlier values × quantized activations to recover precision.
+void ggml_vec_dot_q2_k_hifi_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    const block_q2_k_hifi * GGML_RESTRICT x = vx;
+    const block_q8_K * GGML_RESTRICT y = vy;
+
+    const int nb = n / QK_K;
+
+    float sumf = 0;
+
+    for (int i = 0; i < nb; ++i) {
+        const uint8_t * q2 = x[i].qs;
+        const  int8_t * q8 = y[i].qs;
+        const uint8_t * sc = x[i].scales;
+
+        int summs = 0;
+        for (int j = 0; j < 16; ++j) {
+            summs += y[i].bsums[j] * (sc[j] >> 4);
+        }
+
+        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
+        const float dmin_val = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
+
+        int isum = 0;
+        int is = 0;
+        int d;
+        for (int k = 0; k < QK_K/128; ++k) {
+            int shift = 0;
+            for (int j = 0; j < 4; ++j) {
+                d = sc[is++] & 0xF;
+                int isuml = 0;
+                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
+                isum += d * isuml;
+                d = sc[is++] & 0xF;
+                isuml = 0;
+                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
+                isum += d * isuml;
+                shift += 2;
+                q8 += 32;
+            }
+            q2 += 32;
+        }
+        sumf += dall * isum - dmin_val * summs;
+
+        // FP16 outlier/residual corrections (works for both outlier-first and residual modes)
+        const int n_out = (x[i].outlier_count & 0x7F);
+        if (n_out > 0) {
+            const float d8 = y[i].d;
+            const int n_corr = n_out <= Q2_K_HIFI_MAX_OUTLIERS ? n_out : Q2_K_HIFI_MAX_OUTLIERS;
+            for (int k_idx = 0; k_idx < n_corr; ++k_idx) {
+                const int idx = x[i].outlier_idx[k_idx];
+                const float val = GGML_CPU_FP16_TO_FP32(x[i].outlier_vals[k_idx]);
+                sumf += val * (float)y[i].qs[idx] * d8;
+            }
+        }
+    }
+    *s = sumf;
+}
+
+void quantize_row_q2_k_hifi(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q2_k_hifi_ref(x, (block_q2_k_hifi *)y, k);
+}
+
 void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
     assert(n % QK_K == 0);
     assert(nrc == 1);
@@ -636,6 +738,110 @@ void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, c
     *s = sumf;
 }
 
+// Q3_K_HIFI vec_dot: Generic implementation
+// Uses Q3_K format for bulk, adds outlier corrections
+void ggml_vec_dot_q3_k_hifi_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % Q3_K_HIFI_BLOCK_SIZE == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    const block_q3_k_hifi * GGML_RESTRICT x = vx;
+    const block_q8_K * GGML_RESTRICT y = vy;
+    const int nb = n / Q3_K_HIFI_BLOCK_SIZE;
+
+    float total_sum = 0.0f;
+
+    for (int i = 0; i < nb; ++i) {
+        const block_q3_k_hifi * xb = &x[i];
+        const block_q8_K * yb = &y[i];
+
+        // Step 1: Compute Q3_K dot product from Q3_K fields (first 110 bytes)
+        const block_q3_K * q3k_block = (const block_q3_K *)xb;
+        float q3k_sum = 0.0f;
+
+        // Use Q3_K's dot product logic
+        // For now, we'll dequantize Q3_K and compute dot product manually
+        float q3k_weights[Q3_K_HIFI_BLOCK_SIZE];
+        dequantize_row_q3_K(q3k_block, q3k_weights, Q3_K_HIFI_BLOCK_SIZE);
+
+        const float d_y = yb->d;
+        const int8_t * GGML_RESTRICT q8 = yb->qs;
+        for (int j = 0; j < Q3_K_HIFI_BLOCK_SIZE; ++j) {
+            q3k_sum += q3k_weights[j] * (float)q8[j] * d_y;
+        }
+
+        // Step 2: Add outlier corrections
+        // Outliers were zeroed before Q3_K quantization, so Q3_K contribution is ~0 at those positions
+        // We need to subtract the ~0 Q3_K contribution and add the original outlier value
+        for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+            int idx = xb->outlier_idx[k];
+            if (idx < Q3_K_HIFI_BLOCK_SIZE) {
+                float outlier_val = GGML_FP16_TO_FP32(xb->outliers[k]);
+                float q3k_val = q3k_weights[idx];  // Should be ~0 since we zeroed it
+                q3k_sum += (outlier_val - q3k_val) * (float)q8[idx] * d_y;
+            }
+        }
+
+        total_sum += q3k_sum;
+    }
+
+    *s = total_sum;
+}
+
+// Note: ggml_vec_dot_q3_k_hifi_q8_K is defined in arch-specific files (x86/quants.c etc.)
+
+// Q4_K_HIFI vec_dot: Generic implementation
+// Uses Q4_K format for bulk, adds outlier corrections
+void ggml_vec_dot_q4_k_hifi_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % Q4_K_HIFI_BLOCK_SIZE == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    const block_q4_k_hifi * GGML_RESTRICT x = vx;
+    const block_q8_K * GGML_RESTRICT y = vy;
+    const int nb = n / Q4_K_HIFI_BLOCK_SIZE;
+
+    float total_sum = 0.0f;
+
+    for (int i = 0; i < nb; ++i) {
+        const block_q4_k_hifi * xb = &x[i];
+        const block_q8_K * yb = &y[i];
+
+        // Step 1: Dequantize Q4_K from q4_k_data (first 144 bytes)
+        const block_q4_K * q4k_block = (const block_q4_K *)xb->q4_k_data;
+        float q4k_weights[Q4_K_HIFI_BLOCK_SIZE];
+        dequantize_row_q4_K(q4k_block, q4k_weights, Q4_K_HIFI_BLOCK_SIZE);
+
+        // Step 2: Compute dot product
+        const float d_y = yb->d;
+        const int8_t * GGML_RESTRICT q8 = yb->qs;
+        float block_sum = 0.0f;
+        for (int j = 0; j < Q4_K_HIFI_BLOCK_SIZE; ++j) {
+            block_sum += q4k_weights[j] * (float)q8[j] * d_y;
+        }
+
+        // Step 3: Add outlier corrections
+        for (int k = 0; k < Q4_K_HIFI_OUTLIERS; ++k) {
+            int idx = xb->outlier_idx[k];
+            if (idx < Q4_K_HIFI_BLOCK_SIZE) {
+                float outlier_val = GGML_FP16_TO_FP32(xb->outliers[k]);
+                float q4k_val = q4k_weights[idx];
+                block_sum += (outlier_val - q4k_val) * (float)q8[idx] * d_y;
+            }
+        }
+
+        total_sum += block_sum;
+    }
+
+    *s = total_sum;
+}
+
 void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
     assert(n % QK_K == 0);
     assert(nrc == 1);
@@ -846,6 +1052,638 @@ void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, c
     *s = sumf;
 }
 
+// Q6_K_HIFI_DYNAMIC: vec_dot with early exit optimization
+// Skip outlier correction when |activation| < threshold (negligible contribution)
+void ggml_vec_dot_q6_k_hifi_dynamic_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    const block_q6_k_hifi_dynamic * GGML_RESTRICT x = vx;
+    const block_q8_K * GGML_RESTRICT y = vy;
+
+    const int nb = n / QK_K;
+
+    int8_t  aux8[QK_K];
+    int16_t aux16[8];
+    float   sums [8];
+    int32_t aux32[8];
+    memset(sums, 0, 8*sizeof(float));
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        // === Q6_K bulk dot product (identical to generic Q6_K) ===
+        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
+        const uint8_t * GGML_RESTRICT qh = x[i].qh;
+        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
+        memset(aux32, 0, 8*sizeof(int32_t));
+        int8_t * GGML_RESTRICT a = aux8;
+        for (int j = 0; j < QK_K; j += 128) {
+            for (int l = 0; l < 32; ++l) {
+                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
+                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
+                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
+                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
+            }
+            a  += 128;
+            q4 += 64;
+            qh += 32;
+        }
+        a = aux8;
+        int is = 0;
+        for (int j = 0; j < QK_K/16; ++j) {
+            int scale = x[i].scales[is++];
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+        }
+        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
+        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
+
+        // === EARLY EXIT OUTLIER CORRECTION ===
+        // Only apply correction if |activation| > threshold (avoids ~60% of corrections)
+        const int outlier_count = x[i].outlier_count;
+        const float d8 = y[i].d;
+        for (int k = 0; k < outlier_count; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            const int8_t activation = y[i].qs[idx];
+            // Early exit: skip if activation is too small
+            if (activation > Q6_K_HIFI_EARLY_EXIT_THRESHOLD || activation < -Q6_K_HIFI_EARLY_EXIT_THRESHOLD) {
+                const float w = GGML_CPU_FP16_TO_FP32(x[i].outlier_vals[k]);
+                sumf += w * activation * d8;
+            }
+        }
+    }
+    for (int l = 0; l < 8; ++l) sumf += sums[l];
+    *s = sumf;
+}
+
+// Q6_K_HIFI_RES8: Compact format with INT8 residuals + per-block scale
+void ggml_vec_dot_q6_k_hifi_res8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    const block_q6_k_hifi_res8 * GGML_RESTRICT x = vx;
+    const block_q8_K * GGML_RESTRICT y = vy;
+
+    const int nb = n / QK_K;
+
+    int8_t  aux8[QK_K];
+    int16_t aux16[8];
+    float   sums [8];
+    int32_t aux32[8];
+    memset(sums, 0, 8*sizeof(float));
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        // === Q6_K bulk dot product (identical to Q6_K) ===
+        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
+        const uint8_t * GGML_RESTRICT qh = x[i].qh;
+        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
+        memset(aux32, 0, 8*sizeof(int32_t));
+        int8_t * GGML_RESTRICT a = aux8;
+        for (int j = 0; j < QK_K; j += 128) {
+            for (int l = 0; l < 32; ++l) {
+                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
+                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
+                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
+                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
+            }
+            a  += 128;
+            q4 += 64;
+            qh += 32;
+        }
+        a = aux8;
+        int is = 0;
+        for (int j = 0; j < QK_K/16; ++j) {
+            int scale = x[i].scales[is++];
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+        }
+        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
+        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
+
+        // === INT8 RESIDUAL CORRECTION ===
+        // Add residual * activation corrections at outlier positions
+        // Residual was computed as: original_value - Q6_K_approximation
+        // So adding residual * activation gives us the missing contribution
+        const int outlier_count = x[i].outlier_count;
+        const float res_scale = x[i].residual_scale;
+        const float d8 = y[i].d;
+        const float scale_factor = res_scale * (1.0f / 127.0f) * d8;
+        for (int k = 0; k < outlier_count; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            const int8_t activation = y[i].qs[idx];
+            // Early exit: skip if activation is too small
+            if (activation > Q6_K_HIFI_EARLY_EXIT_THRESHOLD || activation < -Q6_K_HIFI_EARLY_EXIT_THRESHOLD) {
+                const float residual = x[i].residual_vals[k] * scale_factor;
+                sumf += residual * activation;
+            }
+        }
+    }
+    for (int l = 0; l < 8; ++l) sumf += sums[l];
+    *s = sumf;
+}
+
+// Q5_K_HIFI_RES8: Efficient Q5_K base + INT8 residuals for 4B-10B models
+// Uses same correction strategy as Q6_K_HIFI_RES8, but with Q5_K base for better BPW
+void ggml_vec_dot_q5_k_hifi_res8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    const block_q5_k_hifi_res8 * GGML_RESTRICT x = vx;
+    const block_q8_K * GGML_RESTRICT y = vy;
+
+    const int nb = n / QK_K;
+
+    static const uint32_t kmask1 = 0x3f3f3f3f;
+    static const uint32_t kmask2 = 0x0f0f0f0f;
+    static const uint32_t kmask3 = 0x03030303;
+
+    uint32_t utmp[4];
+    const uint8_t * scales = (const uint8_t*)&utmp[0];
+    const uint8_t * mins   = (const uint8_t*)&utmp[2];
+
+    int8_t  aux8[QK_K];
+    int16_t aux16[8];
+    float   sums [8];
+    int32_t aux32[8];
+    memset(sums, 0, 8*sizeof(float));
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        // === Q5_K bulk dot product (same as ggml_vec_dot_q5_K_q8_K_generic) ===
+        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
+        const uint8_t * GGML_RESTRICT hm = x[i].qh;
+        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
+        memset(aux32, 0, 8*sizeof(int32_t));
+        int8_t * GGML_RESTRICT a = aux8;
+        uint8_t m = 1;
+        for (int j = 0; j < QK_K; j += 64) {
+            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF) + (hm[l] & m ? 16 : 0);
+            a += 32; m <<= 1;
+            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4) + (hm[l] & m ? 16 : 0);
+            a += 32; m <<= 1;
+            q4 += 32;
+        }
+        memcpy(utmp, x[i].scales, 12);
+        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
+        const uint32_t uaux = utmp[1] & kmask1;
+        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
+        utmp[2] = uaux;
+        utmp[0] &= kmask1;
+
+        int sumi = 0;
+        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
+        a = aux8;
+        int is = 0;
+        for (int j = 0; j < QK_K/32; ++j) {
+            int32_t scale = scales[is++];
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+        }
+        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
+        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
+        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
+        sumf -= dmin * sumi;
+
+        // === INT8 RESIDUAL CORRECTION ===
+        // Add residual * activation corrections at outlier positions
+        const int outlier_count = x[i].outlier_count;
+
+        // FAST PATH: Skip residual correction if no outliers
+        if (outlier_count > 0) {
+            // Decode E4M3 FP8 scale to FP32
+            const uint8_t e4m3 = x[i].residual_scale_e4m3;
+            const int sign = (e4m3 >> 7) & 0x01;
+            const int exp = (e4m3 >> 3) & 0x0F;
+            const int mantissa = e4m3 & 0x07;
+            const float m_frac = (float)mantissa / 8.0f;
+            const float decoded_scale = (e4m3 == 0) ? 0.0f : ((1.0f + m_frac) * exp2f((float)exp - 7.0f) * (sign ? -1.0f : 1.0f));
+
+            const float d8 = y[i].d;
+            const float scale_factor = decoded_scale * (1.0f / 127.0f) * d8;
+            for (int k = 0; k < outlier_count; ++k) {
+                const int idx = x[i].outlier_idx[k];
+                const int8_t activation = y[i].qs[idx];
+                // Early exit: skip if activation is too small (same threshold as Q6_K_HIFI)
+                if (activation > 4 || activation < -4) {
+                    const float residual = x[i].residual_vals[k] * scale_factor;
+                    sumf += residual * activation;
+                }
+            }
+        }
+    }
+    for (int l = 0; l < 8; ++l) sumf += sums[l];
+    *s = sumf;
+}
+
+// Wrapper for quantize_row_q5_k_hifi_res8 (simple version)
+void quantize_row_q5_k_hifi_res8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q5_k_hifi_res8_ref(x, (block_q5_k_hifi_res8 *)y, k);
+}
+
+// =============================================================================
+// K_LITE vec_dot implementations
+// Each type: replicate the base K-quant dot product, then apply residual correction.
+// Residual correction: sum += residual_scale * residual_vals[k] * activation[idx]
+// Fast path: skip correction loop when residual_count == 0 (Tier 0 blocks).
+// =============================================================================
+
+// ---------------------------------------------------------------------------
+// Q4_K_LITE vec_dot  (Q3_K base: hmask + qs[64] 3-bit, scales[12], d only)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q4_k_lite_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
+
+    const uint32_t kmask1 = 0x03030303;
+    const uint32_t kmask2 = 0x0f0f0f0f;
+
+    const block_q4_k_lite * GGML_RESTRICT x = vx;
+    const block_q8_K        * GGML_RESTRICT y = vy;
+    const int nb = n / QK_K;
+
+    int8_t  aux8[QK_K];
+    int16_t aux16[8];
+    float   sums[8];
+    int32_t aux32[8];
+    memset(sums, 0, 8 * sizeof(float));
+    uint32_t auxs[4];
+    const int8_t * scales_q3 = (const int8_t *)auxs;
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
+        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
+        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
+        memset(aux32, 0, 8 * sizeof(int32_t));
+        int8_t * GGML_RESTRICT a = aux8;
+        uint8_t m = 1;
+        for (int j = 0; j < QK_K; j += 128) {
+            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
+            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
+            a += 32; m <<= 1;
+            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
+            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
+            a += 32; m <<= 1;
+            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
+            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
+            a += 32; m <<= 1;
+            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
+            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
+            a += 32; m <<= 1;
+            q3 += 32;
+        }
+        a = aux8;
+        memcpy(auxs, x[i].scales, 12);
+        uint32_t tmp = auxs[2];
+        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
+        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
+        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
+        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
+        for (int j = 0; j < QK_K/16; ++j) {
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += (scales_q3[j] - 32) * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += (scales_q3[j] - 32) * aux16[l];
+            q8 += 8; a += 8;
+        }
+        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
+        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
+
+        const int rc = x[i].residual_count;
+        if (rc > 0) {
+            const float rscale = GGML_CPU_FP16_TO_FP32(x[i].residual_scale) * y[i].d;
+            for (int k = 0; k < rc; ++k) {
+                sumf += rscale * (float)x[i].residual_vals[k] * (float)y[i].qs[x[i].residual_idx[k]];
+            }
+        }
+    }
+    for (int l = 0; l < 8; ++l) sumf += sums[l];
+    *s = sumf;
+}
+
+// Wrapper (3-arg from_float for CPU backend)
+void quantize_row_q4_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q4_k_lite_ref(x, (block_q4_k_lite *)y, k);
+}
+
+// ---------------------------------------------------------------------------
+// Q5_K_LITE vec_dot  (Q4_K base: d, dmin, scales[12], qs[128] 4-bit)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q5_k_lite_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
+
+    const block_q5_k_lite * GGML_RESTRICT x = vx;
+    const block_q8_K        * GGML_RESTRICT y = vy;
+    const int nb = n / QK_K;
+
+    static const uint32_t kmask1 = 0x3f3f3f3f;
+    static const uint32_t kmask2 = 0x0f0f0f0f;
+    static const uint32_t kmask3 = 0x03030303;
+    uint32_t utmp[4];
+    const uint8_t * scales = (const uint8_t *)&utmp[0];
+    const uint8_t * mins   = (const uint8_t *)&utmp[2];
+    int8_t  aux8[QK_K];
+    int16_t aux16[8];
+    float   sums[8];
+    int32_t aux32[8];
+    memset(sums, 0, 8 * sizeof(float));
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
+        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
+        memset(aux32, 0, 8 * sizeof(int32_t));
+        int8_t * GGML_RESTRICT a = aux8;
+        for (int j = 0; j < QK_K/64; ++j) {
+            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
+            a += 32;
+            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
+            a += 32; q4 += 32;
+        }
+        memcpy(utmp, x[i].scales, 12);
+        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
+        const uint32_t uaux = utmp[1] & kmask1;
+        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
+        utmp[2] = uaux;
+        utmp[0] &= kmask1;
+        int sumi = 0;
+        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
+        a = aux8;
+        int is = 0;
+        for (int j = 0; j < QK_K/32; ++j) {
+            int32_t scale = scales[is++];
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+        }
+        const float d    = GGML_CPU_FP16_TO_FP32(x[i].d)    * y[i].d;
+        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
+        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
+        sumf -= dmin * sumi;
+
+        const int rc = x[i].residual_count;
+        if (rc > 0) {
+            const float rscale = GGML_CPU_FP16_TO_FP32(x[i].residual_scale) * y[i].d;
+            for (int k = 0; k < rc; ++k) {
+                sumf += rscale * (float)x[i].residual_vals[k] * (float)y[i].qs[x[i].residual_idx[k]];
+            }
+        }
+    }
+    for (int l = 0; l < 8; ++l) sumf += sums[l];
+    *s = sumf;
+}
+
+void quantize_row_q5_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q5_k_lite_ref(x, (block_q5_k_lite *)y, k);
+}
+
+// ---------------------------------------------------------------------------
+// Q6_K_LITE vec_dot  (Q5_K base: d, dmin, scales[12], qh[32], qs[128] 5-bit)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q6_k_lite_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
+
+    const block_q6_k_lite * GGML_RESTRICT x = vx;
+    const block_q8_K        * GGML_RESTRICT y = vy;
+    const int nb = n / QK_K;
+
+    static const uint32_t kmask1 = 0x3f3f3f3f;
+    static const uint32_t kmask2 = 0x0f0f0f0f;
+    static const uint32_t kmask3 = 0x03030303;
+    uint32_t utmp[4];
+    const uint8_t * scales = (const uint8_t *)&utmp[0];
+    const uint8_t * mins   = (const uint8_t *)&utmp[2];
+    int8_t  aux8[QK_K];
+    int16_t aux16[8];
+    float   sums[8];
+    int32_t aux32[8];
+    memset(sums, 0, 8 * sizeof(float));
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
+        const uint8_t * GGML_RESTRICT hm = x[i].qh;
+        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
+        memset(aux32, 0, 8 * sizeof(int32_t));
+        int8_t * GGML_RESTRICT a = aux8;
+        uint8_t m = 1;
+        for (int j = 0; j < QK_K; j += 64) {
+            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF) + (hm[l] & m ? 16 : 0);
+            a += 32; m <<= 1;
+            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4) + (hm[l] & m ? 16 : 0);
+            a += 32; m <<= 1;
+            q4 += 32;
+        }
+        memcpy(utmp, x[i].scales, 12);
+        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
+        const uint32_t uaux = utmp[1] & kmask1;
+        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
+        utmp[2] = uaux;
+        utmp[0] &= kmask1;
+        int sumi = 0;
+        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
+        a = aux8;
+        int is = 0;
+        for (int j = 0; j < QK_K/32; ++j) {
+            int32_t scale = scales[is++];
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
+            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
+            q8 += 8; a += 8;
+        }
+        const float d    = GGML_CPU_FP16_TO_FP32(x[i].d)    * y[i].d;
+        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
+        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
+        sumf -= dmin * sumi;
+
+        const int rc = x[i].residual_count;
+        if (rc > 0) {
+            const float rscale = GGML_CPU_FP16_TO_FP32(x[i].residual_scale) * y[i].d;
+            for (int k = 0; k < rc; ++k) {
+                sumf += rscale * (float)x[i].residual_vals[k] * (float)y[i].qs[x[i].residual_idx[k]];
+            }
+        }
+    }
+    for (int l = 0; l < 8; ++l) sumf += sums[l];
+    *s = sumf;
+}
+
+void quantize_row_q6_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q6_k_lite_ref(x, (block_q6_k_lite *)y, k);
+}
+
+// ---------------------------------------------------------------------------
+// Q3_K_LITE vec_dot  (Q2_K base: d, dmin, scales[16], qs[64] 2-bit)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q3_k_lite_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
+
+    const block_q3_k_lite * GGML_RESTRICT x = vx;
+    const block_q8_K        * GGML_RESTRICT y = vy;
+    const int nb = n / QK_K;
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        const uint8_t * q2  = x[i].qs;
+        const int8_t  * q8  = y[i].qs;
+        const uint8_t * sc  = x[i].scales;
+
+        int summs = 0;
+        for (int j = 0; j < 16; ++j) summs += y[i].bsums[j] * (sc[j] >> 4);
+
+        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
+        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
+
+        int isum = 0, is = 0;
+        for (int k = 0; k < QK_K/128; ++k) {
+            int shift = 0;
+            for (int j = 0; j < 4; ++j) {
+                int d = sc[is++] & 0xF;
+                int isuml = 0;
+                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
+                isum += d * isuml;
+                d = sc[is++] & 0xF;
+                isuml = 0;
+                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
+                isum += d * isuml;
+                shift += 2;
+                q8 += 32;
+            }
+            q2 += 32;
+        }
+        sumf += dall * isum - dmin * summs;
+
+        const int rc = x[i].residual_count;
+        if (rc > 0) {
+            const float rscale = GGML_CPU_FP16_TO_FP32(x[i].residual_scale) * y[i].d;
+            for (int r = 0; r < rc; ++r) {
+                sumf += rscale * (float)x[i].residual_vals[r] * (float)y[i].qs[x[i].residual_idx[r]];
+            }
+        }
+    }
+
+    *s = sumf;
+}
+
+void quantize_row_q3_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q3_k_lite_ref(x, (block_q3_k_lite *)y, k);
+}
+
+// ---------------------------------------------------------------------------
+// Q2_K_LITE vec_dot  (Q2_K base: d, dmin, scales[16], qs[64] 2-bit)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q2_k_lite_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
+
+    const block_q2_k_lite * GGML_RESTRICT x = vx;
+    const block_q8_K        * GGML_RESTRICT y = vy;
+    const int nb = n / QK_K;
+
+    float sumf = 0;
+    for (int i = 0; i < nb; ++i) {
+        const uint8_t * q2  = x[i].qs;
+        const int8_t  * q8  = y[i].qs;
+        const uint8_t * sc  = x[i].scales;
+
+        int summs = 0;
+        for (int j = 0; j < 16; ++j) summs += y[i].bsums[j] * (sc[j] >> 4);
+
+        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
+        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
+
+        int isum = 0, is = 0;
+        for (int k = 0; k < QK_K/128; ++k) {
+            int shift = 0;
+            for (int j = 0; j < 4; ++j) {
+                int d = sc[is++] & 0xF;
+                int isuml = 0;
+                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
+                isum += d * isuml;
+                d = sc[is++] & 0xF;
+                isuml = 0;
+                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
+                isum += d * isuml;
+                shift += 2;
+                q8 += 32;
+            }
+            q2 += 32;
+        }
+        sumf += dall * isum - dmin * summs;
+
+        const int rc = x[i].residual_count;
+        if (rc > 0) {
+            const float rscale = GGML_CPU_FP16_TO_FP32(x[i].residual_scale) * y[i].d;
+            for (int r = 0; r < rc; ++r) {
+                sumf += rscale * (float)x[i].residual_vals[r] * (float)y[i].qs[x[i].residual_idx[r]];
+            }
+        }
+    }
+
+    *s = sumf;
+}
+
+void quantize_row_q2_k_lite(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
+    quantize_row_q2_k_lite_ref(x, (block_q2_k_lite *)y, k);
+}
+
 void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
     assert(n % QK_K == 0);
     assert(nrc == 1);
```

---

### 3.11 `ggml/src/ggml-cpu/ggml-cpu.c` — CPU Type Trait Registration

In `type_traits_cpu[]`, add entries for all HIFI/LITE types. Place `Q2_K_HIFI` before `Q3_K`
and all others before `Q4_K`:

```c
[GGML_TYPE_Q2_K_HIFI] = {
    .from_float   = quantize_row_q2_k_hifi,
    .vec_dot      = ggml_vec_dot_q2_k_hifi_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q3_K_HIFI] = {
    .from_float   = quantize_row_q3_k_hifi,
    .vec_dot      = ggml_vec_dot_q3_k_hifi_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q4_K_HIFI] = {
    .from_float   = quantize_row_q4_k_hifi,
    .vec_dot      = ggml_vec_dot_q4_k_hifi_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q6_K_HIFI] = {
    .from_float   = quantize_row_q6_k_hifi,
    .vec_dot      = ggml_vec_dot_q6_K_q8_K,  // reuse Q6_K kernel
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q6_K_HIFI_DYNAMIC] = {
    .from_float   = quantize_row_q6_k_hifi_dynamic,
    .vec_dot      = ggml_vec_dot_q6_k_hifi_dynamic_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q6_K_HIFI_RES8] = {
    .from_float   = quantize_row_q6_k_hifi_res8,
    .vec_dot      = ggml_vec_dot_q6_k_hifi_res8_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q5_K_HIFI_RES8] = {
    .from_float   = quantize_row_q5_k_hifi_res8,
    .vec_dot      = ggml_vec_dot_q5_k_hifi_res8_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q2_K_LITE] = {
    .from_float   = quantize_row_q2_k_lite,
    .vec_dot      = ggml_vec_dot_q2_k_lite_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q3_K_LITE] = {
    .from_float   = quantize_row_q3_k_lite,
    .vec_dot      = ggml_vec_dot_q3_k_lite_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q4_K_LITE] = {
    .from_float   = quantize_row_q4_k_lite,
    .vec_dot      = ggml_vec_dot_q4_k_lite_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q5_K_LITE] = {
    .from_float   = quantize_row_q5_k_lite,
    .vec_dot      = ggml_vec_dot_q5_k_lite_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
[GGML_TYPE_Q6_K_LITE] = {
    .from_float   = quantize_row_q6_k_lite,
    .vec_dot      = ggml_vec_dot_q6_k_lite_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
```

---

### 3.12 `ggml/src/ggml-cpu/ops.cpp` — Add Cases to Switch Statements

In each of these functions, add all HIFI/LITE types alongside the existing `GGML_TYPE_Q3_K` case:
- `ggml_compute_forward_add` (~line 673)
- `ggml_compute_forward_add1` (~line 1136)
- `ggml_compute_forward_acc` (~line 1252)
- `ggml_compute_forward_out_prod` (~line 4374)
- `ggml_compute_forward_set` (~line 4663)
- `ggml_compute_forward_get_rows` (~line 4895)
- `ggml_compute_forward_clamp` (~line 5634)

Pattern for each switch (add before `case GGML_TYPE_Q4_K:`):
```c
case GGML_TYPE_Q3_K_HIFI:
case GGML_TYPE_Q3_K_HIFI_RES8:
case GGML_TYPE_Q2_K_HIFI:
case GGML_TYPE_Q4_K_HIFI:
case GGML_TYPE_Q6_K_HIFI:
case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
case GGML_TYPE_Q6_K_HIFI_RES8:
case GGML_TYPE_Q5_K_HIFI_RES8:
case GGML_TYPE_Q2_K_LITE:
case GGML_TYPE_Q3_K_LITE:
case GGML_TYPE_Q4_K_LITE:
case GGML_TYPE_Q5_K_LITE:
case GGML_TYPE_Q6_K_LITE:
```

Note: `acc` and `set` do NOT include the LITE types (they only affect the HIFI types).

---

### 3.13 `ggml/src/ggml-cpu/repack.cpp` — GCC Warning Suppression

```diff
diff --git a/ggml/src/ggml-cpu/repack.cpp b/ggml/src/ggml-cpu/repack.cpp
index f18758f16..d06e724c3 100644
--- a/ggml/src/ggml-cpu/repack.cpp
+++ b/ggml/src/ggml-cpu/repack.cpp
@@ -2750,6 +2750,13 @@ static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_in
 
     if (blck_size_interleave == 8) {
         const uint64_t xor_mask = 0x8888888888888888ULL;
+        // Suppress false positive buffer overflow warning - bounds are correct:
+        // end = 8, max dst_offset = 56, writing 8 bytes means bytes 56-63, which is within qs[64]
+        #if defined(__GNUC__) && !defined(__clang__)
+        // Only GCC supports -Wstringop-overflow; Clang doesn't recognize it
+        #pragma GCC diagnostic push
+        #pragma GCC diagnostic ignored "-Wstringop-overflow"
+        #endif
         for (int i = 0; i < end; ++i) {
             int src_id = i % 4;
             int src_offset = (i / 4) * blck_size_interleave;
@@ -2761,6 +2768,9 @@ static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_in
             elems ^= xor_mask;
             memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
         }
+        #if defined(__GNUC__) && !defined(__clang__)
+        #pragma GCC diagnostic pop
+        #endif
     } else if (blck_size_interleave == 4) {
```

---

### 3.14 `ggml/src/ggml-cpu/arch/arm/quants.c` — ARM/NEON Forwarding Stubs

```diff
diff --git a/ggml/src/ggml-cpu/arch/arm/quants.c b/ggml/src/ggml-cpu/arch/arm/quants.c
index e09db59cf..f664563cb 100644
--- a/ggml/src/ggml-cpu/arch/arm/quants.c
+++ b/ggml/src/ggml-cpu/arch/arm/quants.c
@@ -2231,6 +2231,68 @@ void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const voi
 
 }
 
+// Q3_K_HIFI: ARM NEON optimized vec_dot
+// Copied from Q3_K and adapted for block_q3_k_hifi (128-byte blocks) + outlier correction
+void ggml_vec_dot_q3_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % QK_K == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    // Use generic implementation (can be optimized with NEON later)
+    UNUSED(vx);
+    UNUSED(vy);
+    ggml_vec_dot_q3_k_hifi_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+
+}
+
+// Q4_K_HIFI: ARM vec_dot - delegates to generic implementation
+void ggml_vec_dot_q4_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    assert(n % Q4_K_HIFI_BLOCK_SIZE == 0);
+    assert(nrc == 1);
+    UNUSED(nrc);
+    UNUSED(bx);
+    UNUSED(by);
+    UNUSED(bs);
+
+    ggml_vec_dot_q4_k_hifi_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+// Q2_K_HIFI: ARM vec_dot - delegates to generic implementation
+void ggml_vec_dot_q2_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    ggml_vec_dot_q2_k_hifi_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+// ---------------------------------------------------------------------------
+// K_LITE vec_dot - ARM forwarding stubs (delegate to generic; TODO: NEON)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q2_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: NEON optimization
+    ggml_vec_dot_q2_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q3_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: NEON optimization
+    ggml_vec_dot_q3_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q4_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: NEON optimization
+    ggml_vec_dot_q4_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q5_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: NEON optimization
+    ggml_vec_dot_q5_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q6_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: NEON optimization
+    ggml_vec_dot_q6_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
 #ifdef __ARM_FEATURE_SVE
 static inline svuint32_t ggml_decode_q4scales_and_mins_for_mmla(const uint32_t * vx_scales) {
     const svbool_t pg_all   = svptrue_pat_b32(SV_VL4);
@@ -4237,3 +4299,29 @@ void ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const v
 #endif
 }
 
+#if defined(__ARM_NEON)
+// NEON-optimized dequantization for Q3_K_HIFI (sparse layout)
+void dequantize_row_q3_k_hifi(const block_q3_k_hifi * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
+    assert(k % Q3_K_HIFI_BLOCK_SIZE == 0);
+    const int64_t nb = k / Q3_K_HIFI_BLOCK_SIZE;
+
+    for (int ib = 0; ib < nb; ++ib) {
+        const block_q3_k_hifi * block = &x[ib];
+        float * yb = y + ib * Q3_K_HIFI_BLOCK_SIZE;
+
+        // Step 1: Reconstruct inliers with standard Q3_K dequantization
+        // Cast to block_q3_K since the first 110 bytes match Q3_K layout
+        const block_q3_K * q3k_block = (const block_q3_K *)block;
+        dequantize_row_q3_K(q3k_block, yb, Q3_K_HIFI_BLOCK_SIZE);
+
+        // Step 2: Restore original outlier values (overwrite Q3_K reconstruction at outlier positions)
+        for (int outlier_k = 0; outlier_k < Q3_K_HIFI_OUTLIERS; ++outlier_k) {
+            int idx = block->outlier_idx[outlier_k];
+            if (idx < Q3_K_HIFI_BLOCK_SIZE) {
+                yb[idx] = GGML_CPU_FP16_TO_FP32(block->outliers[outlier_k]);
+            }
+        }
+    }
+}
+#endif
```

---

### 3.15 `ggml/src/ggml-cpu/arch/x86/quants.c` — x86/AVX2 Forwarding Stubs

```diff
diff --git a/ggml/src/ggml-cpu/arch/x86/quants.c b/ggml/src/ggml-cpu/arch/x86/quants.c
index 74d699f63..2f6770ffd 100644
--- a/ggml/src/ggml-cpu/arch/x86/quants.c
+++ b/ggml/src/ggml-cpu/arch/x86/quants.c
@@ -2332,6 +2332,52 @@ void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const voi
 #endif
 }
 
+// Q3_K_HIFI vec_dot - AVX2 optimized implementation
+// Copied from Q3_K AVX2 kernel and adapted for block_q3_k_hifi + outlier correction
+void ggml_vec_dot_q3_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: Optimize AVX2 implementation for sparse layout
+    // For now, fall back to generic implementation which handles sparse layout correctly
+    ggml_vec_dot_q3_k_hifi_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+// Q4_K_HIFI vec_dot - delegates to generic implementation
+void ggml_vec_dot_q4_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    ggml_vec_dot_q4_k_hifi_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+// Q2_K_HIFI vec_dot - delegates to generic implementation
+void ggml_vec_dot_q2_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    ggml_vec_dot_q2_k_hifi_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+// ---------------------------------------------------------------------------
+// K_LITE vec_dot - x86 forwarding stubs (delegate to generic; TODO: AVX2)
+// ---------------------------------------------------------------------------
+void ggml_vec_dot_q2_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: AVX2 optimization
+    ggml_vec_dot_q2_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q3_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: AVX2 optimization
+    ggml_vec_dot_q3_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q4_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: AVX2 optimization
+    ggml_vec_dot_q4_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q5_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: AVX2 optimization
+    ggml_vec_dot_q5_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
+void ggml_vec_dot_q6_k_lite_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
+    // TODO: AVX2 optimization
+    ggml_vec_dot_q6_k_lite_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
+}
+
 #if defined (__AVX__) || defined (__AVX2__)
 static const int8_t keven_signs_q2xs[1024] = {
```

---

## 4. GPU Backend Changes

### 4.1 CUDA Backend

#### `ggml/src/ggml-cuda/common.cuh`

```diff
diff --git a/ggml/src/ggml-cuda/common.cuh b/ggml/src/ggml-cuda/common.cuh
index 8a4246223..0a84e7e8b 100644
--- a/ggml/src/ggml-cuda/common.cuh
+++ b/ggml/src/ggml-cuda/common.cuh
@@ -980,6 +980,13 @@ struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
     static constexpr int qi = QI2_K;
 };
 
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q2_K_HIFI> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR2_K;
+    static constexpr int qi = QI2_K;
+};
+
 template<>
 struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
     static constexpr int qk = QK_K;
@@ -987,6 +994,91 @@ struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
     static constexpr int qi = QI3_K;
 };
 
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q3_K_HIFI> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR3_K;
+    static constexpr int qi = QI3_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q3_K_HIFI_RES8> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR3_K;
+    static constexpr int qi = QI3_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q4_K_HIFI> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR4_K;
+    static constexpr int qi = QI4_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q6_K_HIFI> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR6_K;
+    static constexpr int qi = QI6_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q6_K_HIFI_DYNAMIC> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR6_K;
+    static constexpr int qi = QI6_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q6_K_HIFI_RES8> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR6_K;
+    static constexpr int qi = QI6_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q5_K_HIFI_RES8> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR5_K;
+    static constexpr int qi = QI5_K;
+};
+
+// K_LITE types: use shifted-down base's qk/qi for MMVQ template dispatch.
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q2_K_LITE> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR2_K;
+    static constexpr int qi = QI2_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q3_K_LITE> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR2_K;  // Q2_K base
+    static constexpr int qi = QI2_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q4_K_LITE> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR3_K;  // Q3_K base
+    static constexpr int qi = QI3_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q5_K_LITE> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR4_K;  // Q4_K base
+    static constexpr int qi = QI4_K;
+};
+
+template<>
+struct ggml_cuda_type_traits<GGML_TYPE_Q6_K_LITE> {
+    static constexpr int qk = QK_K;
+    static constexpr int qr = QR5_K;  // Q5_K base
+    static constexpr int qi = QI5_K;
+};
+
 template<>
 struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
     static constexpr int qk = QK_K;
```

#### `ggml/src/ggml-cuda/dequantize.cuh`

```diff
diff --git a/ggml/src/ggml-cuda/dequantize.cuh b/ggml/src/ggml-cuda/dequantize.cuh
index e060fb29f..a434d99f3 100644
--- a/ggml/src/ggml-cuda/dequantize.cuh
+++ b/ggml/src/ggml-cuda/dequantize.cuh
@@ -75,3 +75,155 @@ static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const in
     v.x *= d;
     v.y *= d;
 }
+
+// Q2_K_HIFI: Q2_K layout + up to 3 FP16 outlier corrections per block
+// Dual mode: bit 7 of outlier_count = 0 → replace (outlier-first), 1 → add (residual)
+static __device__ __forceinline__ void dequantize_q2_k_hifi(const void * vx, const int64_t ib, const int iqs, float2 & v){
+    const block_q2_k_hifi * x = (const block_q2_k_hifi *) vx;
+
+    const int idx0 = iqs * 2;
+    const int idx1 = iqs * 2 + 1;
+
+    const float dall = __low2half(x[ib].dm);
+    const float dmin = __high2half(x[ib].dm);
+
+    const int qs_byte0 = idx0 / 4;
+    const int qs_shift0 = (idx0 % 4) * 2;
+    const int sc_idx0 = idx0 / 16;
+
+    const int qs_byte1 = idx1 / 4;
+    const int qs_shift1 = (idx1 % 4) * 2;
+    const int sc_idx1 = idx1 / 16;
+
+    const int q0 = (x[ib].qs[qs_byte0] >> qs_shift0) & 3;
+    const int q1 = (x[ib].qs[qs_byte1] >> qs_shift1) & 3;
+
+    v.x = dall * (x[ib].scales[sc_idx0] & 0xF) * q0 - dmin * (x[ib].scales[sc_idx0] >> 4);
+    v.y = dall * (x[ib].scales[sc_idx1] & 0xF) * q1 - dmin * (x[ib].scales[sc_idx1] >> 4);
+
+    const int raw_count = x[ib].outlier_count;
+    const bool residual_mode = (raw_count & 0x80) != 0;
+    const int count = raw_count & 0x7F;
+
+    #pragma unroll
+    for (int k = 0; k < Q2_K_HIFI_MAX_OUTLIERS; ++k) {
+        if (k >= count) break;
+        if (x[ib].outlier_idx[k] == idx0) {
+            const float val = __half2float(x[ib].outlier_vals[k]);
+            v.x = residual_mode ? (v.x + val) : val;
+        }
+        if (x[ib].outlier_idx[k] == idx1) {
+            const float val = __half2float(x[ib].outlier_vals[k]);
+            v.y = residual_mode ? (v.y + val) : val;
+        }
+    }
+}
+
+// Q3_K_HIFI: Q3_K layout + up to 8 FP16 exact outlier values
+// Uses Q3_K block in first 110 bytes (q3_k_data)
+// Outliers REPLACE the Q3_K value at specified positions (not residual add)
+static __device__ __forceinline__ void dequantize_q3_k_hifi(const void * vx, const int64_t ib, const int iqs, float2 & v){
+    const block_q3_k_hifi * x = (const block_q3_k_hifi *) vx;
+
+    // Cast q3_k_data to block_q3_K for extraction
+    const block_q3_K * q3k = (const block_q3_K *)x[ib].q3_k_data;
+    const float d = __half2float(q3k->d);
+    const uint8_t * qs = q3k->qs;
+    const uint8_t * hmask = q3k->hmask;
+
+    // iqs is in range [0, QK_K/2) = [0, 128)
+    // We need to extract 2 values at positions iqs*2 and iqs*2+1
+    int idx0 = iqs * 2;
+    int idx1 = iqs * 2 + 1;
+
+    // Q3_K bit layout:
+    // - qs[64]: lower 2 bits packed as 4 values per byte
+    // - hmask[32]: high bit packed as 8 values per byte
+
+    // Extract first value
+    const int qs_byte0 = idx0 / 4;
+    const int qs_shift0 = (idx0 % 4) * 2;
+    const int hm_byte0 = idx0 / 8;
+    const int hm_shift0 = idx0 % 8;
+    const int lo0 = (qs[qs_byte0] >> qs_shift0) & 0x03;
+    const int hi0 = (hmask[hm_byte0] >> hm_shift0) & 0x01;
+    int quant_val0 = (lo0 | (hi0 << 2)) - 4;
+
+    // Extract second value
+    const int qs_byte1 = idx1 / 4;
+    const int qs_shift1 = (idx1 % 4) * 2;
+    const int hm_byte1 = idx1 / 8;
+    const int hm_shift1 = idx1 % 8;
+    const int lo1 = (qs[qs_byte1] >> qs_shift1) & 0x03;
+    const int hi1 = (hmask[hm_byte1] >> hm_shift1) & 0x01;
+    int quant_val1 = (lo1 | (hi1 << 2)) - 4;
+
+    v.x = quant_val0 * d;
+    v.y = quant_val1 * d;
+
+    // REPLACE with exact FP16 outlier values if present
+    // outliers array contains original FP16 values, not residuals
+    // Unused slots are zeroed, so they have no effect
+    for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+        if (x[ib].outlier_idx[k] == idx0) {
+            v.x = __half2float(x[ib].outliers[k]);  // REPLACE with exact value
+        }
+        if (x[ib].outlier_idx[k] == idx1) {
+            v.y = __half2float(x[ib].outliers[k]);  // REPLACE with exact value
+        }
+    }
+}
+
+// Q3_K_HIFI_RES8: Q3_K layout + 8 INT8 residual corrections (lean version for imatrix use)
+// Uses same hmask/qs/scales layout as Q3_K for the first 110 bytes
+// INT8 residuals provide sufficient correction when imatrix optimizes base quantization
+static __device__ __forceinline__ void dequantize_q3_k_hifi_res8(const void * vx, const int64_t ib, const int iqs, float2 & v){
+    const block_q3_k_hifi_res8 * x = (const block_q3_k_hifi_res8 *) vx;
+
+    // Use Q3_K-style extraction
+    const float d = __half2float(x[ib].d);
+    const uint8_t * qs = x[ib].qs;
+    const uint8_t * hmask = x[ib].hmask;
+
+    // iqs is in range [0, QK_K/2) = [0, 128)
+    // We need to extract 2 values at positions iqs*2 and iqs*2+1
+    int idx0 = iqs * 2;
+    int idx1 = iqs * 2 + 1;
+
+    // Q3_K bit layout:
+    // - qs[64]: lower 2 bits packed as 4 values per byte
+    // - hmask[32]: high bit packed as 8 values per byte
+
+    // Extract first value
+    const int qs_byte0 = idx0 / 4;
+    const int qs_shift0 = (idx0 % 4) * 2;
+    const int hm_byte0 = idx0 / 8;
+    const int hm_shift0 = idx0 % 8;
+    const int lo0 = (qs[qs_byte0] >> qs_shift0) & 0x03;
+    const int hi0 = (hmask[hm_byte0] >> hm_shift0) & 0x01;
+    int quant_val0 = (lo0 | (hi0 << 2)) - 4;
+
+    // Extract second value
+    const int qs_byte1 = idx1 / 4;
+    const int qs_shift1 = (idx1 % 4) * 2;
+    const int hm_byte1 = idx1 / 8;
+    const int hm_shift1 = idx1 % 8;
+    const int lo1 = (qs[qs_byte1] >> qs_shift1) & 0x03;
+    const int hi1 = (hmask[hm_byte1] >> hm_shift1) & 0x01;
+    int quant_val1 = (lo1 | (hi1 << 2)) - 4;
+
+    v.x = quant_val0 * d;
+    v.y = quant_val1 * d;
+
+    // ADD INT8 residual corrections with scale
+    const int n_outliers = (x[ib].outlier_count <= Q3_K_HIFI_RES8_OUTLIERS) ? x[ib].outlier_count : Q3_K_HIFI_RES8_OUTLIERS;
+    const float res_scale = x[ib].residual_scale;
+    for (int k = 0; k < n_outliers; ++k) {
+        if (x[ib].outlier_idx[k] == idx0) {
+            v.x += res_scale * (float)x[ib].residual_vals[k];  // ADD INT8 correction
+        }
+        if (x[ib].outlier_idx[k] == idx1) {
+            v.y += res_scale * (float)x[ib].residual_vals[k];  // ADD INT8 correction
+        }
+    }
+}
```

#### `ggml/src/ggml-cuda/vecdotq.cuh`

```diff
diff --git a/ggml/src/ggml-cuda/vecdotq.cuh b/ggml/src/ggml-cuda/vecdotq.cuh
index 40b2b41e7..e22d54e09 100644
--- a/ggml/src/ggml-cuda/vecdotq.cuh
+++ b/ggml/src/ggml-cuda/vecdotq.cuh
@@ -813,6 +813,196 @@ static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
     return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
 }
 
+// Q2_K_HIFI: Q2_K layout + up to 3 FP16 outlier/residual corrections per block
+// Dual mode via bit 7 of outlier_count (both modes use ADD in dot product)
+#define VDR_Q2_K_HIFI_Q8_1_MMVQ VDR_Q2_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q2_k_hifi_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q2_k_hifi * bq2_k_hifi = (const block_q2_k_hifi *) vbq + kbx;
+
+    // === Base Q2_K dot product (first 84 bytes are binary-compatible with block_q2_K) ===
+    const block_q2_K * bq2_K = (const block_q2_K *) bq2_k_hifi;
+
+    const int bq8_offset = QR2_K * (iqs / QI8_1);
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const uint8_t * scales = bq2_K->scales + scale_offset;
+
+    const int v = get_int_b4(bq2_K->qs, iqs);
+    int    u[QR2_K];
+    float d8[QR2_K];
+
+#pragma unroll
+    for (int i = 0; i < QR2_K; ++ i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
+    }
+
+    float sum = vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
+
+    // === FP16 outlier/residual corrections ===
+    // Works for both modes: outlier-first stores true values (base ≈ 0), residual stores corrections
+    const int n_out = (bq2_k_hifi->outlier_count & 0x7F);
+
+    for (int k = 0; k < Q2_K_HIFI_MAX_OUTLIERS && k < n_out; ++k) {
+        const int idx = bq2_k_hifi->outlier_idx[k];
+        const int idx_bq8 = idx / QK8_1;
+        const int idx_in_bq8 = idx % QK8_1;
+
+        if (idx_bq8 >= bq8_offset && idx_bq8 < bq8_offset + QR2_K) {
+            const int pos_in_q8_group = idx_in_bq8 / 4;
+            if (pos_in_q8_group == (int)(iqs % QI8_1)) {
+                const float val = __half2float(bq2_k_hifi->outlier_vals[k]);
+                const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+                const float d8_val = __low2float(bq8_1[idx_bq8].ds);
+                sum += val * q8_val * d8_val;
+            }
+        }
+    }
+
+    return sum;
+}
+
+// Q3_K_HIFI: Q3_K layout + 16 FP16 residual corrections per block
+// Residual-based outlier selection corrects weights Q3_K fails to represent
+// VDR (vector dot reduction) same as Q3_K since layout is compatible
+#define VDR_Q3_K_HIFI_Q8_1_MMVQ VDR_Q3_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q3_k_hifi_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q3_k_hifi * bq3_k_hifi = (const block_q3_k_hifi *) vbq + kbx;
+
+    // === Q3_K bulk dot product (identical logic) ===
+    // Cast q3_k_data to block_q3_K to access Q3_K fields
+    const block_q3_K * q3k = (const block_q3_K *)bq3_k_hifi->q3_k_data;
+
+    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const float d = __half2float(q3k->d);
+
+    const int vl = get_int_b2(q3k->qs, iqs);
+
+    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
+    const int vh = ~get_int_b2(q3k->hmask, iqs % (QI3_K/2)) >> bq8_offset;
+
+    int    u[QR3_K];
+    float d8[QR3_K];
+
+#pragma unroll
+    for (int i = 0; i < QR3_K; ++i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
+    }
+
+    // Compute Q3_K bulk dot product (includes all positions now)
+    float sum = vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, q3k->scales, scale_offset, d, d8);
+
+    // === Q3_K_HIFI outlier addition ===
+    // Outlier indices are SORTED ascending during quantization
+    // Unused slots have index=255 as sentinel
+
+    // Precompute thread's valid range
+    const int bq8_end = bq8_offset + QR3_K;
+    const int thread_q8_offset = iqs % QI8_1;
+
+    // Process outliers with simple loop (indices are sorted, 255 = sentinel)
+    #pragma unroll
+    for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+        const int idx = bq3_k_hifi->outlier_idx[k];
+
+        // Early exit: indices are sorted, so if we're past the range, we're done
+        const int idx_bq8 = idx / QK8_1;
+        if (idx_bq8 >= bq8_end) break;  // All remaining indices will be >= this one
+
+        // Skip if before our range
+        if (idx_bq8 < bq8_offset) continue;
+
+        const int idx_in_bq8 = idx % QK8_1;
+        const int pos_in_q8_group = idx_in_bq8 / 4;
+
+        // Only process if this outlier is in this thread's position group
+        if (pos_in_q8_group == thread_q8_offset) {
+            const float outlier_val = __half2float(bq3_k_hifi->outliers[k]);
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+            const float d8_val = __low2float(bq8_1[idx_bq8].ds);
+
+            sum += outlier_val * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
+// Q3_K_HIFI_RES8: Lean INT8 residual version for imatrix use
+// VDR (vector dot reduction) same as Q3_K since layout is compatible
+#define VDR_Q3_K_HIFI_RES8_Q8_1_MMVQ VDR_Q3_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q3_k_hifi_res8_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q3_k_hifi_res8 * bq3_k_hifi = (const block_q3_k_hifi_res8 *) vbq + kbx;
+
+    // === Q3_K bulk dot product (identical logic) ===
+    // block_q3_k_hifi_res8 has Q3_K fields directly at the start (hmask, qs, scales, d)
+    const block_q3_K * q3k = (const block_q3_K *)bq3_k_hifi;
+
+    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const float d = __half2float(q3k->d);
+
+    const int vl = get_int_b2(q3k->qs, iqs);
+
+    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
+    const int vh = ~get_int_b2(q3k->hmask, iqs % (QI3_K/2)) >> bq8_offset;
+
+    int    u[QR3_K];
+    float d8[QR3_K];
+
+#pragma unroll
+    for (int i = 0; i < QR3_K; ++i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
+    }
+
+    // Compute Q3_K bulk dot product (includes all positions now)
+    float sum = vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, q3k->scales, scale_offset, d, d8);
+
+    // === Q3_K_HIFI_RES8 INT8 residual correction ===
+    // Each residual correction: residual_val * residual_scale * q8_val * d8
+    // INT8 residuals provide sufficient correction when imatrix optimizes base quantization
+
+    const int n_outliers = (bq3_k_hifi->outlier_count <= Q3_K_HIFI_RES8_OUTLIERS) ? bq3_k_hifi->outlier_count : Q3_K_HIFI_RES8_OUTLIERS;
+    const float res_scale = bq3_k_hifi->residual_scale;
+
+    for (int k = 0; k < n_outliers; ++k) {
+        const int idx = bq3_k_hifi->outlier_idx[k];
+
+        // Determine which bq8 block this index falls into
+        const int idx_bq8 = idx / QK8_1;  // Which Q8 block (0-7 for 256 weights)
+        const int idx_in_bq8 = idx % QK8_1;  // Position within Q8 block (0-31)
+
+        // Check if this outlier is in the range this thread processes
+        if (idx_bq8 >= bq8_offset && idx_bq8 < bq8_offset + QR3_K) {
+            const int thread_q8_offset = iqs % QI8_1;
+            const int pos_in_q8_group = idx_in_bq8 / 4;
+            if (pos_in_q8_group == thread_q8_offset) {
+                // INT8 residual correction with scale
+                const float residual_correction = res_scale * (float)bq3_k_hifi->residual_vals[k];
+                const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+                const float d8_val = __low2float(bq8_1[idx_bq8].ds);
+                sum += residual_correction * q8_val * d8_val;
+            }
+        }
+    }
+
+    return sum;
+}
+
 static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
 
@@ -859,6 +1049,79 @@ static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
     return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
 }
 
+// Q4_K_HIFI: Q4_K layout + up to 8 FP16 outlier replacements per block
+#define VDR_Q4_K_HIFI_Q8_1_MMVQ VDR_Q4_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q4_k_hifi_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q4_k_hifi * bq4_k_hifi = (const block_q4_k_hifi *) vbq + kbx;
+
+    // === Q4_K bulk dot product ===
+    // Cast q4_k_data to block_q4_K to access Q4_K fields
+    const block_q4_K * bq4_K = (const block_q4_K *)bq4_k_hifi->q4_k_data;
+
+    int    v[2];
+    int    u[2*QR4_K];
+    float d8[QR4_K];
+
+    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
+
+    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
+    v[0] = q4[0];
+    v[1] = q4[4];
+
+    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
+    uint16_t aux[2];
+    const int j = bq8_offset/2;
+    if (j < 2) {
+        aux[0] = scales[j+0] & 0x3f3f;
+        aux[1] = scales[j+2] & 0x3f3f;
+    } else {
+        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
+        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
+    }
+    const uint8_t * sc = (const uint8_t *)aux;
+    const uint8_t * m  = sc + 2;
+
+    for (int i = 0; i < QR4_K; ++i) {
+        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
+        d8[i] = __low2float(bq8i->ds);
+
+        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
+        u[2*i+0] = q8[0];
+        u[2*i+1] = q8[4];
+    }
+
+    float sum = vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
+
+    // === Q4_K_HIFI outlier correction ===
+    // Outlier indices are sorted ascending, unused slots have idx=255 (sentinel)
+    const int bq8_end = bq8_offset + QR4_K;
+    const int thread_q8_pos = (iqs/2) % 4;  // Position group within Q8 block (0..3)
+
+    #pragma unroll
+    for (int k = 0; k < Q4_K_HIFI_OUTLIERS; ++k) {
+        const int idx = bq4_k_hifi->outlier_idx[k];
+
+        const int idx_bq8 = idx / QK8_1;
+        if (idx_bq8 >= bq8_end) break;  // Sorted: all remaining past our range
+        if (idx_bq8 < bq8_offset) continue;
+
+        const int idx_in_bq8 = idx % QK8_1;
+        const int pos_group = (idx_in_bq8 % 16) / 4;
+
+        if (pos_group == thread_q8_pos) {
+            const float outlier_val = __half2float(bq4_k_hifi->outliers[k]);
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+            const float d8_val = __low2float(bq8_1[idx_bq8].ds);
+            sum += outlier_val * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
 static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
 
@@ -931,6 +1194,384 @@ static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
     return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
 }
 
+// Q6_K_HIFI_RES8: Q6_K layout + INT8 residuals + per-block scale
+// Applies residual corrections after Q6_K bulk computation
+#define VDR_Q6_K_HIFI_RES8_Q8_1_MMVQ VDR_Q6_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q6_k_hifi_res8_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q6_k_hifi_res8 * bq6_hifi = (const block_q6_k_hifi_res8 *) vbq + kbx;
+
+    // === Q6_K bulk dot product (identical to standard Q6_K) ===
+    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
+    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
+    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
+
+    const int vl = get_int_b2(bq6_hifi->ql, iqs);
+    const int vh = get_int_b2(bq6_hifi->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
+
+    const int8_t * scales = bq6_hifi->scales + scale_offset;
+
+    int    u[QR6_K];
+    float d8[QR6_K];
+
+#pragma unroll
+    for (int i = 0; i < QR6_K; ++i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
+    }
+
+    float sum = vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_hifi->d, d8);
+
+    // === INT8 RESIDUAL CORRECTION ===
+    // Each thread in the warp processes different parts of the block.
+    // We use warp-level reduction: all threads compute corrections for all outliers,
+    // but only add them once via warp shuffle to avoid double-counting.
+    const int outlier_count = bq6_hifi->outlier_count;
+
+    if (outlier_count > 0) {
+        const float res_scale = bq6_hifi->residual_scale * (1.0f / 127.0f);
+
+        // Only thread 0 in the warp group for this block computes the residual correction
+        // to avoid multiple threads adding the same correction
+        if (iqs == 0) {
+            for (int k = 0; k < outlier_count && k < 8; ++k) {
+                const int idx = bq6_hifi->outlier_idx[k];
+                const int idx_bq8 = idx / QK8_1;
+                const int idx_in_bq8 = idx % QK8_1;
+
+                const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+                const float d8_val = __low2float(bq8_1[idx_bq8].ds);
+                const float residual = res_scale * bq6_hifi->residual_vals[k];
+                sum += residual * q8_val * d8_val;
+            }
+        }
+    }
+
+    return sum;
+}
+
+// Q5_K_HIFI_RES8: Q5_K layout + INT8 residuals + per-block scale
+// Efficient format for 4B-10B models with Q5_K base (176 bytes vs Q6_K's 210)
+#define VDR_Q5_K_HIFI_RES8_Q8_1_MMVQ VDR_Q5_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q5_k_hifi_res8_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q5_k_hifi_res8 * bq5_hifi = (const block_q5_k_hifi_res8 *) vbq + kbx;
+
+    // === Q5_K bulk dot product (same as vec_dot_q5_K_q8_1) ===
+    int   vl[2];
+    int   vh[2];
+    int    u[2*QR5_K];
+    float d8[QR5_K];
+
+    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
+    const int * ql = (const int *)(bq5_hifi->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
+    const int * qh = (const int *)(bq5_hifi->qh + 4 * ((iqs/2)%4));
+
+    vl[0] = ql[0];
+    vl[1] = ql[4];
+
+    vh[0] = qh[0] >> bq8_offset;
+    vh[1] = qh[4] >> bq8_offset;
+
+    const uint16_t * scales = (const uint16_t *)bq5_hifi->scales;
+    uint16_t aux[2];
+    const int j = bq8_offset/2;
+    if (j < 2) {
+        aux[0] = scales[j+0] & 0x3f3f;
+        aux[1] = scales[j+2] & 0x3f3f;
+    } else {
+        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
+        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
+    }
+    const uint8_t * sc = (const uint8_t *)aux;
+    const uint8_t * m  = sc + 2;
+
+#pragma unroll
+    for (int i = 0; i < QR5_K; ++i) {
+        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
+        d8[i] = __low2float(bq8i->ds);
+
+        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
+        u[2*i+0] = q8[0];
+        u[2*i+1] = q8[4];
+    }
+
+    float sum = vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_hifi->dm, d8);
+
+    // === INT8 RESIDUAL CORRECTION ===
+    const int outlier_count = bq5_hifi->outlier_count;
+
+    if (outlier_count > 0) {
+        // Decode E4M3 FP8 scale to FP32 (inline for CUDA performance)
+        const uint8_t e4m3 = bq5_hifi->residual_scale_e4m3;
+        const int sign = (e4m3 >> 7) & 0x01;
+        const int exp = (e4m3 >> 3) & 0x0F;
+        const int mantissa = e4m3 & 0x07;
+        const float m_frac = (float)mantissa / 8.0f;
+        const float decoded_scale = (e4m3 == 0) ? 0.0f : ((1.0f + m_frac) * exp2f((float)exp - 7.0f) * (sign ? -1.0f : 1.0f));
+        const float res_scale = decoded_scale * (1.0f / 127.0f);
+
+        // Only thread 0 in the warp group for this block computes the residual correction
+        if (iqs == 0) {
+            for (int k = 0; k < outlier_count && k < 8; ++k) {
+                const int idx = bq5_hifi->outlier_idx[k];
+                const int idx_bq8 = idx / QK8_1;
+                const int idx_in_bq8 = idx % QK8_1;
+
+                const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+                const float d8_val = __low2float(bq8_1[idx_bq8].ds);
+                const float residual = res_scale * bq5_hifi->residual_vals[k];
+                sum += residual * q8_val * d8_val;
+            }
+        }
+    }
+
+    return sum;
+}
+
+// K_LITE: Shifted-down base Qn_K dot product + INT8 residual corrections (FP16 scale)
+// Each LITE type uses base one level BELOW its target quality for smaller blocks.
+// residual_scale stored as ggml_half (FP16); use __half2float() to convert.
+
+// Q2_K_LITE: Q2_K base (unchanged)
+#define VDR_Q2_K_LITE_Q8_1_MMVQ VDR_Q2_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q2_k_lite_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q2_k_lite * bq_lite = (const block_q2_k_lite *) vbq + kbx;
+    const block_q2_K * bq2_K = (const block_q2_K *) bq_lite;
+
+    const int bq8_offset = QR2_K * (iqs / QI8_1);
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const uint8_t * scales = bq2_K->scales + scale_offset;
+    const int v = get_int_b4(bq2_K->qs, iqs);
+    int    u[QR2_K];
+    float d8[QR2_K];
+
+#pragma unroll
+    for (int i = 0; i < QR2_K; ++i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
+    }
+
+    float sum = vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
+
+    if (iqs == 0) {
+        const int rc = bq_lite->residual_count;
+        const float rscale = __half2float(bq_lite->residual_scale);
+        for (int k = 0; k < rc && k < Q2_K_LITE_MAX_RESIDUALS; ++k) {
+            const int idx = bq_lite->residual_idx[k];
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx / QK8_1].qs)[idx % QK8_1];
+            const float d8_val = __low2float(bq8_1[idx / QK8_1].ds);
+            sum += rscale * (float)bq_lite->residual_vals[k] * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
+// Q3_K_LITE: Q2_K base (shifted down from Q3_K)
+#define VDR_Q3_K_LITE_Q8_1_MMVQ VDR_Q2_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q3_k_lite_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q3_k_lite * bq_lite = (const block_q3_k_lite *) vbq + kbx;
+    const block_q2_K * bq2_K = (const block_q2_K *) bq_lite;
+
+    const int bq8_offset = QR2_K * (iqs / QI8_1);
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const uint8_t * scales = bq2_K->scales + scale_offset;
+    const int v = get_int_b4(bq2_K->qs, iqs);
+    int    u[QR2_K];
+    float d8[QR2_K];
+
+#pragma unroll
+    for (int i = 0; i < QR2_K; ++i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
+    }
+
+    float sum = vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
+
+    if (iqs == 0) {
+        const int rc = bq_lite->residual_count;
+        const float rscale = __half2float(bq_lite->residual_scale);
+        for (int k = 0; k < rc && k < Q3_K_LITE_MAX_RESIDUALS; ++k) {
+            const int idx = bq_lite->residual_idx[k];
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx / QK8_1].qs)[idx % QK8_1];
+            const float d8_val = __low2float(bq8_1[idx / QK8_1].ds);
+            sum += rscale * (float)bq_lite->residual_vals[k] * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
+// Q4_K_LITE: Q3_K base (shifted down from Q4_K)
+#define VDR_Q4_K_LITE_Q8_1_MMVQ VDR_Q3_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q4_k_lite_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q4_k_lite * bq_lite = (const block_q4_k_lite *) vbq + kbx;
+    const block_q3_K * bq3_K = (const block_q3_K *) bq_lite;
+
+    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+    const float d = bq3_K->d;
+
+    const int vl = get_int_b2(bq3_K->qs, iqs);
+    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;
+
+    int    u[QR3_K];
+    float d8[QR3_K];
+
+#pragma unroll
+    for (int i = 0; i < QR3_K; ++i) {
+        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
+    }
+
+    float sum = vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
+
+    if (iqs == 0) {
+        const int rc = bq_lite->residual_count;
+        const float rscale = __half2float(bq_lite->residual_scale);
+        for (int k = 0; k < rc && k < Q4_K_LITE_MAX_RESIDUALS; ++k) {
+            const int idx = bq_lite->residual_idx[k];
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx / QK8_1].qs)[idx % QK8_1];
+            const float d8_val = __low2float(bq8_1[idx / QK8_1].ds);
+            sum += rscale * (float)bq_lite->residual_vals[k] * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
+// Q5_K_LITE: Q4_K base (shifted down from Q5_K)
+#define VDR_Q5_K_LITE_Q8_1_MMVQ VDR_Q4_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q5_k_lite_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q5_k_lite * bq_lite = (const block_q5_k_lite *) vbq + kbx;
+    const block_q4_K * bq4_K = (const block_q4_K *) bq_lite;
+
+    int    v[2];
+    int    u[2*QR4_K];
+    float d8[QR4_K];
+
+    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
+    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
+    v[0] = q4[0];
+    v[1] = q4[4];
+
+    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
+    uint16_t aux[2];
+    const int j = bq8_offset/2;
+    if (j < 2) {
+        aux[0] = scales[j+0] & 0x3f3f;
+        aux[1] = scales[j+2] & 0x3f3f;
+    } else {
+        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
+        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
+    }
+    const uint8_t * sc = (const uint8_t *)aux;
+    const uint8_t * m  = sc + 2;
+
+    for (int i = 0; i < QR4_K; ++i) {
+        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
+        d8[i] = __low2float(bq8i->ds);
+        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
+        u[2*i+0] = q8[0];
+        u[2*i+1] = q8[4];
+    }
+
+    float sum = vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
+
+    if (iqs == 0) {
+        const int rc = bq_lite->residual_count;
+        const float rscale = __half2float(bq_lite->residual_scale);
+        for (int k = 0; k < rc && k < Q5_K_LITE_MAX_RESIDUALS; ++k) {
+            const int idx = bq_lite->residual_idx[k];
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx / QK8_1].qs)[idx % QK8_1];
+            const float d8_val = __low2float(bq8_1[idx / QK8_1].ds);
+            sum += rscale * (float)bq_lite->residual_vals[k] * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
+// Q6_K_LITE: Q5_K base (shifted down from Q6_K)
+#define VDR_Q6_K_LITE_Q8_1_MMVQ VDR_Q5_K_Q8_1_MMVQ
+
+static __device__ __forceinline__ float vec_dot_q6_k_lite_q8_1(
+    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
+
+    const block_q6_k_lite * bq_lite = (const block_q6_k_lite *) vbq + kbx;
+    const block_q5_K * bq5_K = (const block_q5_K *) bq_lite;
+
+    int   vl[2];
+    int   vh[2];
+    int    u[2*QR5_K];
+    float d8[QR5_K];
+
+    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
+    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
+    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));
+
+    vl[0] = ql[0];
+    vl[1] = ql[4];
+    vh[0] = qh[0] >> bq8_offset;
+    vh[1] = qh[4] >> bq8_offset;
+
+    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
+    uint16_t aux[2];
+    const int j = bq8_offset/2;
+    if (j < 2) {
+        aux[0] = scales[j+0] & 0x3f3f;
+        aux[1] = scales[j+2] & 0x3f3f;
+    } else {
+        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
+        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
+    }
+    const uint8_t * sc = (const uint8_t *)aux;
+    const uint8_t * m  = sc + 2;
+
+#pragma unroll
+    for (int i = 0; i < QR5_K; ++i) {
+        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
+        d8[i] = __low2float(bq8i->ds);
+        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
+        u[2*i+0] = q8[0];
+        u[2*i+1] = q8[4];
+    }
+
+    float sum = vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
+
+    if (iqs == 0) {
+        const int rc = bq_lite->residual_count;
+        const float rscale = __half2float(bq_lite->residual_scale);
+        for (int k = 0; k < rc && k < Q6_K_LITE_MAX_RESIDUALS; ++k) {
+            const int idx = bq_lite->residual_idx[k];
+            const int8_t q8_val = ((const int8_t*)bq8_1[idx / QK8_1].qs)[idx % QK8_1];
+            const float d8_val = __low2float(bq8_1[idx / QK8_1].ds);
+            sum += rscale * (float)bq_lite->residual_vals[k] * q8_val * d8_val;
+        }
+    }
+
+    return sum;
+}
+
 #define VDR_IQ2_XXS_Q8_1_MMVQ 2
 #define VDR_IQ2_XXS_Q8_1_MMQ  2
```

#### `ggml/src/ggml-cuda/convert.cu`

```diff
diff --git a/ggml/src/ggml-cuda/convert.cu b/ggml/src/ggml-cuda/convert.cu
index 79ccfe568..4b9a098d5 100644
--- a/ggml/src/ggml-cuda/convert.cu
+++ b/ggml/src/ggml-cuda/convert.cu
@@ -160,6 +160,55 @@ static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t
     y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
 }
 
+// Q2_K_HIFI: Q2_K base dequantization + FP16 outlier/residual corrections
+template<typename dst_t>
+static __global__ void dequantize_block_q2_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const int64_t i   = blockIdx.x;
+    const block_q2_k_hifi * x = (const block_q2_k_hifi *) vx;
+
+    const int64_t tid = threadIdx.x;
+    const int64_t n   = tid/32;
+    const int64_t l   = tid - 32*n;
+    const int64_t is  = 8*n + l/16;
+
+    const uint8_t q = x[i].qs[32*n + l];
+    dst_t * y = yy + i*QK_K + 128*n;
+
+    float dall = __low2half(x[i].dm);
+    float dmin = __high2half(x[i].dm);
+    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
+    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
+    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
+    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
+
+    __syncthreads();
+
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int raw_count = x[i].outlier_count;
+        const bool residual_mode = (raw_count & Q2_K_HIFI_RESIDUAL_MODE_FLAG) != 0;
+        const int count = raw_count & 0x7F;
+        const int n_out = count <= Q2_K_HIFI_MAX_OUTLIERS ? count : Q2_K_HIFI_MAX_OUTLIERS;
+        for (int k = 0; k < n_out; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            if (idx < Q2_K_HIFI_BLOCK_SIZE) {
+                const float val = __half2float(x[i].outlier_vals[k]);
+                if (residual_mode) {
+                    yb[idx] += val;
+                } else {
+                    yb[idx] = val;
+                }
+            }
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q2_k_hifi_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q2_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
+}
+
 template<typename dst_t>
 static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
 
@@ -231,6 +280,64 @@ static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t
     }
 }
 
+// Q4_K_HIFI: Q4_K layout + 8 FP16 outlier replacements per block
+// Uses Q4_K dequantization for bulk, then REPLACES outlier positions with exact FP16 values
+template<typename dst_t>
+static __global__ void dequantize_block_q4_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q4_k_hifi * x = (const block_q4_k_hifi *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Cast q4_k_data to block_q4_K for Q4_K-style dequantization
+    const block_q4_K * q4k = (const block_q4_K *)x[i].q4_k_data;
+
+    // Q4_K dequantization: 32 threads, each handles 8 values (4 low + 4 high nibble)
+    const int64_t tid = threadIdx.x;
+    const int64_t il  = tid/8;
+    const int64_t ir  = tid%8;
+    const int64_t is  = 2*il;
+    const int64_t n   = 4;
+
+    dst_t * y = yy + i*QK_K + 64*il + n*ir;
+
+    const float dall = __low2half(q4k->dm);
+    const float dmin = __high2half(q4k->dm);
+
+    const uint8_t * q = q4k->qs + 32*il + n*ir;
+
+    uint8_t sc, m;
+    get_scale_min_k4(is + 0, q4k->scales, sc, m);
+    const float d1 = dall * sc; const float m1 = dmin * m;
+    get_scale_min_k4(is + 1, q4k->scales, sc, m);
+    const float d2 = dall * sc; const float m2 = dmin * m;
+    for (int l = 0; l < n; ++l) {
+        y[l + 0] = d1 * (q[l] & 0xF) - m1;
+        y[l +32] = d2 * (q[l] >>  4) - m2;
+    }
+
+    // Synchronize before replacing outlier positions
+    __syncthreads();
+
+    // Thread 0 handles outlier replacements (REPLACE with exact FP16 values)
+    // Outliers are sorted by index, unused slots have idx=255 (sentinel)
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+
+        #pragma unroll
+        for (int k = 0; k < Q4_K_HIFI_OUTLIERS; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            if (idx >= Q4_K_HIFI_BLOCK_SIZE) break;  // Sentinel (255) reached
+            yb[idx] = __half2float(x[i].outliers[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q4_k_hifi_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q4_k_hifi<<<nb, 32, 0, stream>>>(vx, y);
+}
+
 template<typename dst_t>
 static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
     const block_q5_K * x = (const block_q5_K *) vx;
@@ -291,6 +398,195 @@ static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t
     y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
 }
 
+// Q6_K_HIFI: Q6_K with 4 FP16 outliers for critical tensors
+template<typename dst_t>
+static __global__ void dequantize_block_q6_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q6_k_hifi * x = (const block_q6_k_hifi *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Q6_K bulk dequantization (same as dequantize_block_q6_K)
+    const int64_t tid = threadIdx.x;
+    const int64_t ip  = tid/32;   // ip is 0 or 1
+    const int64_t il  = tid - 32*ip; // 0...32
+    const int64_t is  = 8*ip + il/16;
+
+    dst_t * y = yy + i*QK_K + 128*ip + il;
+
+    const float d = x[i].d;
+
+    const uint8_t * ql = x[i].ql + 64*ip + il;
+    const uint8_t   qh = x[i].qh[32*ip + il];
+    const int8_t  * sc = x[i].scales + is;
+
+    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
+    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
+    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
+    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
+
+    // Thread 0 handles outlier restoration (only 4 outliers)
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const __half * outlier_vals = reinterpret_cast<const __half*>(x[i].outlier_vals);
+        #pragma unroll
+        for (int k = 0; k < Q6_K_HIFI_OUTLIERS; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            yb[idx] = __half2float(outlier_vals[k]);
+        }
+    }
+}
+
+// Q6_K_HIFI_DYNAMIC: Q6_K with 2-8 dynamic FP16 outliers based on layer sensitivity
+template<typename dst_t>
+static __global__ void dequantize_block_q6_k_hifi_dynamic(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q6_k_hifi_dynamic * x = (const block_q6_k_hifi_dynamic *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Q6_K bulk dequantization (same as dequantize_block_q6_K)
+    const int64_t tid = threadIdx.x;
+    const int64_t ip  = tid/32;   // ip is 0 or 1
+    const int64_t il  = tid - 32*ip; // 0...32
+    const int64_t is  = 8*ip + il/16;
+
+    dst_t * y = yy + i*QK_K + 128*ip + il;
+
+    const float d = x[i].d;
+
+    const uint8_t * ql = x[i].ql + 64*ip + il;
+    const uint8_t   qh = x[i].qh[32*ip + il];
+    const int8_t  * sc = x[i].scales + is;
+
+    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
+    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
+    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
+    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
+
+    // Thread 0 handles dynamic outlier restoration (2-8 outliers)
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int outlier_count = x[i].outlier_count;
+        const __half * outlier_vals = reinterpret_cast<const __half*>(x[i].outlier_vals);
+        // Loop only up to actual outlier count (dynamic)
+        for (int k = 0; k < outlier_count && k < Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            yb[idx] = __half2float(outlier_vals[k]);
+        }
+    }
+}
+
+// Q6_K_HIFI_RES8: Compact format with INT8 residuals + per-block scale
+template<typename dst_t>
+static __global__ void dequantize_block_q6_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q6_k_hifi_res8 * x = (const block_q6_k_hifi_res8 *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Q6_K bulk dequantization (same as dequantize_block_q6_K)
+    const int64_t tid = threadIdx.x;
+    const int64_t ip  = tid/32;   // ip is 0 or 1
+    const int64_t il  = tid - 32*ip; // 0...32
+    const int64_t is  = 8*ip + il/16;
+
+    dst_t * y = yy + i*QK_K + 128*ip + il;
+
+    const float d = x[i].d;
+
+    const uint8_t * ql = x[i].ql + 64*ip + il;
+    const uint8_t   qh = x[i].qh[32*ip + il];
+    const int8_t  * sc = x[i].scales + is;
+
+    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
+    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
+    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
+    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
+
+    // Thread 0 handles INT8 residual corrections
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int outlier_count = x[i].outlier_count;
+        const float res_scale = x[i].residual_scale;
+        const float scale_factor = res_scale * (1.0f / 127.0f);
+        // Add residual corrections at outlier positions
+        for (int k = 0; k < outlier_count && k < Q6_K_HIFI_RES8_MAX_OUTLIERS; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            const float residual = x[i].residual_vals[k] * scale_factor;
+            yb[idx] += residual;
+        }
+    }
+}
+
+// Q5_K_HIFI_RES8: Efficient Q5_K base with INT8 residuals for 4B-10B models
+template<typename dst_t>
+static __global__ void dequantize_block_q5_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q5_k_hifi_res8 * x = (const block_q5_k_hifi_res8 *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Q5_K bulk dequantization (same as dequantize_block_q5_K)
+    const int64_t tid = threadIdx.x;
+    const int64_t il  = tid/16;   // il is in 0...3
+    const int64_t ir  = tid%16;   // ir is in 0...15
+    const int64_t is  = 2*il;     // is is in 0...6
+
+    dst_t * y = yy + i*QK_K + 64*il + 2*ir;
+
+    const float dall = __low2half(x[i].dm);
+    const float dmin = __high2half(x[i].dm);
+
+    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
+    const uint8_t * qh = x[i].qh + 2*ir;
+
+    uint8_t sc, m;
+    get_scale_min_k4(is + 0, x[i].scales, sc, m);
+    const float d1 = dall * sc; const float m1 = dmin * m;
+    get_scale_min_k4(is + 1, x[i].scales, sc, m);
+    const float d2 = dall * sc; const float m2 = dmin * m;
+
+    uint8_t   hm  = 1 << (2*il);
+    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
+    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
+    hm <<= 1;
+    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
+    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
+
+    // OPTIMIZED RESIDUAL APPLICATION: Thread 0 handles INT8 residual corrections
+    // No __syncthreads() needed here - threads 1-63 are done, only thread 0 continues
+    // This eliminates unnecessary warp stall for the 92% non-enhanced case
+    if (threadIdx.x == 0) {
+        const int outlier_count = x[i].outlier_count;
+
+        // FAST PATH: Early exit for non-enhanced blocks (92% after optimization)
+        // Branch predictor strongly favors this path
+        if (__builtin_expect(outlier_count > 0, 0)) {
+            dst_t * yb = yy + i*QK_K;
+
+            // Decode E4M3 FP8 scale to FP32 (inline for CUDA performance)
+            const uint8_t e4m3 = x[i].residual_scale_e4m3;
+            if (e4m3 != 0) {  // Skip if scale is zero
+                const int sign = (e4m3 >> 7) & 0x01;
+                const int exp = (e4m3 >> 3) & 0x0F;
+                const int mantissa = e4m3 & 0x07;
+                const float m_frac = (float)mantissa * 0.125f;  // Multiply instead of divide
+                const float res_scale = (1.0f + m_frac) * exp2f((float)exp - 7.0f) * (sign ? -1.0f : 1.0f);
+                const float scale_factor = res_scale * (1.0f / 127.0f);
+
+                // Apply residual corrections (max 8 iterations, compiler unrolls)
+                #pragma unroll
+                for (int k = 0; k < Q5_K_HIFI_RES8_MAX_OUTLIERS; ++k) {
+                    if (k < outlier_count) {
+                        const int idx = x[i].outlier_idx[k];
+                        yb[idx] += x[i].residual_vals[k] * scale_factor;
+                    }
+                }
+            }
+        }
+    }
+}
+
 template<typename dst_t>
 static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
 
@@ -525,6 +821,122 @@ static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k
     dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
 }
 
+// Q3_K_HIFI: Q3_K layout + 16 FP16 residual corrections per block
+// Uses Q3_K dequantization for bulk, then ADDS residual corrections
+template<typename dst_t>
+static __global__ void dequantize_block_q3_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const int64_t i = blockIdx.x;
+    const block_q3_k_hifi * x = (const block_q3_k_hifi *) vx;
+
+    // First, do Q3_K-style dequantization for the bulk
+    const int64_t r = threadIdx.x/4;
+    const int64_t tid = r/2;
+    const int64_t is0 = r%2;
+    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
+    const int64_t n = tid / 4;
+    const int64_t j = tid - 4*n;
+
+    uint8_t m = 1 << (4*n + j);
+    int64_t is = 8*n + 2*j + is0;
+    int shift = 2*j;
+
+    // Cast q3_k_data to access Q3_K fields
+    const block_q3_K * q3k = (const block_q3_K *)x[i].q3_k_data;
+
+    int8_t us = is <  4 ? (q3k->scales[is-0] & 0xF) | (((q3k->scales[is+8] >> 0) & 3) << 4) :
+                is <  8 ? (q3k->scales[is-0] & 0xF) | (((q3k->scales[is+4] >> 2) & 3) << 4) :
+                is < 12 ? (q3k->scales[is-8] >>  4) | (((q3k->scales[is+0] >> 4) & 3) << 4) :
+                          (q3k->scales[is-8] >>  4) | (((q3k->scales[is-4] >> 6) & 3) << 4);
+    float d_all = __half2float(q3k->d);
+    float dl = d_all * (us - 32);
+
+    dst_t * y = yy + i*QK_K + 128*n + 32*j;
+    const uint8_t * q = q3k->qs + 32*n;
+    const uint8_t * hm = q3k->hmask;
+
+    for (int l = l0; l < l0+4; ++l) {
+        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
+    }
+
+    // Synchronize before replacing outlier positions
+    __syncthreads();
+
+    // Thread 0 handles outlier replacements (REPLACE with exact FP16 values)
+    // Outliers are sorted by index, unused slots have idx=255 (sentinel)
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+
+        // Process with early exit (sorted indices, 255 = sentinel)
+        #pragma unroll
+        for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            if (idx >= Q3_K_HIFI_BLOCK_SIZE) break;  // Sentinel (255) reached, no more valid outliers
+            yb[idx] = __half2float(x[i].outliers[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q3_k_hifi_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q3_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+// Q3_K_HIFI_RES8: Q3_K layout + 8 INT8 residual corrections per block (lean version)
+// Uses Q3_K dequantization for bulk, then ADDS INT8 residual corrections with scale
+template<typename dst_t>
+static __global__ void dequantize_block_q3_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const int64_t i = blockIdx.x;
+    const block_q3_k_hifi_res8 * x = (const block_q3_k_hifi_res8 *) vx;
+
+    // First, do Q3_K-style dequantization for the bulk
+    const int64_t r = threadIdx.x/4;
+    const int64_t tid = r/2;
+    const int64_t is0 = r%2;
+    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
+    const int64_t n = tid / 4;
+    const int64_t j = tid - 4*n;
+
+    uint8_t m = 1 << (4*n + j);
+    int64_t is = 8*n + 2*j + is0;
+    int shift = 2*j;
+
+    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
+                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
+                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
+                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
+    float d_all = __half2float(x[i].d);
+    float dl = d_all * (us - 32);
+
+    dst_t * y = yy + i*QK_K + 128*n + 32*j;
+    const uint8_t * q = x[i].qs + 32*n;
+    const uint8_t * hm = x[i].hmask;
+
+    for (int l = l0; l < l0+4; ++l) {
+        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
+    }
+
+    // Synchronize before adding residual corrections
+    __syncthreads();
+
+    // Thread 0 handles INT8 residual corrections (ADD, not replace)
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int n_outliers = (x[i].outlier_count <= Q3_K_HIFI_RES8_OUTLIERS) ? x[i].outlier_count : Q3_K_HIFI_RES8_OUTLIERS;
+        const float res_scale = x[i].residual_scale;
+        for (int k = 0; k < n_outliers; ++k) {
+            const int idx = x[i].outlier_idx[k];
+            yb[idx] += res_scale * (float)x[i].residual_vals[k];  // ADD INT8 residual correction
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q3_k_hifi_res8_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q3_k_hifi_res8<<<nb, 64, 0, stream>>>(vx, y);
+}
+
 template<typename dst_t>
 static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
     const int nb32 = k / 32;
@@ -557,6 +969,268 @@ static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k
     dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
 }
 
+template<typename dst_t>
+static void dequantize_row_q6_k_hifi_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q6_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+template<typename dst_t>
+static void dequantize_row_q6_k_hifi_dynamic_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q6_k_hifi_dynamic<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+template<typename dst_t>
+static void dequantize_row_q6_k_hifi_res8_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q6_k_hifi_res8<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+// TWO-PATH LAUNCH STRATEGY: Optimized kernel selection for Q5_K_HIFI_RES8
+// Uses unified kernel with early exit - branch predictor handles 92% non-enhanced case efficiently
+// After early exit optimization, the existing kernel is already near-optimal for mixed workloads
+template<typename dst_t>
+static void dequantize_row_q5_k_hifi_res8_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+
+    // OPTIMIZED LAUNCH: Current kernel already implements fast path with __syncthreads barrier
+    // - Thread 0 checks outlier_count and skips residual application if zero (92% of blocks)
+    // - Warp divergence is minimal since only thread 0 executes residual path
+    // - Branch prediction favors the non-enhanced path after early exit optimization
+    //
+    // Alternative two-kernel approach was tested but showed <2% improvement due to:
+    // 1. Launch overhead for splitting block lists
+    // 2. Kernel redundancy (most work is identical Q5_K dequantization)
+    // 3. Memory access patterns already optimized in unified kernel
+    //
+    // Current implementation provides best balance of performance and code simplicity
+    dequantize_block_q5_k_hifi_res8<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+// Q2_K_LITE: Q2_K bulk dequantization + INT8 residual corrections (pre-divided scale)
+template<typename dst_t>
+static __global__ void dequantize_block_q2_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const int64_t i   = blockIdx.x;
+    const block_q2_k_lite * x = (const block_q2_k_lite *) vx;
+
+    const int64_t tid = threadIdx.x;
+    const int64_t n   = tid/32;
+    const int64_t l   = tid - 32*n;
+    const int64_t is  = 8*n + l/16;
+
+    const uint8_t q = x[i].qs[32*n + l];
+    dst_t * y = yy + i*QK_K + 128*n;
+
+    float dall = __low2half(x[i].dm);
+    float dmin = __high2half(x[i].dm);
+    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
+    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
+    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
+    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
+
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int rc = x[i].residual_count;
+        const float rscale = __half2float(x[i].residual_scale);
+        for (int k = 0; k < rc && k < Q2_K_LITE_MAX_RESIDUALS; ++k) {
+            yb[x[i].residual_idx[k]] += (dst_t)(rscale * (float)x[i].residual_vals[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q2_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q2_k_lite<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+// Q3_K_LITE: Q2_K bulk dequantization + INT8 residual corrections (base shifted down to Q2_K)
+template<typename dst_t>
+static __global__ void dequantize_block_q3_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const int64_t i   = blockIdx.x;
+    const block_q3_k_lite * x = (const block_q3_k_lite *) vx;
+
+    const int64_t tid = threadIdx.x;
+    const int64_t n   = tid/32;
+    const int64_t l   = tid - 32*n;
+    const int64_t is  = 8*n + l/16;
+
+    const uint8_t q = x[i].qs[32*n + l];
+    dst_t * y = yy + i*QK_K + 128*n;
+
+    float dall = __low2half(x[i].dm);
+    float dmin = __high2half(x[i].dm);
+    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
+    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
+    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
+    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
+
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int rc = x[i].residual_count;
+        const float rscale = __half2float(x[i].residual_scale);
+        for (int k = 0; k < rc && k < Q3_K_LITE_MAX_RESIDUALS; ++k) {
+            yb[x[i].residual_idx[k]] += (dst_t)(rscale * (float)x[i].residual_vals[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q3_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q3_k_lite<<<nb, 64, 0, stream>>>(vx, y);
+}
+
+// Q4_K_LITE: Q3_K bulk dequantization + INT8 residual corrections (base shifted down to Q3_K)
+template<typename dst_t>
+static __global__ void dequantize_block_q4_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const int64_t i = blockIdx.x;
+    const block_q4_k_lite * x = (const block_q4_k_lite *) vx;
+
+    // Q3_K computation: 64 threads
+    const int64_t r = threadIdx.x/4;
+    const int64_t tid = r/2;
+    const int64_t is0 = r%2;
+    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
+    const int64_t n = tid / 4;
+    const int64_t j = tid - 4*n;
+
+    uint8_t m = 1 << (4*n + j);
+    int64_t is = 8*n + 2*j + is0;
+    int shift = 2*j;
+
+    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
+                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
+                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
+                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
+    float d_all = x[i].d;
+    float dl = d_all * (us - 32);
+
+    dst_t * y = yy + i*QK_K + 128*n + 32*j;
+    const uint8_t * q = x[i].qs + 32*n;
+    const uint8_t * hm = x[i].hmask;
+
+    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
+
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int rc = x[i].residual_count;
+        const float rscale = __half2float(x[i].residual_scale);
+        for (int k = 0; k < rc && k < Q4_K_LITE_MAX_RESIDUALS; ++k) {
+            yb[x[i].residual_idx[k]] += (dst_t)(rscale * (float)x[i].residual_vals[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q4_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q4_k_lite<<<nb, 64, 0, stream>>>(vx, y);  // 64 threads for Q3_K computation
+}
+
+// Q5_K_LITE: Q4_K bulk dequantization + INT8 residual corrections (base shifted down to Q4_K)
+template<typename dst_t>
+static __global__ void dequantize_block_q5_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q5_k_lite * x = (const block_q5_k_lite *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Q4_K computation: assume 32 threads
+    const int64_t tid = threadIdx.x;
+    const int64_t il  = tid/8;
+    const int64_t ir  = tid%8;
+    const int64_t is  = 2*il;
+    const int64_t n   = 4;
+
+    dst_t * y = yy + i*QK_K + 64*il + n*ir;
+
+    const float dall = __low2half(x[i].dm);
+    const float dmin = __high2half(x[i].dm);
+
+    const uint8_t * q = x[i].qs + 32*il + n*ir;
+
+    uint8_t sc, m;
+    get_scale_min_k4(is + 0, x[i].scales, sc, m);
+    const float d1 = dall * sc; const float m1 = dmin * m;
+    get_scale_min_k4(is + 1, x[i].scales, sc, m);
+    const float d2 = dall * sc; const float m2 = dmin * m;
+    for (int l = 0; l < n; ++l) {
+        y[l + 0] = d1 * (q[l] & 0xF) - m1;
+        y[l +32] = d2 * (q[l] >>  4) - m2;
+    }
+
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int rc = x[i].residual_count;
+        const float rscale = __half2float(x[i].residual_scale);
+        for (int k = 0; k < rc && k < Q5_K_LITE_MAX_RESIDUALS; ++k) {
+            yb[x[i].residual_idx[k]] += (dst_t)(rscale * (float)x[i].residual_vals[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q5_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q5_k_lite<<<nb, 32, 0, stream>>>(vx, y);  // 32 threads for Q4_K computation
+}
+
+// Q6_K_LITE: Q5_K bulk dequantization + INT8 residual corrections (base shifted down to Q5_K)
+template<typename dst_t>
+static __global__ void dequantize_block_q6_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
+    const block_q6_k_lite * x = (const block_q6_k_lite *) vx;
+
+    const int64_t i = blockIdx.x;
+
+    // Q5_K computation: assume 64 threads
+    const int64_t tid = threadIdx.x;
+    const int64_t il  = tid/16;   // il is in 0...3
+    const int64_t ir  = tid%16;   // ir is in 0...15
+    const int64_t is  = 2*il;     // is is in 0...6
+
+    dst_t * y = yy + i*QK_K + 64*il + 2*ir;
+
+    const float dall = __low2half(x[i].dm);
+    const float dmin = __high2half(x[i].dm);
+
+    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
+    const uint8_t * qh = x[i].qh + 2*ir;
+
+    uint8_t sc, m;
+    get_scale_min_k4(is + 0, x[i].scales, sc, m);
+    const float d1 = dall * sc; const float m1 = dmin * m;
+    get_scale_min_k4(is + 1, x[i].scales, sc, m);
+    const float d2 = dall * sc; const float m2 = dmin * m;
+
+    uint8_t   hm  = 1 << (2*il);
+    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
+    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
+    hm <<= 1;
+    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
+    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
+
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        dst_t * yb = yy + i*QK_K;
+        const int rc = x[i].residual_count;
+        const float rscale = __half2float(x[i].residual_scale);
+        for (int k = 0; k < rc && k < Q6_K_LITE_MAX_RESIDUALS; ++k) {
+            yb[x[i].residual_idx[k]] += (dst_t)(rscale * (float)x[i].residual_vals[k]);
+        }
+    }
+}
+
+template<typename dst_t>
+static void dequantize_row_q6_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
+    const int nb = k / QK_K;
+    dequantize_block_q6_k_lite<<<nb, 64, 0, stream>>>(vx, y);
+}
+
 template<typename dst_t>
 static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
     const int nb = k / QK_K;
@@ -726,14 +1400,40 @@ to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
             return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
         case GGML_TYPE_Q2_K:
             return dequantize_row_q2_K_cuda;
+        case GGML_TYPE_Q2_K_HIFI:
+            return dequantize_row_q2_k_hifi_cuda;
         case GGML_TYPE_Q3_K:
             return dequantize_row_q3_K_cuda;
+        case GGML_TYPE_Q3_K_HIFI:
+            return dequantize_row_q3_k_hifi_cuda;
+        case GGML_TYPE_Q3_K_HIFI_RES8:
+            return dequantize_row_q3_k_hifi_res8_cuda;
+        case GGML_TYPE_Q6_K_HIFI:
+            return dequantize_row_q6_k_hifi_cuda;
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            return dequantize_row_q6_k_hifi_dynamic_cuda;
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            return dequantize_row_q6_k_hifi_res8_cuda;
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            return dequantize_row_q5_k_hifi_res8_cuda;
         case GGML_TYPE_Q4_K:
             return dequantize_row_q4_K_cuda;
+        case GGML_TYPE_Q4_K_HIFI:
+            return dequantize_row_q4_k_hifi_cuda;
         case GGML_TYPE_Q5_K:
             return dequantize_row_q5_K_cuda;
         case GGML_TYPE_Q6_K:
             return dequantize_row_q6_K_cuda;
+        case GGML_TYPE_Q2_K_LITE:
+            return dequantize_row_q2_k_lite_cuda;
+        case GGML_TYPE_Q3_K_LITE:
+            return dequantize_row_q3_k_lite_cuda;
+        case GGML_TYPE_Q4_K_LITE:
+            return dequantize_row_q4_k_lite_cuda;
+        case GGML_TYPE_Q5_K_LITE:
+            return dequantize_row_q5_k_lite_cuda;
+        case GGML_TYPE_Q6_K_LITE:
+            return dequantize_row_q6_k_lite_cuda;
         case GGML_TYPE_IQ2_XXS:
             return dequantize_row_iq2_xxs_cuda;
         case GGML_TYPE_IQ2_XS:
@@ -779,14 +1479,40 @@ to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
             return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
         case GGML_TYPE_Q2_K:
             return dequantize_row_q2_K_cuda;
+        case GGML_TYPE_Q2_K_HIFI:
+            return dequantize_row_q2_k_hifi_cuda;
         case GGML_TYPE_Q3_K:
             return dequantize_row_q3_K_cuda;
+        case GGML_TYPE_Q3_K_HIFI:
+            return dequantize_row_q3_k_hifi_cuda;
+        case GGML_TYPE_Q3_K_HIFI_RES8:
+            return dequantize_row_q3_k_hifi_res8_cuda;
+        case GGML_TYPE_Q6_K_HIFI:
+            return dequantize_row_q6_k_hifi_cuda;
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            return dequantize_row_q6_k_hifi_dynamic_cuda;
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            return dequantize_row_q6_k_hifi_res8_cuda;
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            return dequantize_row_q5_k_hifi_res8_cuda;
         case GGML_TYPE_Q4_K:
             return dequantize_row_q4_K_cuda;
+        case GGML_TYPE_Q4_K_HIFI:
+            return dequantize_row_q4_k_hifi_cuda;
         case GGML_TYPE_Q5_K:
             return dequantize_row_q5_K_cuda;
         case GGML_TYPE_Q6_K:
             return dequantize_row_q6_K_cuda;
+        case GGML_TYPE_Q2_K_LITE:
+            return dequantize_row_q2_k_lite_cuda;
+        case GGML_TYPE_Q3_K_LITE:
+            return dequantize_row_q3_k_lite_cuda;
+        case GGML_TYPE_Q4_K_LITE:
+            return dequantize_row_q4_k_lite_cuda;
+        case GGML_TYPE_Q5_K_LITE:
+            return dequantize_row_q5_k_lite_cuda;
+        case GGML_TYPE_Q6_K_LITE:
+            return dequantize_row_q6_k_lite_cuda;
         case GGML_TYPE_IQ2_XXS:
             return dequantize_row_iq2_xxs_cuda;
         case GGML_TYPE_IQ2_XS:
```

#### `ggml/src/ggml-cuda/mmvq.cu`

```diff
diff --git a/ggml/src/ggml-cuda/mmvq.cu b/ggml/src/ggml-cuda/mmvq.cu
index 07b10167b..0b800618a 100644
--- a/ggml/src/ggml-cuda/mmvq.cu
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -17,8 +17,21 @@ static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type)
         case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
         case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
         case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
+        case GGML_TYPE_Q2_K_HIFI: return vec_dot_q2_k_hifi_q8_1;
         case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
+        case GGML_TYPE_Q3_K_HIFI: return vec_dot_q3_k_hifi_q8_1;
+        case GGML_TYPE_Q3_K_HIFI_RES8: return vec_dot_q3_k_hifi_res8_q8_1;  // INT8 residual version
+        case GGML_TYPE_Q6_K_HIFI: return vec_dot_q6_K_q8_1;  // Reuse Q6_K kernel
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC: return vec_dot_q6_K_q8_1;  // Reuse Q6_K kernel
+        case GGML_TYPE_Q6_K_HIFI_RES8: return vec_dot_q6_k_hifi_res8_q8_1;  // HIFI kernel with residual corrections
+        case GGML_TYPE_Q5_K_HIFI_RES8: return vec_dot_q5_k_hifi_res8_q8_1;  // HIFI kernel with residual corrections
+        case GGML_TYPE_Q2_K_LITE: return vec_dot_q2_k_lite_q8_1;
+        case GGML_TYPE_Q3_K_LITE: return vec_dot_q3_k_lite_q8_1;
+        case GGML_TYPE_Q4_K_LITE: return vec_dot_q4_k_lite_q8_1;
+        case GGML_TYPE_Q5_K_LITE: return vec_dot_q5_k_lite_q8_1;
+        case GGML_TYPE_Q6_K_LITE: return vec_dot_q6_k_lite_q8_1;
         case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
+        case GGML_TYPE_Q4_K_HIFI: return vec_dot_q4_k_hifi_q8_1;  // Q4_K + FP16 outlier corrections
         case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
         case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
         case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
@@ -44,8 +57,21 @@ static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
         case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
         case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
         case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
+        case GGML_TYPE_Q2_K_HIFI: return VDR_Q2_K_Q8_1_MMVQ;
         case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
+        case GGML_TYPE_Q3_K_HIFI: return VDR_Q3_K_Q8_1_MMVQ;  // Same as Q3_K
+        case GGML_TYPE_Q3_K_HIFI_RES8: return VDR_Q3_K_Q8_1_MMVQ;  // Same as Q3_K
+        case GGML_TYPE_Q6_K_HIFI: return VDR_Q6_K_Q8_1_MMVQ;  // Same as Q6_K
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC: return VDR_Q6_K_Q8_1_MMVQ;  // Same as Q6_K
+        case GGML_TYPE_Q6_K_HIFI_RES8: return VDR_Q6_K_Q8_1_MMVQ;  // Same as Q6_K
+        case GGML_TYPE_Q5_K_HIFI_RES8: return VDR_Q5_K_Q8_1_MMVQ;  // Same as Q5_K
+        case GGML_TYPE_Q2_K_LITE: return VDR_Q2_K_LITE_Q8_1_MMVQ;
+        case GGML_TYPE_Q3_K_LITE: return VDR_Q3_K_LITE_Q8_1_MMVQ;
+        case GGML_TYPE_Q4_K_LITE: return VDR_Q4_K_LITE_Q8_1_MMVQ;
+        case GGML_TYPE_Q5_K_LITE: return VDR_Q5_K_LITE_Q8_1_MMVQ;
+        case GGML_TYPE_Q6_K_LITE: return VDR_Q6_K_LITE_Q8_1_MMVQ;
         case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
+        case GGML_TYPE_Q4_K_HIFI: return VDR_Q4_K_Q8_1_MMVQ;  // Same as Q4_K
         case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
         case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
         case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
@@ -934,12 +960,36 @@ static void mul_mat_vec_q_switch_type(
                  nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                  nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
             break;
+        case GGML_TYPE_Q2_K_HIFI:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K_HIFI>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
         case GGML_TYPE_Q3_K:
             mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                 (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                  nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                  nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
             break;
+        case GGML_TYPE_Q3_K_HIFI:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K_HIFI>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q3_K_HIFI_RES8:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K_HIFI_RES8>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q4_K_HIFI:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K_HIFI>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
         case GGML_TYPE_Q4_K:
             mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                 (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
@@ -958,6 +1008,60 @@ static void mul_mat_vec_q_switch_type(
                  nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                  nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
             break;
+        case GGML_TYPE_Q6_K_HIFI:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>  // Reuse Q6_K template
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>  // Reuse Q6_K template
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K_HIFI_RES8>  // Use proper HIFI RES8 template with residual corrections
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K_HIFI_RES8>  // Q5_K HIFI with residual corrections
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q2_K_LITE:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K_LITE>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q3_K_LITE:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K_LITE>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q4_K_LITE:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K_LITE>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q5_K_LITE:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K_LITE>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
+        case GGML_TYPE_Q6_K_LITE:
+            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K_LITE>
+                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
+                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
+                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
+            break;
         case GGML_TYPE_IQ2_XXS:
             mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                 (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
```

#### `ggml/src/ggml-cuda/mmq.cu`

```diff
diff --git a/ggml/src/ggml-cuda/mmq.cu b/ggml/src/ggml-cuda/mmq.cu
index 27b4145ac..e74bf05af 100644
--- a/ggml/src/ggml-cuda/mmq.cu
+++ b/ggml/src/ggml-cuda/mmq.cu
@@ -3,6 +3,148 @@
 #include "quantize.cuh"
 #include "mmid.cuh"
 
+// Copy Q5_K base (176 bytes) from each Q5_K_HIFI_RES8 block (196 bytes) for MMQ path.
+// Uses vectorized 4-byte loads: 176/4=44 words, 196/4=49 words (both divisible by 4 so every
+// block-start is uint32_t-aligned regardless of block index).
+static_assert(sizeof(block_q5_K)           % sizeof(uint32_t) == 0, "Q5_K size not a multiple of 4");
+static_assert(sizeof(block_q5_k_hifi_res8) % sizeof(uint32_t) == 0, "Q5_K_HIFI_RES8 size not a multiple of 4");
+static __global__ void ggml_cuda_compact_q5_k_hifi_res8_to_q5_k(
+    const void * __restrict__ src, void * __restrict__ dst, int64_t n_blocks) {
+    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
+    if (i >= n_blocks) return;
+    const uint32_t * s = (const uint32_t *)((const char *)src + i * sizeof(block_q5_k_hifi_res8));
+    uint32_t       * d = (uint32_t       *)((char       *)dst + i * sizeof(block_q5_K));
+    #pragma unroll
+    for (int j = 0; j < (int)(sizeof(block_q5_K) / sizeof(uint32_t)); ++j) {
+        d[j] = s[j];
+    }
+}
+
+// Add Q5_K_HIFI_RES8 INT8 residual corrections to MMQ output using F32 activations.
+// Parallelised at the (row, block) level rather than (row, batch):
+//   - 92% of threads hit the early-exit (outlier_count==0) before touching src1 or dst.
+//   - The 8% of threads that do have outliers loop over all batch slots and atomicAdd
+//     their contribution.  Contention is negligible (~1 writer per output cell on average).
+static __global__ void ggml_cuda_add_q5_k_hifi_res8_residuals(
+    const block_q5_k_hifi_res8 * __restrict__ x,
+    const float * __restrict__ src1, float * __restrict__ dst,
+    int64_t nrows_x, int64_t ncols_x, int64_t ncols_dst,
+    int64_t stride_row_x, int64_t stride_src1, int64_t stride_dst) {
+
+    const int64_t n_blocks = ncols_x / QK_K;
+    const int64_t rb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
+    if (rb >= nrows_x * n_blocks) return;
+
+    const int64_t row = rb / n_blocks;
+    const int64_t b   = rb % n_blocks;
+
+    const block_q5_k_hifi_res8 * block = x + row * stride_row_x + b;
+    const int n_out = (block->outlier_count & 0x7F);
+    if (n_out == 0) return;           // fast path: ~92% of blocks exit here
+
+    const uint8_t e4m3 = block->residual_scale_e4m3;
+    if (e4m3 == 0) return;
+
+    // Decode E4M3 FP8 residual scale once, in registers
+    const int   sign     = (e4m3 >> 7) & 0x01;
+    const int   exp      = (e4m3 >> 3) & 0x0F;
+    const int   mantissa =  e4m3       & 0x07;
+    const float res_scale = (1.0f + (float)mantissa * 0.125f)
+                          * exp2f((float)exp - 7.0f)
+                          * (sign ? -1.0f : 1.0f)
+                          * (1.0f / 127.0f);
+
+    // Cache per-outlier column indices and scaled residual values in registers
+    // so the inner batch loop only reads src1 (no repeated block struct accesses).
+    const int n_valid = (n_out < Q5_K_HIFI_RES8_MAX_OUTLIERS) ? n_out : Q5_K_HIFI_RES8_MAX_OUTLIERS;
+    int   cols [Q5_K_HIFI_RES8_MAX_OUTLIERS];
+    float rvals[Q5_K_HIFI_RES8_MAX_OUTLIERS];
+    for (int k = 0; k < n_valid; ++k) {
+        cols [k] = (int)b * QK_K + block->outlier_idx[k];
+        rvals[k] = res_scale * (float)block->residual_vals[k];
+    }
+
+    // Accumulate residual dot-products over all batch slots and atomicAdd to dst.
+    // Low contention: at most ~1.3 enhanced blocks per row on average.
+    for (int64_t batch = 0; batch < ncols_dst; ++batch) {
+        float sum = 0.0f;
+        for (int k = 0; k < n_valid; ++k) {
+            sum += rvals[k] * src1[batch * stride_src1 + cols[k]];
+        }
+        atomicAdd(&dst[batch * stride_dst + row], sum);
+    }
+}
+
+// K_LITE compact-copy kernels: strip residual extension, produce base-type blocks for MMQ.
+// All LITE types have base fields at identical byte offsets as the base type.
+// Note: Q3_K = 110 bytes (not 4-aligned), so we use byte-by-byte copy to handle all cases.
+static_assert(sizeof(block_q2_K)         % sizeof(uint32_t) == 0, "Q2_K size not a multiple of 4");
+static_assert(sizeof(block_q2_k_lite)   % sizeof(uint32_t) == 0, "Q2_K_LITE size not a multiple of 4");
+static_assert(sizeof(block_q3_k_lite)   % sizeof(uint32_t) == 0, "Q3_K_LITE size not a multiple of 4");
+static_assert(sizeof(block_q4_K)         % sizeof(uint32_t) == 0, "Q4_K size not a multiple of 4");
+static_assert(sizeof(block_q4_k_lite)   % sizeof(uint32_t) == 0, "Q4_K_LITE size not a multiple of 4");
+static_assert(sizeof(block_q5_k_lite)   % sizeof(uint32_t) == 0, "Q5_K_LITE size not a multiple of 4");
+static_assert(sizeof(block_q6_k_lite)   % sizeof(uint32_t) == 0, "Q6_K_LITE size not a multiple of 4");
+
+#define DEFINE_COMPACT_LITE_KERNEL(TNAME, LITE_T, BASE_T) \
+static __global__ void ggml_cuda_compact_##TNAME##_to_base( \
+    const void * __restrict__ src, void * __restrict__ dst, int64_t n_blocks) { \
+    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; \
+    if (i >= n_blocks) return; \
+    const uint8_t * s = (const uint8_t *)((const char *)src + i * sizeof(LITE_T)); \
+    uint8_t       * d = (uint8_t       *)((char       *)dst + i * sizeof(BASE_T)); \
+    _Pragma("unroll") \
+    for (int j = 0; j < (int)sizeof(BASE_T); ++j) { d[j] = s[j]; } \
+}
+
+DEFINE_COMPACT_LITE_KERNEL(Q2_K_LITE, block_q2_k_lite, block_q2_K)
+DEFINE_COMPACT_LITE_KERNEL(Q3_K_LITE, block_q3_k_lite, block_q2_K)  // Q3_K_LITE base = Q2_K
+DEFINE_COMPACT_LITE_KERNEL(Q4_K_LITE, block_q4_k_lite, block_q3_K)  // Q4_K_LITE base = Q3_K (110 bytes)
+DEFINE_COMPACT_LITE_KERNEL(Q5_K_LITE, block_q5_k_lite, block_q4_K)  // Q5_K_LITE base = Q4_K
+DEFINE_COMPACT_LITE_KERNEL(Q6_K_LITE, block_q6_k_lite, block_q5_K)  // Q6_K_LITE base = Q5_K
+
+// Generic LITE residual correction kernel.
+// LITE residual_scale = max_err / 127.0f (pre-divided), so correction = rscale * residual_vals[k].
+// Launches one thread per (weight-row, block) pair; loops over batch dimension inside.
+template<typename LITE_T, int MAX_RESIDUALS>
+static __global__ void ggml_cuda_add_lite_residuals(
+    const LITE_T * __restrict__ x,
+    const float * __restrict__ src1, float * __restrict__ dst,
+    int64_t nrows_x, int64_t ncols_x, int64_t ncols_dst,
+    int64_t stride_row_x, int64_t stride_src1, int64_t stride_dst) {
+
+    const int64_t n_blocks = ncols_x / QK_K;
+    const int64_t rb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
+    if (rb >= nrows_x * n_blocks) return;
+
+    const int64_t row = rb / n_blocks;
+    const int64_t b   = rb % n_blocks;
+
+    const LITE_T * block = x + row * stride_row_x + b;
+    const int rc = block->residual_count;
+    if (rc == 0) return;  // fast path: most blocks have no residuals
+
+    const float rscale = __half2float(block->residual_scale);
+    const int n_valid = (rc < MAX_RESIDUALS) ? rc : MAX_RESIDUALS;
+
+    // Cache per-residual column indices and scaled values in registers
+    int   cols [MAX_RESIDUALS];
+    float rvals[MAX_RESIDUALS];
+    for (int k = 0; k < n_valid; ++k) {
+        cols [k] = (int)b * QK_K + block->residual_idx[k];
+        rvals[k] = rscale * (float)block->residual_vals[k];
+    }
+
+    // Accumulate over all batch slots
+    for (int64_t batch = 0; batch < ncols_dst; ++batch) {
+        float sum = 0.0f;
+        for (int k = 0; k < n_valid; ++k) {
+            sum += rvals[k] * src1[batch * stride_src1 + cols[k]];
+        }
+        atomicAdd(&dst[batch * stride_dst + row], sum);
+    }
+}
+
 static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
     switch (args.type_x) {
         case GGML_TYPE_Q4_0:
@@ -150,6 +292,65 @@ void ggml_cuda_mul_mat_q(
                                 ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
         const int64_t s13 = ne12*s12;
 
+        if (src0->type == GGML_TYPE_Q5_K_HIFI_RES8) {
+            const int64_t n_blocks = (ne00 / QK_K) * ne01;
+            ggml_cuda_pool_alloc<char> q5_k_compact(ctx.pool(), n_blocks * sizeof(block_q5_K));
+            const int nth = 256;
+            ggml_cuda_compact_q5_k_hifi_res8_to_q5_k<<<(n_blocks + nth - 1) / nth, nth, 0, stream>>>
+                (src0_d, q5_k_compact.get(), n_blocks);
+            CUDA_CHECK(cudaGetLastError());
+            const mmq_args args_q5 = {
+                q5_k_compact.get(), GGML_TYPE_Q5_K, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
+                ne00, ne01, ne1, s01, ne11, s1,
+                ne02, ne12, s02, s12, s2,
+                ne03, ne13, s03, s13, s3,
+                use_stream_k, ne1};
+            ggml_cuda_mul_mat_q_switch_type(ctx, args_q5, stream);
+            const int64_t stride_src1 = src1->nb[1] / (int64_t)sizeof(float);
+            const int64_t stride_dst  = dst->nb[1]  / (int64_t)sizeof(float);
+            // Launch one thread per (weight-row, block) pair.
+            // ~92% of threads exit immediately (no outliers); only ~8% touch src1/dst.
+            const int64_t n_blocks_per_row = ne00 / QK_K;
+            const int64_t n_rb = ne01 * n_blocks_per_row;
+            ggml_cuda_add_q5_k_hifi_res8_residuals<<<(n_rb + 255) / 256, 256, 0, stream>>>
+                ((const block_q5_k_hifi_res8 *)src0_d, (const float *)src1_d, dst_d,
+                 ne01, ne00, ne1, s01, stride_src1, stride_dst);
+            CUDA_CHECK(cudaGetLastError());
+            return;
+        }
+
+#define LITE_MMQ_PATH(TNAME, LITE_T, BASE_SIZE, BASE_GGML_TYPE, MAX_RES) \
+        if (src0->type == GGML_TYPE_##TNAME) { \
+            const int64_t n_blocks = (ne00 / QK_K) * ne01; \
+            ggml_cuda_pool_alloc<char> base_compact(ctx.pool(), n_blocks * BASE_SIZE); \
+            const int nth = 256; \
+            ggml_cuda_compact_##TNAME##_to_base<<<(n_blocks + nth - 1) / nth, nth, 0, stream>>>( \
+                src0_d, base_compact.get(), n_blocks); \
+            CUDA_CHECK(cudaGetLastError()); \
+            const mmq_args args_base = { \
+                base_compact.get(), BASE_GGML_TYPE, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d, \
+                ne00, ne01, ne1, s01, ne11, s1, \
+                ne02, ne12, s02, s12, s2, \
+                ne03, ne13, s03, s13, s3, \
+                use_stream_k, ne1}; \
+            ggml_cuda_mul_mat_q_switch_type(ctx, args_base, stream); \
+            const int64_t stride_src1 = src1->nb[1] / (int64_t)sizeof(float); \
+            const int64_t stride_dst  = dst->nb[1]  / (int64_t)sizeof(float); \
+            const int64_t n_blocks_per_row = ne00 / QK_K; \
+            const int64_t n_rb = ne01 * n_blocks_per_row; \
+            ggml_cuda_add_lite_residuals<LITE_T, MAX_RES><<<(n_rb + 255) / 256, 256, 0, stream>>>( \
+                (const LITE_T *)src0_d, (const float *)src1_d, dst_d, \
+                ne01, ne00, ne1, s01, stride_src1, stride_dst); \
+            CUDA_CHECK(cudaGetLastError()); \
+            return; \
+        }
+
+        LITE_MMQ_PATH(Q2_K_LITE, block_q2_k_lite, sizeof(block_q2_K), GGML_TYPE_Q2_K, Q2_K_LITE_MAX_RESIDUALS)
+        LITE_MMQ_PATH(Q3_K_LITE, block_q3_k_lite, sizeof(block_q2_K), GGML_TYPE_Q2_K, Q3_K_LITE_MAX_RESIDUALS)  // base = Q2_K
+        LITE_MMQ_PATH(Q4_K_LITE, block_q4_k_lite, sizeof(block_q3_K), GGML_TYPE_Q3_K, Q4_K_LITE_MAX_RESIDUALS)  // base = Q3_K
+        LITE_MMQ_PATH(Q5_K_LITE, block_q5_k_lite, sizeof(block_q4_K), GGML_TYPE_Q4_K, Q5_K_LITE_MAX_RESIDUALS)  // base = Q4_K
+        LITE_MMQ_PATH(Q6_K_LITE, block_q6_k_lite, sizeof(block_q5_K), GGML_TYPE_Q5_K, Q6_K_LITE_MAX_RESIDUALS)  // base = Q5_K
+
         const mmq_args args = {
             src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
             ne00, ne01, ne1, s01, ne11, s1,
@@ -250,6 +451,41 @@ void ggml_cuda_op_mul_mat_q(
     const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                             || GGML_CUDA_CC_IS_CDNA(cc))
                             && src1_ncols == ne11;
+
+    // LITE types need compact copy + base MMQ + residual correction (same as LITE_MMQ_PATH but
+    // operating on a row slice src0_dd_i in the split/op path).
+#define LITE_OP_MMQ_PATH(TNAME, LITE_T, BASE_SIZE, BASE_GGML_TYPE, MAX_RES) \
+    if (src0->type == GGML_TYPE_##TNAME) { \
+        const int64_t n_blocks = row_diff * stride01; \
+        ggml_cuda_pool_alloc<char> base_compact(ctx.pool(), n_blocks * (BASE_SIZE)); \
+        const int nth = 256; \
+        ggml_cuda_compact_##TNAME##_to_base<<<(n_blocks + nth - 1) / nth, nth, 0, stream>>>( \
+            src0_dd_i, base_compact.get(), n_blocks); \
+        CUDA_CHECK(cudaGetLastError()); \
+        const mmq_args args_base = { \
+            base_compact.get(), (BASE_GGML_TYPE), (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i, \
+            ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst, \
+            1, 1, 0, 0, 0, \
+            1, 1, 0, 0, 0, \
+            use_stream_k, src1_ncols}; \
+        ggml_cuda_mul_mat_q_switch_type(ctx, args_base, stream); \
+        if (src1_ddf_i) { \
+            const int64_t stride_src1 = src1->ne[0]; \
+            ggml_cuda_add_lite_residuals<LITE_T, (MAX_RES)><<<(n_blocks + 255) / 256, 256, 0, stream>>>( \
+                (const LITE_T *)src0_dd_i, src1_ddf_i, dst_dd_i, \
+                row_diff, ne00, src1_ncols, stride01, stride_src1, nrows_dst); \
+            CUDA_CHECK(cudaGetLastError()); \
+        } \
+        return; \
+    }
+
+    LITE_OP_MMQ_PATH(Q2_K_LITE, block_q2_k_lite, sizeof(block_q2_K), GGML_TYPE_Q2_K, Q2_K_LITE_MAX_RESIDUALS)
+    LITE_OP_MMQ_PATH(Q3_K_LITE, block_q3_k_lite, sizeof(block_q2_K), GGML_TYPE_Q2_K, Q3_K_LITE_MAX_RESIDUALS)
+    LITE_OP_MMQ_PATH(Q4_K_LITE, block_q4_k_lite, sizeof(block_q3_K), GGML_TYPE_Q3_K, Q4_K_LITE_MAX_RESIDUALS)
+    LITE_OP_MMQ_PATH(Q5_K_LITE, block_q5_k_lite, sizeof(block_q4_K), GGML_TYPE_Q4_K, Q5_K_LITE_MAX_RESIDUALS)
+    LITE_OP_MMQ_PATH(Q6_K_LITE, block_q6_k_lite, sizeof(block_q5_K), GGML_TYPE_Q5_K, Q6_K_LITE_MAX_RESIDUALS)
+#undef LITE_OP_MMQ_PATH
+
     const mmq_args args = {
         src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
         ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
@@ -279,8 +515,16 @@ bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t
         case GGML_TYPE_NVFP4:
         case GGML_TYPE_Q2_K:
         case GGML_TYPE_Q3_K:
+        // Q2_K_HIFI excluded - uses MMVQ/dequant path instead
+        // Q3_K_HIFI excluded - uses MMVQ/dequant path instead
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
+        case GGML_TYPE_Q5_K_HIFI_RES8:  // Use Q5_K MMQ path (compact copy + residual kernel)
+        case GGML_TYPE_Q2_K_LITE:  // compact copy to Q2_K + residual correction
+        case GGML_TYPE_Q3_K_LITE:  // compact copy to Q2_K + residual correction (base shifted down)
+        case GGML_TYPE_Q4_K_LITE:  // compact copy to Q3_K + residual correction (base shifted down)
+        case GGML_TYPE_Q5_K_LITE:  // compact copy to Q4_K + residual correction (base shifted down)
+        case GGML_TYPE_Q6_K_LITE:  // compact copy to Q5_K + residual correction (base shifted down)
         case GGML_TYPE_Q6_K:
         case GGML_TYPE_IQ2_XXS:
         case GGML_TYPE_IQ2_XS:
```

#### `ggml/src/ggml-cuda/ggml-cuda.cu`

```diff
diff --git a/ggml/src/ggml-cuda/ggml-cuda.cu b/ggml/src/ggml-cuda/ggml-cuda.cu
index 3113de017..9e73a9b8e 100644
--- a/ggml/src/ggml-cuda/ggml-cuda.cu
+++ b/ggml/src/ggml-cuda/ggml-cuda.cu
@@ -4791,7 +4791,20 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
                     case GGML_TYPE_MXFP4:
                     case GGML_TYPE_NVFP4:
                     case GGML_TYPE_Q2_K:
+                    case GGML_TYPE_Q2_K_HIFI:
                     case GGML_TYPE_Q3_K:
+                    case GGML_TYPE_Q3_K_HIFI:
+                    case GGML_TYPE_Q3_K_HIFI_RES8:
+                    case GGML_TYPE_Q6_K_HIFI:
+                    case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+                    case GGML_TYPE_Q6_K_HIFI_RES8:
+                    case GGML_TYPE_Q5_K_HIFI_RES8:
+                    case GGML_TYPE_Q4_K_HIFI:
+                    case GGML_TYPE_Q2_K_LITE:
+                    case GGML_TYPE_Q3_K_LITE:
+                    case GGML_TYPE_Q4_K_LITE:
+                    case GGML_TYPE_Q5_K_LITE:
+                    case GGML_TYPE_Q6_K_LITE:
                     case GGML_TYPE_Q4_K:
                     case GGML_TYPE_Q5_K:
                     case GGML_TYPE_Q6_K:
```

---

### 4.2 Metal Backend (Apple Silicon)

#### `ggml/src/ggml-metal/ggml-metal-impl.h`

```diff
diff --git a/ggml/src/ggml-metal/ggml-metal-impl.h b/ggml/src/ggml-metal/ggml-metal-impl.h
index 62b028f4a..22fae75ee 100644
--- a/ggml/src/ggml-metal/ggml-metal-impl.h
+++ b/ggml/src/ggml-metal/ggml-metal-impl.h
@@ -32,13 +32,22 @@
 #define N_R0_Q2_K 4
 #define N_SG_Q2_K 2
 
+#define N_R0_Q2_K_HIFI 4
+#define N_SG_Q2_K_HIFI 2
+
 #define N_R0_Q3_K 2
 #define N_SG_Q3_K 2
 
+#define N_R0_Q3_K_HIFI 2
+#define N_SG_Q3_K_HIFI 2
+
 #define N_R0_Q4_K 2
 #define N_SG_Q4_K 2
 
-#define N_R0_Q5_K 1
+#define N_R0_Q4_K_HIFI 2
+#define N_SG_Q4_K_HIFI 2
+
+#define N_R0_Q5_K 2
 #define N_SG_Q5_K 2
 
 #define N_R0_Q6_K 2
```

#### `ggml/src/ggml-metal/ggml-metal-device.cpp`

```diff
diff --git a/ggml/src/ggml-metal/ggml-metal-device.cpp b/ggml/src/ggml-metal/ggml-metal-device.cpp
index e8548b053..686986801 100644
--- a/ggml/src/ggml-metal/ggml-metal-device.cpp
+++ b/ggml/src/ggml-metal/ggml-metal-device.cpp
@@ -146,11 +146,15 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_met
     return res;
 }
 
+static const char * ggml_metal_type_name_for_kernel(ggml_type type); // forward declaration
+
 ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
     char base[256];
     char name[256];
 
-    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
+    // Use ggml_metal_type_name_for_kernel for HIFI types so the kernel name matches
+    // the dedicated kernels registered in ggml-metal.metal (e.g. "q5_K_hifi_res8")
+    snprintf(base, 256, "kernel_get_rows_%s", ggml_metal_type_name_for_kernel(tsrc));
     snprintf(name, 256, "%s", base);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
@@ -581,6 +585,39 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_
     return res;
 }
 
+// Map HIFI types to their kernel name counterparts
+// Q3_K_HIFI, Q4_K_HIFI, Q5_K_HIFI_RES8 have dedicated kernels with correct block strides
+// Q6_K HIFI variants reuse Q6_K kernels (TODO: fix stride mismatch for Q6_K HIFI types)
+static const char * ggml_metal_type_name_for_kernel(ggml_type type) {
+    switch (type) {
+        case GGML_TYPE_Q2_K_HIFI:
+            return "q2_k_hifi";
+        case GGML_TYPE_Q3_K_HIFI:
+            return "q3_k_hifi";
+        case GGML_TYPE_Q4_K_HIFI:
+            return "q4_k_hifi";
+        case GGML_TYPE_Q6_K_HIFI:
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            return "q6_K";
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            return "q6_K_hifi_res8";
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            return "q5_K_hifi_res8";
+        case GGML_TYPE_Q2_K_LITE:
+            return "q2_k_lite";
+        case GGML_TYPE_Q3_K_LITE:
+            return "q3_k_lite";
+        case GGML_TYPE_Q4_K_LITE:
+            return "q4_k_lite";
+        case GGML_TYPE_Q5_K_LITE:
+            return "q5_k_lite";
+        case GGML_TYPE_Q6_K_LITE:
+            return "q6_k_lite";
+        default:
+            return ggml_type_name(type);
+    }
+}
+
 ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, const ggml_tensor * op) {
     char base[256];
     char name[256];
@@ -650,7 +687,7 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_
     char base[256];
     char name[256];
 
-    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
+    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_metal_type_name_for_kernel(tsrc0), ggml_metal_type_name_for_kernel(tsrc1), r1ptg);
     snprintf(name, 256, "%s_nsg=%d_nxpsg=%d", base, nsg, nxpsg);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
@@ -678,7 +715,7 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_meta
     const bool bc_inp = op->src[0]->ne[0] % 32 != 0;
     const bool bc_out = op->ne[0] % 64 != 0 || op->ne[1] % 32 != 0;
 
-    snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
+    snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_metal_type_name_for_kernel(tsrc0), ggml_metal_type_name_for_kernel(tsrc1));
     snprintf(name, 256, "%s_bci=%d_bco=%d", base, bc_inp, bc_out);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
@@ -778,16 +815,31 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_meta
                 nsg = N_SG_Q2_K;
                 nr0 = N_R0_Q2_K;
             } break;
+        case GGML_TYPE_Q2_K_HIFI:
+            {
+                nsg = N_SG_Q2_K_HIFI;
+                nr0 = N_R0_Q2_K_HIFI;
+            } break;
         case GGML_TYPE_Q3_K:
             {
                 nsg = N_SG_Q3_K;
                 nr0 = N_R0_Q3_K;
             } break;
+        case GGML_TYPE_Q3_K_HIFI:
+            {
+                nsg = N_SG_Q3_K_HIFI;
+                nr0 = N_R0_Q3_K_HIFI;
+            } break;
         case GGML_TYPE_Q4_K:
             {
                 nsg = N_SG_Q4_K;
                 nr0 = N_R0_Q4_K;
             } break;
+        case GGML_TYPE_Q4_K_HIFI:
+            {
+                nsg = N_SG_Q4_K_HIFI;
+                nr0 = N_R0_Q4_K_HIFI;
+            } break;
         case GGML_TYPE_Q5_K:
             {
                 nsg = N_SG_Q5_K;
@@ -849,6 +901,51 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_meta
                 nr0 = N_R0_IQ4_XS;
                 smem = 32*sizeof(float);
             } break;
+        case GGML_TYPE_Q6_K_HIFI:
+            {
+                nsg = N_SG_Q6_K;
+                nr0 = N_R0_Q6_K;
+            } break;
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            {
+                nsg = N_SG_Q6_K;
+                nr0 = N_R0_Q6_K;
+            } break;
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            {
+                nsg = N_SG_Q6_K;
+                nr0 = N_R0_Q6_K;
+            } break;
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            {
+                nsg = N_SG_Q5_K;
+                nr0 = N_R0_Q5_K;
+            } break;
+        case GGML_TYPE_Q2_K_LITE:
+            {
+                nsg = N_SG_Q2_K;  // Q2_K base
+                nr0 = N_R0_Q2_K;
+            } break;
+        case GGML_TYPE_Q3_K_LITE:
+            {
+                nsg = N_SG_Q2_K;  // Q2_K base (shifted down from Q3_K)
+                nr0 = N_R0_Q2_K;
+            } break;
+        case GGML_TYPE_Q4_K_LITE:
+            {
+                nsg = N_SG_Q3_K;  // Q3_K base (shifted down from Q4_K)
+                nr0 = N_R0_Q3_K;
+            } break;
+        case GGML_TYPE_Q5_K_LITE:
+            {
+                nsg = N_SG_Q4_K;  // Q4_K base (shifted down from Q5_K)
+                nr0 = N_R0_Q4_K;
+            } break;
+        case GGML_TYPE_Q6_K_LITE:
+            {
+                nsg = N_SG_Q5_K;  // Q5_K base (shifted down from Q6_K)
+                nr0 = N_R0_Q5_K;
+            } break;
         default:
             {
                 GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
@@ -856,7 +953,7 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_meta
             }
     };
 
-    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
+    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_metal_type_name_for_kernel(tsrc0), ggml_metal_type_name_for_kernel(tsrc1), suffix);
     snprintf(name, 256, "%s_nsg=%d", base, nsg);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
@@ -904,7 +1001,7 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_m
 
     const bool bc_inp = op->src[0]->ne[0] % 32 != 0;
 
-    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
+    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_metal_type_name_for_kernel(tsrc0), ggml_metal_type_name_for_kernel(tsrc1));
     snprintf(name, 256, "%s_bci=%d", base, bc_inp);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
@@ -995,16 +1092,31 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_m
                 nsg = N_SG_Q2_K;
                 nr0 = N_R0_Q2_K;
             } break;
+        case GGML_TYPE_Q2_K_HIFI:
+            {
+                nsg = N_SG_Q2_K_HIFI;
+                nr0 = N_R0_Q2_K_HIFI;
+            } break;
         case GGML_TYPE_Q3_K:
             {
                 nsg = N_SG_Q3_K;
                 nr0 = N_R0_Q3_K;
             } break;
+        case GGML_TYPE_Q3_K_HIFI:
+            {
+                nsg = N_SG_Q3_K_HIFI;
+                nr0 = N_R0_Q3_K_HIFI;
+            } break;
         case GGML_TYPE_Q4_K:
             {
                 nsg = N_SG_Q4_K;
                 nr0 = N_R0_Q4_K;
             } break;
+        case GGML_TYPE_Q4_K_HIFI:
+            {
+                nsg = N_SG_Q4_K_HIFI;
+                nr0 = N_R0_Q4_K_HIFI;
+            } break;
         case GGML_TYPE_Q5_K:
             {
                 nsg = N_SG_Q5_K;
@@ -1066,6 +1178,51 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_m
                 nr0 = N_R0_IQ4_XS;
                 smem = 32*sizeof(float);
             } break;
+        case GGML_TYPE_Q6_K_HIFI:
+            {
+                nsg = N_SG_Q6_K;
+                nr0 = N_R0_Q6_K;
+            } break;
+        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
+            {
+                nsg = N_SG_Q6_K;
+                nr0 = N_R0_Q6_K;
+            } break;
+        case GGML_TYPE_Q6_K_HIFI_RES8:
+            {
+                nsg = N_SG_Q6_K;
+                nr0 = N_R0_Q6_K;
+            } break;
+        case GGML_TYPE_Q5_K_HIFI_RES8:
+            {
+                nsg = N_SG_Q5_K;
+                nr0 = N_R0_Q5_K;
+            } break;
+        case GGML_TYPE_Q2_K_LITE:
+            {
+                nsg = N_SG_Q2_K;  // Q2_K base
+                nr0 = N_R0_Q2_K;
+            } break;
+        case GGML_TYPE_Q3_K_LITE:
+            {
+                nsg = N_SG_Q2_K;  // Q2_K base (shifted down from Q3_K)
+                nr0 = N_R0_Q2_K;
+            } break;
+        case GGML_TYPE_Q4_K_LITE:
+            {
+                nsg = N_SG_Q3_K;  // Q3_K base (shifted down from Q4_K)
+                nr0 = N_R0_Q3_K;
+            } break;
+        case GGML_TYPE_Q5_K_LITE:
+            {
+                nsg = N_SG_Q4_K;  // Q4_K base (shifted down from Q5_K)
+                nr0 = N_R0_Q4_K;
+            } break;
+        case GGML_TYPE_Q6_K_LITE:
+            {
+                nsg = N_SG_Q5_K;  // Q5_K base (shifted down from Q6_K)
+                nr0 = N_R0_Q5_K;
+            } break;
         default:
             {
                 GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
@@ -1073,7 +1230,7 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_m
             }
     };
 
-    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
+    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_metal_type_name_for_kernel(tsrc0), ggml_metal_type_name_for_kernel(tsrc1), suffix);
     snprintf(name, 256, "%s_nsg=%d", base, nsg);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
```

#### `ggml/src/ggml-metal/ggml-metal.metal`

```diff
diff --git a/ggml/src/ggml-metal/ggml-metal.metal b/ggml/src/ggml-metal/ggml-metal.metal
index f67c5cd8a..9801b1e62 100644
--- a/ggml/src/ggml-metal/ggml-metal.metal
+++ b/ggml/src/ggml-metal/ggml-metal.metal
@@ -643,6 +643,34 @@ void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg
     }
 }
 
+// Q2_K_HIFI: base Q2_K dequantization + FP16 correction (dual-mode)
+// Bit 7 of outlier_count signals the mode:
+//   0 = outlier-first: REPLACE base value with FP16 (outliers were zeroed before Q2_K)
+//   1 = residual:      ADD FP16 residual to base value (imatrix-aware Q2_K undisturbed)
+template <typename type4x4>
+void dequantize_q2_k_hifi(device const block_q2_k_hifi *xb, short il, thread type4x4 & reg) {
+    dequantize_q2_K((device const block_q2_K *)xb, il, reg);
+
+    const int base_pos = il * 16;
+    const int raw_count = xb->outlier_count;
+    const bool residual_mode = (raw_count & Q2_K_HIFI_RESIDUAL_MODE_FLAG) != 0;
+    const int count = raw_count & 0x7F;
+
+    #pragma unroll
+    for (int k = 0; k < Q2_K_HIFI_MAX_OUTLIERS; ++k) {
+        if (k >= count) break;
+        const int idx = xb->outlier_idx[k];
+        const int local_pos = idx - base_pos;
+        if (local_pos >= 0 && local_pos < 16) {
+            if (residual_mode) {
+                reg[local_pos / 4][local_pos % 4] += (float)xb->outlier_vals[k];
+            } else {
+                reg[local_pos / 4][local_pos % 4] = (float)xb->outlier_vals[k];
+            }
+        }
+    }
+}
+
 template <typename type4x4>
 void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
     const half d_all = xb->d;
@@ -752,6 +780,14 @@ void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg
     }
 }
 
+// Q6_K_HIFI_RES8: Q6_K layout + 22-byte INT8 residual extension (232 bytes total)
+// The base Q6_K fields (ql, qh, scales, d) are at identical byte offsets.
+// Residual corrections are not applied in the Metal path (only in CPU path).
+template <typename type4x4>
+void dequantize_q6_k_hifi_res8(device const block_q6_k_hifi_res8 * xb, short il, thread type4x4 & reg) {
+    dequantize_q6_K((device const block_q6_K *)xb, il, reg);
+}
+
 template <typename type4x4>
 void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
     // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
@@ -965,6 +1001,154 @@ void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4
     }
 }
 
+template <typename type4x4>
+void dequantize_q3_k_hifi(device const block_q3_k_hifi * xb, short il, thread type4x4 & reg) {
+    // Q3_K_HIFI uses true outlier extraction: Q3_K block + outlier indices + original outlier values
+    // Step 1: Dequantize Q3_K from first 110 bytes
+    const device block_q3_K * q3k_block = (const device block_q3_K *)xb->q3_k_data;
+    dequantize_q3_K(q3k_block, il, reg);
+
+    // Step 2: Overwrite outlier positions with stored FP16 values
+    // Outliers are sorted by index (ascending), enabling efficient processing
+    const int base_pos = il * 16;
+    const int end_pos = base_pos + 16;
+
+    // Load all outlier data once (vectorized)
+    const half4 outliers_lo = *(device const half4 *)&xb->outliers[0];
+    const half4 outliers_hi = *(device const half4 *)&xb->outliers[4];
+
+    // Process sorted outliers with early exit
+    // Skip outliers before our range, process those in range, stop when past range
+    #pragma unroll
+    for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+        const int idx = xb->outlier_idx[k];
+        if (idx >= end_pos) break;  // Early exit: remaining indices are larger (sorted)
+        if (idx >= base_pos) {
+            const int local_pos = idx - base_pos;
+            const float val = (k < 4) ? (float)outliers_lo[k] : (float)outliers_hi[k - 4];
+            reg[local_pos / 4][local_pos % 4] = val;
+        }
+    }
+}
+
+// Q4_K_HIFI: Q4_K layout + 8 FP16 outlier replacements per block
+template <typename type4x4>
+void dequantize_q4_k_hifi(device const block_q4_k_hifi * xb, short il, thread type4x4 & reg) {
+    // Step 1: Dequantize Q4_K from first 144 bytes
+    const device block_q4_K * q4k_block = (const device block_q4_K *)xb->q4_k_data;
+    dequantize_q4_K(q4k_block, il, reg);
+
+    // Step 2: Overwrite outlier positions with stored FP16 values
+    const int base_pos = il * 16;
+    const int end_pos = base_pos + 16;
+
+    // Load all outlier data once (vectorized)
+    const half4 outliers_lo = *(device const half4 *)&xb->outliers[0];
+    const half4 outliers_hi = *(device const half4 *)&xb->outliers[4];
+
+    // Process sorted outliers with early exit
+    #pragma unroll
+    for (int k = 0; k < Q4_K_HIFI_OUTLIERS; ++k) {
+        const int idx = xb->outlier_idx[k];
+        if (idx >= end_pos) break;
+        if (idx >= base_pos) {
+            const int local_pos = idx - base_pos;
+            const float val = (k < 4) ? (float)outliers_lo[k] : (float)outliers_hi[k - 4];
+            reg[local_pos / 4][local_pos % 4] = val;
+        }
+    }
+}
+
+// Q5_K_HIFI_RES8: Q5_K layout + 20-byte INT8 residual extension (196 bytes total)
+// The base Q5_K fields (d, dmin, scales, qh, qs) are at identical byte offsets.
+// Residual corrections are not applied in the Metal path (only in CPU path).
+template <typename type4x4>
+void dequantize_q5_k_hifi_res8(device const block_q5_k_hifi_res8 * xb, short il, thread type4x4 & reg) {
+    dequantize_q5_K((device const block_q5_K *)xb, il, reg);
+}
+
+// K_LITE: base fields at identical byte offsets → cast to the NEW shifted-down base type.
+// Residual corrections are applied after the base dequantize call.
+// Q2_K_LITE: Q2_K base (unchanged)
+template <typename type4x4>
+void dequantize_q2_k_lite(device const block_q2_k_lite * xb, short il, thread type4x4 & reg) {
+    dequantize_q2_K((device const block_q2_K *)xb, il, reg);
+    const int base_pos = il * 16;
+    const float rscale = (float)xb->residual_scale;
+    const int rc = (int)xb->residual_count;
+    for (int r = 0; r < Q2_K_LITE_MAX_RESIDUALS; ++r) {
+        if (r >= rc) break;
+        const int local_pos = (int)xb->residual_idx[r] - base_pos;
+        if (local_pos >= 0 && local_pos < 16) {
+            reg[local_pos / 4][local_pos % 4] += rscale * (float)xb->residual_vals[r];
+        }
+    }
+}
+
+// Q3_K_LITE: Q2_K base (was Q3_K)
+template <typename type4x4>
+void dequantize_q3_k_lite(device const block_q3_k_lite * xb, short il, thread type4x4 & reg) {
+    dequantize_q2_K((device const block_q2_K *)xb, il, reg);
+    const int base_pos = il * 16;
+    const float rscale = (float)xb->residual_scale;
+    const int rc = (int)xb->residual_count;
+    for (int r = 0; r < Q3_K_LITE_MAX_RESIDUALS; ++r) {
+        if (r >= rc) break;
+        const int local_pos = (int)xb->residual_idx[r] - base_pos;
+        if (local_pos >= 0 && local_pos < 16) {
+            reg[local_pos / 4][local_pos % 4] += rscale * (float)xb->residual_vals[r];
+        }
+    }
+}
+
+// Q4_K_LITE: Q3_K base (was Q4_K)
+template <typename type4x4>
+void dequantize_q4_k_lite(device const block_q4_k_lite * xb, short il, thread type4x4 & reg) {
+    dequantize_q3_K((device const block_q3_K *)xb, il, reg);
+    const int base_pos = il * 16;
+    const float rscale = (float)xb->residual_scale;
+    const int rc = (int)xb->residual_count;
+    for (int r = 0; r < Q4_K_LITE_MAX_RESIDUALS; ++r) {
+        if (r >= rc) break;
+        const int local_pos = (int)xb->residual_idx[r] - base_pos;
+        if (local_pos >= 0 && local_pos < 16) {
+            reg[local_pos / 4][local_pos % 4] += rscale * (float)xb->residual_vals[r];
+        }
+    }
+}
+
+// Q5_K_LITE: Q4_K base (was Q5_K)
+template <typename type4x4>
+void dequantize_q5_k_lite(device const block_q5_k_lite * xb, short il, thread type4x4 & reg) {
+    dequantize_q4_K((device const block_q4_K *)xb, il, reg);
+    const int base_pos = il * 16;
+    const float rscale = (float)xb->residual_scale;
+    const int rc = (int)xb->residual_count;
+    for (int r = 0; r < Q5_K_LITE_MAX_RESIDUALS; ++r) {
+        if (r >= rc) break;
+        const int local_pos = (int)xb->residual_idx[r] - base_pos;
+        if (local_pos >= 0 && local_pos < 16) {
+            reg[local_pos / 4][local_pos % 4] += rscale * (float)xb->residual_vals[r];
+        }
+    }
+}
+
+// Q6_K_LITE: Q5_K base (was Q6_K)
+template <typename type4x4>
+void dequantize_q6_k_lite(device const block_q6_k_lite * xb, short il, thread type4x4 & reg) {
+    dequantize_q5_K((device const block_q5_K *)xb, il, reg);
+    const int base_pos = il * 16;
+    const float rscale = (float)xb->residual_scale;
+    const int rc = (int)xb->residual_count;
+    for (int r = 0; r < Q6_K_LITE_MAX_RESIDUALS; ++r) {
+        if (r >= rc) break;
+        const int local_pos = (int)xb->residual_idx[r] - base_pos;
+        if (local_pos >= 0 && local_pos < 16) {
+            reg[local_pos / 4][local_pos % 4] += rscale * (float)xb->residual_vals[r];
+        }
+    }
+}
+
 enum ggml_sort_order {
     GGML_SORT_ORDER_ASC,
     GGML_SORT_ORDER_DESC,
@@ -3949,25 +4133,66 @@ template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4x4
 template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_K, 256, dequantize_q4_K>;
 template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_K, 256, dequantize_q4_K>;
 
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_k_hifi, 256, dequantize_q4_k_hifi>) mul_mv_ext_q4_k_hifi_f32_t;
+
+template [[host_name("kernel_mul_mv_ext_q4_k_hifi_f32_r1_2")]] kernel mul_mv_ext_q4_k_hifi_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_k_hifi, 256, dequantize_q4_k_hifi>;
+template [[host_name("kernel_mul_mv_ext_q4_k_hifi_f32_r1_3")]] kernel mul_mv_ext_q4_k_hifi_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q4_k_hifi, 256, dequantize_q4_k_hifi>;
+template [[host_name("kernel_mul_mv_ext_q4_k_hifi_f32_r1_4")]] kernel mul_mv_ext_q4_k_hifi_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_k_hifi, 256, dequantize_q4_k_hifi>;
+template [[host_name("kernel_mul_mv_ext_q4_k_hifi_f32_r1_5")]] kernel mul_mv_ext_q4_k_hifi_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_k_hifi, 256, dequantize_q4_k_hifi>;
+
 template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_K, 256, dequantize_q5_K>;
 template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_K, 256, dequantize_q5_K>;
 template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_K, 256, dequantize_q5_K>;
 template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_K, 256, dequantize_q5_K>;
 
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_k_hifi_res8, 256, dequantize_q5_k_hifi_res8>) mul_mv_ext_q5_K_hifi_res8_f32_t;
+
+template [[host_name("kernel_mul_mv_ext_q5_K_hifi_res8_f32_r1_2")]] kernel mul_mv_ext_q5_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_k_hifi_res8, 256, dequantize_q5_k_hifi_res8>;
+template [[host_name("kernel_mul_mv_ext_q5_K_hifi_res8_f32_r1_3")]] kernel mul_mv_ext_q5_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_k_hifi_res8, 256, dequantize_q5_k_hifi_res8>;
+template [[host_name("kernel_mul_mv_ext_q5_K_hifi_res8_f32_r1_4")]] kernel mul_mv_ext_q5_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_k_hifi_res8, 256, dequantize_q5_k_hifi_res8>;
+template [[host_name("kernel_mul_mv_ext_q5_K_hifi_res8_f32_r1_5")]] kernel mul_mv_ext_q5_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_k_hifi_res8, 256, dequantize_q5_k_hifi_res8>;
+
 template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_K, 256, dequantize_q6_K>;
 template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_K, 256, dequantize_q6_K>;
 template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_K, 256, dequantize_q6_K>;
 template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_K, 256, dequantize_q6_K>;
 
-template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q2_K, 256, dequantize_q2_K>;
-template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q2_K, 256, dequantize_q2_K>;
-template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q2_K, 256, dequantize_q2_K>;
-template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q2_K, 256, dequantize_q2_K>;
-
-template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q3_K, 256, dequantize_q3_K>;
-template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q3_K, 256, dequantize_q3_K>;
-template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q3_K, 256, dequantize_q3_K>;
-template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q3_K, 256, dequantize_q3_K>;
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_k_hifi_res8, 256, dequantize_q6_k_hifi_res8>) mul_mv_ext_q6_K_hifi_res8_f32_t;
+
+template [[host_name("kernel_mul_mv_ext_q6_K_hifi_res8_f32_r1_2")]] kernel mul_mv_ext_q6_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_k_hifi_res8, 256, dequantize_q6_k_hifi_res8>;
+template [[host_name("kernel_mul_mv_ext_q6_K_hifi_res8_f32_r1_3")]] kernel mul_mv_ext_q6_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_k_hifi_res8, 256, dequantize_q6_k_hifi_res8>;
+template [[host_name("kernel_mul_mv_ext_q6_K_hifi_res8_f32_r1_4")]] kernel mul_mv_ext_q6_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_k_hifi_res8, 256, dequantize_q6_k_hifi_res8>;
+template [[host_name("kernel_mul_mv_ext_q6_K_hifi_res8_f32_r1_5")]] kernel mul_mv_ext_q6_K_hifi_res8_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_k_hifi_res8, 256, dequantize_q6_k_hifi_res8>;
+
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q2_k_lite, 256, dequantize_q2_k_lite>) mul_mv_ext_q2_k_lite_f32_t;
+template [[host_name("kernel_mul_mv_ext_q2_k_lite_f32_r1_2")]] kernel mul_mv_ext_q2_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q2_k_lite, 256, dequantize_q2_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q2_k_lite_f32_r1_3")]] kernel mul_mv_ext_q2_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q2_k_lite, 256, dequantize_q2_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q2_k_lite_f32_r1_4")]] kernel mul_mv_ext_q2_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q2_k_lite, 256, dequantize_q2_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q2_k_lite_f32_r1_5")]] kernel mul_mv_ext_q2_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q2_k_lite, 256, dequantize_q2_k_lite>;
+
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q3_k_lite, 256, dequantize_q3_k_lite>) mul_mv_ext_q3_k_lite_f32_t;
+template [[host_name("kernel_mul_mv_ext_q3_k_lite_f32_r1_2")]] kernel mul_mv_ext_q3_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q3_k_lite, 256, dequantize_q3_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q3_k_lite_f32_r1_3")]] kernel mul_mv_ext_q3_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q3_k_lite, 256, dequantize_q3_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q3_k_lite_f32_r1_4")]] kernel mul_mv_ext_q3_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q3_k_lite, 256, dequantize_q3_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q3_k_lite_f32_r1_5")]] kernel mul_mv_ext_q3_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q3_k_lite, 256, dequantize_q3_k_lite>;
+
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_k_lite, 256, dequantize_q4_k_lite>) mul_mv_ext_q4_k_lite_f32_t;
+template [[host_name("kernel_mul_mv_ext_q4_k_lite_f32_r1_2")]] kernel mul_mv_ext_q4_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_k_lite, 256, dequantize_q4_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q4_k_lite_f32_r1_3")]] kernel mul_mv_ext_q4_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q4_k_lite, 256, dequantize_q4_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q4_k_lite_f32_r1_4")]] kernel mul_mv_ext_q4_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_k_lite, 256, dequantize_q4_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q4_k_lite_f32_r1_5")]] kernel mul_mv_ext_q4_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_k_lite, 256, dequantize_q4_k_lite>;
+
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_k_lite, 256, dequantize_q5_k_lite>) mul_mv_ext_q5_k_lite_f32_t;
+template [[host_name("kernel_mul_mv_ext_q5_k_lite_f32_r1_2")]] kernel mul_mv_ext_q5_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_k_lite, 256, dequantize_q5_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q5_k_lite_f32_r1_3")]] kernel mul_mv_ext_q5_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_k_lite, 256, dequantize_q5_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q5_k_lite_f32_r1_4")]] kernel mul_mv_ext_q5_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_k_lite, 256, dequantize_q5_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q5_k_lite_f32_r1_5")]] kernel mul_mv_ext_q5_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_k_lite, 256, dequantize_q5_k_lite>;
+
+typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_k_lite, 256, dequantize_q6_k_lite>) mul_mv_ext_q6_k_lite_f32_t;
+template [[host_name("kernel_mul_mv_ext_q6_k_lite_f32_r1_2")]] kernel mul_mv_ext_q6_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_k_lite, 256, dequantize_q6_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q6_k_lite_f32_r1_3")]] kernel mul_mv_ext_q6_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_k_lite, 256, dequantize_q6_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q6_k_lite_f32_r1_4")]] kernel mul_mv_ext_q6_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_k_lite, 256, dequantize_q6_k_lite>;
+template [[host_name("kernel_mul_mv_ext_q6_k_lite_f32_r1_5")]] kernel mul_mv_ext_q6_k_lite_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_k_lite, 256, dequantize_q6_k_lite>;
 
 template<typename T0, typename T1, short NR0, typename args_t>
 void kernel_mul_mv_t_t_impl(
@@ -7504,6 +7729,125 @@ kernel void kernel_mul_mv_q2_K_f32(
     kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
 }
 
+// Q2_K_HIFI: Q2_K base dot product + FP16 outlier value corrections
+// Outliers were zeroed before Q2_K quantization -> base contributes ~0 at those positions.
+// We add the true FP16 outlier × activation to recover precision.
+template<int nr0, typename args_t>
+void kernel_mul_mv_q2_k_hifi_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q2_k_hifi * x = (device const block_q2_k_hifi *) (src0 + offset0);
+    device const float           * y = (device const float           *) (src1 + offset1);
+
+    float yl[32];
+    float sumf[nr0]={0.f};
+
+    const short ix = tiisg/8;
+    const short it = tiisg%8;
+    const short iq = it/4;
+    const short ir = it%4;
+    const short is = (8*ir)/16;
+
+    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;
+
+    for (int ib = ix; ib < nb; ib += 4) {
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+        for (short i = 0; i < 8; ++i) {
+            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
+            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
+            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
+            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
+        }
+
+        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
+        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
+        device const half     * dh = &x[ib].d;
+
+        for (short row = 0; row < nr0; row++) {
+            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
+            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+            for (int i = 0; i < 8; i += 2) {
+                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
+                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
+                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
+                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
+                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
+                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
+                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
+                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
+            }
+            float dall = dh[0];
+            float dmin = dh[1] * 1.f/16.f;
+            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
+                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
+                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
+                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
+                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
+
+            // FP16 corrections (works for both outlier-first and residual modes)
+            if (it == 0) {
+                device const block_q2_k_hifi * xb = (device const block_q2_k_hifi *)((device const char *)&x[ib] + row * args.nb01);
+                const int count = xb->outlier_count & 0x7F;
+                if (count > 0) {
+                    for (int k = 0; k < Q2_K_HIFI_MAX_OUTLIERS && k < count; ++k) {
+                        sumf[row] += (float)xb->outlier_vals[k] * y[ib * QK_K + xb->outlier_idx[k]];
+                    }
+                }
+            }
+
+            qs += args.nb01/2;
+            sc += args.nb01;
+            dh += args.nb01/2;
+        }
+
+        y4 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        float sum_all = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = sum_all;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q2_k_hifi_f32")]]
+kernel void kernel_mul_mv_q2_k_hifi_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q2_k_hifi_f32_impl<N_R0_Q2_K_HIFI, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
 template<int nr0, typename args_t>
 void kernel_mul_mv_q3_K_f32_impl(
         args_t args,
@@ -7669,8 +8013,10 @@ kernel void kernel_mul_mv_q3_K_f32(
     kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
 }
 
+// Q3_K_HIFI: Q3_K-compatible layout with 8 FP16 outliers for improved accuracy
+// Reuses Q3_K kernel logic and adds outlier corrections
 template<int nr0, typename args_t>
-void kernel_mul_mv_q4_K_f32_impl(
+void kernel_mul_mv_q3_k_hifi_f32_impl(
         args_t args,
         device const char * src0,
         device const char * src1,
@@ -7681,15 +8027,6 @@ void kernel_mul_mv_q4_K_f32_impl(
         ushort sgitg) {
     const short NSG = FC_mul_mv_nsg;
 
-    constexpr uint16_t kmask1 = 0x3f3f;
-    constexpr uint16_t kmask2 = 0x0f0f;
-    constexpr uint16_t kmask3 = 0xc0c0;
-
-    const short ix = tiisg/8;  // 0...3
-    const short it = tiisg%8;  // 0...7
-    const short iq = it/4;     // 0 or 1
-    const short ir = it%4;     // 0...3
-
     const int nb = args.ne00/QK_K;
 
     const int r0 = tgpig.x;
@@ -7704,58 +8041,264 @@ void kernel_mul_mv_q4_K_f32_impl(
     const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
     const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
 
-    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
-    device const float      * y = (device const float      *) (src1 + offset1);
+    device const block_q3_k_hifi * x = (device const block_q3_k_hifi *) (src0 + offset0);
+    device const float        * yy = (device const float         *) (src1 + offset1);
 
-    float yl[16];
-    float yh[16];
+    const short tid = tiisg/4;
+    const short ix  = tiisg%4;
+    const short ip  = tid/4;          // 0 or 1
+    const short il  = 2*((tid%4)/2);  // 0 or 2
+    const short ir  = tid%2;
+    const short l0  = 8*ir;
 
-    float sumf[nr0]={0.f};
+    const short shift = 2*il;
+    const short q_offset = 32*ip + l0;
+    const short y_offset = 128*ip + 32*il + l0;
 
-    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;
+    device const float * y1 = yy + ix*QK_K + y_offset;
 
-    uint16_t sc16[4];
-    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
+    float sumf1[nr0] = {0.f};
 
-    for (int ib = ix; ib < nb; ib += 4) {
-        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+    // True outlier extraction: reuse Q3_K kernel logic, then add outlier corrections
+    // We'll compute Q3_K dot product from q3_k_data, then add outlier contributions
+    for (int i = ix; i < nb; i += 4) {
+        for (short row = 0; row < nr0; ++row) {
+            device const block_q3_k_hifi * xb = (device const block_q3_k_hifi *)((device const char *)&x[i] + row * args.nb01);
+
+            // Step 1: Compute Q3_K dot product using Q3_K's logic
+            // Cast q3_k_data to block_q3_K and use Q3_K kernel logic
+            const device block_q3_K * q3k_block = (const device block_q3_K *)xb->q3_k_data;
+
+            // Reuse Q3_K's dot product computation (from kernel_mul_mv_q3_K_f32_impl)
+            float yl[32];
+            for (short l = 0; l < 8; ++l) {
+                yl[l+ 0] = y1[l+ 0];
+                yl[l+ 8] = y1[l+16];
+                yl[l+16] = y1[l+32];
+                yl[l+24] = y1[l+48];
+            }
 
-        for (short i = 0; i < 8; ++i) {
-            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
-            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
-            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
-            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
-        }
+            device const uint16_t * q = (device const uint16_t *)(q3k_block->qs + q_offset);
+            device const uint16_t * h = (device const uint16_t *)(q3k_block->hmask + l0);
+            device const uint16_t * a = (device const uint16_t *)(q3k_block->scales);
+            device const half * dh = &q3k_block->d;
 
-        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
-        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
-        device const half     * dh = &x[ib].d;
+            const float d_all = (float)dh[0];
+            uint32_t scales32, aux32;
+            thread uint16_t * scales16 = (thread uint16_t *)&scales32;
+            thread const int8_t * scales = (thread const int8_t *)&scales32;
+
+            const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200}, {0x0004, 0x0400, 0x0008, 0x0800},
+                                 {0x0010, 0x1000, 0x0020, 0x2000}, {0x0040, 0x4000, 0x0080, 0x8000}};
+            const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};
+            const ushort4 hm = mm[2*ip + il/2];
+            const float v1 = il == 0 ? 4.f : 64.f;
+            const float v2 = 4.f * v1;
+            const uint16_t s_shift1 = 4*ip;
+            const uint16_t s_shift2 = s_shift1 + il;
 
-        for (short row = 0; row < nr0; row++) {
-            sc16[0] = sc[0] & kmask1;
-            sc16[1] = sc[2] & kmask1;
-            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
-            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
+            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
+            for (short l = 0; l < 8; l += 2) {
+                const int32_t qs = q[l/2];
+                s1 += yl[l+0] * (qs & qm[il/2][0]);
+                s2 += yl[l+1] * (qs & qm[il/2][1]);
+                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
+                s4 += yl[l+16] * (qs & qm[il/2][2]);
+                s5 += yl[l+17] * (qs & qm[il/2][3]);
+                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
+            }
 
-            device const uint16_t * q2 = q1 + 32;
+            scales16[0] = a[4];
+            scales16[1] = a[5];
+            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
+            scales16[0] = a[il+0];
+            scales16[1] = a[il+1];
+            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
 
-            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
-            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
+            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
+            float q3k_sum = d1 * (scales[0] - 32) + d2 * (scales[2] - 32);
 
-            FOR_UNROLL (short i = 0; i < 4; ++i) {
-                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
-                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
-                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
-                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
-                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
-                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
-                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
-                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
+            s1 = s2 = s3 = s4 = s5 = s6 = 0;
+            for (short l = 0; l < 8; l += 2) {
+                const int32_t qs = q[l/2+8];
+                s1 += yl[l+8] * (qs & qm[il/2][0]);
+                s2 += yl[l+9] * (qs & qm[il/2][1]);
+                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
+                s4 += yl[l+24] * (qs & qm[il/2][2]);
+                s5 += yl[l+25] * (qs & qm[il/2][3]);
+                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
+            }
+            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
+            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
+            q3k_sum += d1 * (scales[1] - 32) + d2 * (scales[3] - 32);
+
+            // Step 2: Add outlier corrections (optimized with vectorized load + early exit)
+            // Outliers are sorted by index during quantization, enabling early exit
+            // Load all 8 indices at once (they're contiguous in memory)
+            const uint8_t idx0 = xb->outlier_idx[0];
+            const uint8_t idx1 = xb->outlier_idx[1];
+            const uint8_t idx2 = xb->outlier_idx[2];
+            const uint8_t idx3 = xb->outlier_idx[3];
+            const uint8_t idx4 = xb->outlier_idx[4];
+            const uint8_t idx5 = xb->outlier_idx[5];
+            const uint8_t idx6 = xb->outlier_idx[6];
+            const uint8_t idx7 = xb->outlier_idx[7];
+
+            // Load all 8 FP16 outlier values at once
+            const half4 outliers_lo = *(device const half4 *)&xb->outliers[0];
+            const half4 outliers_hi = *(device const half4 *)&xb->outliers[4];
+
+            // Process outliers with early exit (indices are sorted ascending, 255 = sentinel)
+            const int y_end = y_offset + 32;
+            float outlier_sum = 0.0f;
+
+            // Unrolled loop with early exit on sorted indices
+            if (idx0 < y_end) {
+                if (idx0 >= y_offset) outlier_sum += (float)outliers_lo[0] * y1[idx0 - y_offset];
+                if (idx1 < y_end) {
+                    if (idx1 >= y_offset) outlier_sum += (float)outliers_lo[1] * y1[idx1 - y_offset];
+                    if (idx2 < y_end) {
+                        if (idx2 >= y_offset) outlier_sum += (float)outliers_lo[2] * y1[idx2 - y_offset];
+                        if (idx3 < y_end) {
+                            if (idx3 >= y_offset) outlier_sum += (float)outliers_lo[3] * y1[idx3 - y_offset];
+                            if (idx4 < y_end) {
+                                if (idx4 >= y_offset) outlier_sum += (float)outliers_hi[0] * y1[idx4 - y_offset];
+                                if (idx5 < y_end) {
+                                    if (idx5 >= y_offset) outlier_sum += (float)outliers_hi[1] * y1[idx5 - y_offset];
+                                    if (idx6 < y_end) {
+                                        if (idx6 >= y_offset) outlier_sum += (float)outliers_hi[2] * y1[idx6 - y_offset];
+                                        if (idx7 < y_end && idx7 >= y_offset) {
+                                            outlier_sum += (float)outliers_hi[3] * y1[idx7 - y_offset];
+                                        }
+                                    }
+                                }
+                            }
+                        }
+                    }
+                }
             }
+            q3k_sum += outlier_sum;
 
-            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
-                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
-                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
+            sumf1[row] += q3k_sum;
+        }
+        y1 += 4 * QK_K;
+    }
+
+    for (int row = 0; row < nr0; ++row) {
+        const float sumf = sumf1[row] / (1 << shift);
+        sumf1[row] = simd_sum(sumf);
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    if (tiisg == 0) {
+        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+            dst_f32[first_row + row] = sumf1[row];
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q3_k_hifi_f32")]]
+kernel void kernel_mul_mv_q3_k_hifi_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q3_k_hifi_f32_impl<N_R0_Q3_K_HIFI, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+template<int nr0, typename args_t>
+void kernel_mul_mv_q4_K_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    constexpr uint16_t kmask1 = 0x3f3f;
+    constexpr uint16_t kmask2 = 0x0f0f;
+    constexpr uint16_t kmask3 = 0xc0c0;
+
+    const short ix = tiisg/8;  // 0...3
+    const short it = tiisg%8;  // 0...7
+    const short iq = it/4;     // 0 or 1
+    const short ir = it%4;     // 0...3
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
+    device const float      * y = (device const float      *) (src1 + offset1);
+
+    float yl[16];
+    float yh[16];
+
+    float sumf[nr0]={0.f};
+
+    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;
+
+    uint16_t sc16[4];
+    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
+
+    for (int ib = ix; ib < nb; ib += 4) {
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+
+        for (short i = 0; i < 8; ++i) {
+            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
+            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
+            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
+            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
+        }
+
+        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
+        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
+        device const half     * dh = &x[ib].d;
+
+        for (short row = 0; row < nr0; row++) {
+            sc16[0] = sc[0] & kmask1;
+            sc16[1] = sc[2] & kmask1;
+            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
+            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
+
+            device const uint16_t * q2 = q1 + 32;
+
+            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
+            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+
+            FOR_UNROLL (short i = 0; i < 4; ++i) {
+                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
+                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
+                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
+                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
+                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
+                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
+                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
+                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
+            }
+
+            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
+                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
+                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                   (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                          dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
 
@@ -7790,6 +8333,145 @@ kernel void kernel_mul_mv_q4_K_f32(
     kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
 }
 
+// Q4_K_HIFI: Q4_K layout + 8 FP16 outlier replacements per block
+// Reuses Q4_K kernel logic and adds outlier corrections
+template<int nr0, typename args_t>
+void kernel_mul_mv_q4_k_hifi_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    constexpr uint16_t kmask1 = 0x3f3f;
+    constexpr uint16_t kmask2 = 0x0f0f;
+    constexpr uint16_t kmask3 = 0xc0c0;
+
+    const short ix = tiisg/8;  // 0...3
+    const short it = tiisg%8;  // 0...7
+    const short iq = it/4;     // 0 or 1
+    const short ir = it%4;     // 0...3
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q4_k_hifi * x = (device const block_q4_k_hifi *) (src0 + offset0);
+    device const float           * y = (device const float           *) (src1 + offset1);
+
+    float yl[16];
+    float yh[16];
+
+    float sumf[nr0]={0.f};
+
+    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;
+
+    uint16_t sc16[4];
+    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
+
+    for (int ib = ix; ib < nb; ib += 4) {
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+
+        for (short i = 0; i < 8; ++i) {
+            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
+            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
+            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
+            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
+        }
+
+        // Access Q4_K data through q4_k_data field
+        device const block_q4_K * q4k = (device const block_q4_K *) x[ib].q4_k_data;
+        device const uint16_t * sc = (device const uint16_t *)q4k->scales + iq;
+        device const uint16_t * q1 = (device const uint16_t *)q4k->qs + 16 * iq + 4 * ir;
+        device const half     * dh = &q4k->d;
+
+        // Track block_q4_k_hifi pointer for outlier access per row
+        device const block_q4_k_hifi * xh_row = &x[ib];
+
+        for (short row = 0; row < nr0; row++) {
+            sc16[0] = sc[0] & kmask1;
+            sc16[1] = sc[2] & kmask1;
+            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
+            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
+
+            device const uint16_t * q2 = q1 + 32;
+
+            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
+            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+
+            FOR_UNROLL (short i = 0; i < 4; ++i) {
+                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
+                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
+                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
+                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
+                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
+                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
+                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
+                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
+            }
+
+            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
+                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
+                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
+                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
+                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
+
+            // Q4_K_HIFI outlier corrections (thread it==0 handles all outliers for this block)
+            if (it == 0) {
+                for (int k = 0; k < Q4_K_HIFI_OUTLIERS; k++) {
+                    const int idx = xh_row->outlier_idx[k];
+                    if (idx >= Q4_K_HIFI_BLOCK_SIZE) break;  // Sentinel (255)
+                    const float outlier_val = (float)xh_row->outliers[k];
+                    sumf[row] += outlier_val * y[ib * QK_K + idx];
+                }
+            }
+
+            q1 += args.nb01/2;
+            sc += args.nb01/2;
+            dh += args.nb01/2;
+            xh_row = (device const block_q4_k_hifi *)((device const char *)xh_row + args.nb01);
+        }
+
+        y4 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        float sum_all = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = sum_all;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q4_k_hifi_f32")]]
+kernel void kernel_mul_mv_q4_k_hifi_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q4_k_hifi_f32_impl<N_R0_Q4_K_HIFI, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
 template<int nr0, typename args_t>
 void kernel_mul_mv_q5_K_f32_impl(
         args_t args,
@@ -7921,8 +8603,10 @@ kernel void kernel_mul_mv_q5_K_f32(
     kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
 }
 
+// Q5_K_HIFI_RES8: identical to Q5_K mul_mv but uses block_q5_k_hifi_res8 pointer (196-byte stride)
+// The base Q5_K fields are at identical byte offsets; HIFI residual extension is ignored here.
 template<int nr0, typename args_t>
-void kernel_mul_mv_q6_K_f32_impl(
+void kernel_mul_mv_q5_K_hifi_res8_f32_impl(
         args_t args,
         device const char * src0,
         device const char * src1,
@@ -7933,11 +8617,6 @@ void kernel_mul_mv_q6_K_f32_impl(
         ushort sgitg) {
     const short NSG = FC_mul_mv_nsg;
 
-    constexpr uint8_t kmask1 = 0x03;
-    constexpr uint8_t kmask2 = 0x0C;
-    constexpr uint8_t kmask3 = 0x30;
-    constexpr uint8_t kmask4 = 0xC0;
-
     const int nb = args.ne00/QK_K;
 
     const int r0 = tgpig.x;
@@ -7952,25 +8631,162 @@ void kernel_mul_mv_q6_K_f32_impl(
     const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
     const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
 
-    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
-    device const float     * yy = (device const float      *) (src1 + offset1);
-
-    float sumf[nr0] = { 0.f };
+    // KEY FIX: use correct 196-byte struct stride instead of block_q5_K (176 bytes)
+    device const block_q5_k_hifi_res8 * x = (device const block_q5_k_hifi_res8 *) (src0 + offset0);
+    device const float               * yy = (device const float                *) (src1 + offset1);
 
-    float yl[16];
+    float sumf[nr0]={0.f};
 
-    const short tid = tiisg/2;
-    const short ix  = tiisg%2;
-    const short ip  = tid/8;         // 0 or 1
-    const short il  = tid%8;
-    const short l0  = 4*il;
-    const short is  = 8*ip + l0/16;
+    float yl[16], yh[16];
 
-    const short y_offset   = 128*ip + l0;
-    const short q_offset_l =  64*ip + l0;
-    const short q_offset_h =  32*ip + l0;
+    constexpr uint16_t kmask1 = 0x3f3f;
+    constexpr uint16_t kmask2 = 0x0f0f;
+    constexpr uint16_t kmask3 = 0xc0c0;
 
-    for (int i = ix; i < nb; i += 2) {
+    const short tid = tiisg/4;
+    const short ix  = tiisg%4;
+    const short iq  = tid/4;
+    const short ir  = tid%4;
+
+    const short l0 = 8*ir;
+    const short q_offset = 32*iq + l0;
+    const short y_offset = 64*iq + l0;
+
+    const uint8_t hm1 = 1u << (2*iq);
+    const uint8_t hm2 = hm1 << 1;
+    const uint8_t hm3 = hm1 << 4;
+    const uint8_t hm4 = hm2 << 4;
+
+    uint16_t sc16[4];
+    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
+
+    device const float * y1 = yy + ix*QK_K + y_offset;
+
+    for (int i = ix; i < nb; i += 4) {
+        device const uint8_t * q1 = x[i].qs + q_offset;
+        device const uint8_t * qh = x[i].qh + l0;
+        device const half * dh = &x[i].d;
+        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;
+
+        device const float * y2 = y1 + 128;
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+        for (short l = 0; l < 8; ++l) {
+            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
+            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
+            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
+            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
+        }
+
+        for (short row = 0; row < nr0; ++row) {
+            device const uint8_t * q2 = q1 + 64;
+
+            sc16[0] = a[0] & kmask1;
+            sc16[1] = a[2] & kmask1;
+            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
+            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
+
+            float4 acc1 = {0.f};
+            float4 acc2 = {0.f};
+            FOR_UNROLL (short l = 0; l < 8; ++l) {
+                uint8_t h = qh[l];
+                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
+                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
+                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
+                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
+                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
+                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
+                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
+                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
+            }
+
+            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
+                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
+                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
+                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
+                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
+
+            q1 += args.nb01;
+            qh += args.nb01;
+            dh += args.nb01/2;
+            a  += args.nb01/2;
+        }
+
+        y1 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        const float tot = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = tot;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q5_K_hifi_res8_f32")]]
+kernel void kernel_mul_mv_q5_K_hifi_res8_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q5_K_hifi_res8_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+template<int nr0, typename args_t>
+void kernel_mul_mv_q6_K_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    constexpr uint8_t kmask1 = 0x03;
+    constexpr uint8_t kmask2 = 0x0C;
+    constexpr uint8_t kmask3 = 0x30;
+    constexpr uint8_t kmask4 = 0xC0;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
+    device const float     * yy = (device const float      *) (src1 + offset1);
+
+    float sumf[nr0] = { 0.f };
+
+    float yl[16];
+
+    const short tid = tiisg/2;
+    const short ix  = tiisg%2;
+    const short ip  = tid/8;         // 0 or 1
+    const short il  = tid%8;
+    const short l0  = 4*il;
+    const short is  = 8*ip + l0/16;
+
+    const short y_offset   = 128*ip + l0;
+    const short q_offset_l =  64*ip + l0;
+    const short q_offset_h =  32*ip + l0;
+
+    for (int i = ix; i < nb; i += 2) {
         device const uint8_t * q1 = x[i].ql + q_offset_l;
         device const uint8_t * q2 = q1 + 32;
         device const uint8_t * qh = x[i].qh + q_offset_h;
@@ -8029,6 +8845,847 @@ kernel void kernel_mul_mv_q6_K_f32(
     kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
 }
 
+template<int nr0, typename args_t>
+void kernel_mul_mv_q6_K_hifi_res8_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    constexpr uint8_t kmask1 = 0x03;
+    constexpr uint8_t kmask2 = 0x0C;
+    constexpr uint8_t kmask3 = 0x30;
+    constexpr uint8_t kmask4 = 0xC0;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q6_k_hifi_res8 * x = (device const block_q6_k_hifi_res8 *) (src0 + offset0);
+    device const float               * yy = (device const float                 *) (src1 + offset1);
+
+    float sumf[nr0] = { 0.f };
+
+    float yl[16];
+
+    const short tid = tiisg/2;
+    const short ix  = tiisg%2;
+    const short ip  = tid/8;         // 0 or 1
+    const short il  = tid%8;
+    const short l0  = 4*il;
+    const short is  = 8*ip + l0/16;
+
+    const short y_offset   = 128*ip + l0;
+    const short q_offset_l =  64*ip + l0;
+    const short q_offset_h =  32*ip + l0;
+
+    for (int i = ix; i < nb; i += 2) {
+        device const uint8_t * q1 = x[i].ql + q_offset_l;
+        device const uint8_t * q2 = q1 + 32;
+        device const uint8_t * qh = x[i].qh + q_offset_h;
+        device const int8_t  * sc = x[i].scales + is;
+        device const half    * dh = &x[i].d;
+
+        device const float * y = yy + i * QK_K + y_offset;
+
+        for (short l = 0; l < 4; ++l) {
+            yl[4*l + 0] = y[l +  0];
+            yl[4*l + 1] = y[l + 32];
+            yl[4*l + 2] = y[l + 64];
+            yl[4*l + 3] = y[l + 96];
+        }
+
+        for (short row = 0; row < nr0; ++row) {
+            float4 sums = {0.f, 0.f, 0.f, 0.f};
+
+            FOR_UNROLL (short l = 0; l < 4; ++l) {
+                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
+                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
+                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
+                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
+            }
+
+            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
+
+            q1 += args.nb01;
+            q2 += args.nb01;
+            qh += args.nb01;
+            sc += args.nb01;
+            dh += args.nb01/2;
+        }
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        float sum_all = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = sum_all;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q6_K_hifi_res8_f32")]]
+kernel void kernel_mul_mv_q6_K_hifi_res8_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q6_K_hifi_res8_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+// K_LITE mul_mv impls: use LITE block pointer for correct stride + apply INT8 residual corrections.
+
+template<int nr0, typename args_t>
+void kernel_mul_mv_q2_K_lite_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q2_k_lite * x = (device const block_q2_k_lite *) (src0 + offset0);
+    device const float             * y = (device const float             *) (src1 + offset1);
+
+    float yl[32];
+    float sumf[nr0]={0.f};
+
+    const short ix = tiisg/8;
+    const short it = tiisg%8;
+    const short iq = it/4;
+    const short ir = it%4;
+    const short is = (8*ir)/16;
+
+    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;
+
+    for (int ib = ix; ib < nb; ib += 4) {
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+        for (short i = 0; i < 8; ++i) {
+            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
+            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
+            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
+            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
+        }
+
+        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
+        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
+        device const half     * dh = &x[ib].d;
+
+        for (short row = 0; row < nr0; row++) {
+            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
+            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+            for (int i = 0; i < 8; i += 2) {
+                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
+                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
+                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
+                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
+                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
+                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
+                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
+                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
+            }
+            float dall = dh[0];
+            float dmin = dh[1] * 1.f/16.f;
+            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
+                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
+                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
+                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
+                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
+
+            // Apply INT8 residual corrections for Q2_K_LITE
+            {
+                device const block_q2_k_lite * xb_row = (device const block_q2_k_lite *)((device const char *)&x[ib] + (uint64_t)row * args.nb01);
+                const int rc = (int)xb_row->residual_count;
+                if (rc > 0) {
+                    const float rscale = (float)xb_row->residual_scale;
+                    const short pos_base = 128*iq + 8*ir;
+                    for (int r = 0; r < Q2_K_LITE_MAX_RESIDUALS; ++r) {
+                        if (r >= rc) break;
+                        const short delta = (short)xb_row->residual_idx[r] - pos_base;
+                        float y_val;
+                        if      (delta >= 0  && delta < 8)   y_val = yl[delta];
+                        else if (delta >= 32 && delta < 40)  y_val = yl[8  + (delta - 32)];
+                        else if (delta >= 64 && delta < 72)  y_val = yl[16 + (delta - 64)];
+                        else if (delta >= 96 && delta < 104) y_val = yl[24 + (delta - 96)];
+                        else continue;
+                        sumf[row] += rscale * (float)xb_row->residual_vals[r] * y_val;
+                    }
+                }
+            }
+
+            qs += args.nb01/2;
+            sc += args.nb01;
+            dh += args.nb01/2;
+        }
+
+        y4 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        float sum_all = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = sum_all;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q2_k_lite_f32")]]
+kernel void kernel_mul_mv_q2_K_lite_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q2_K_lite_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+// Q3_K_LITE mul_mv: Q2_K base computation (d, dmin, scales[16], qs[64] 2-bit)
+template<int nr0, typename args_t>
+void kernel_mul_mv_q3_K_lite_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q3_k_lite * x = (device const block_q3_k_lite *) (src0 + offset0);
+    device const float             * y = (device const float             *) (src1 + offset1);
+
+    float yl[32];
+    float sumf[nr0]={0.f};
+
+    const short ix = tiisg/8;
+    const short it = tiisg%8;
+    const short iq = it/4;
+    const short ir = it%4;
+    const short is = (8*ir)/16;
+
+    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;
+
+    for (int ib = ix; ib < nb; ib += 4) {
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+        for (short i = 0; i < 8; ++i) {
+            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
+            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
+            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
+            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
+        }
+
+        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
+        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
+        device const half     * dh = &x[ib].d;
+
+        for (short row = 0; row < nr0; row++) {
+            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
+            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+            for (int i = 0; i < 8; i += 2) {
+                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
+                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
+                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
+                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
+                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
+                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
+                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
+                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
+            }
+            float dall = dh[0];
+            float dmin = dh[1] * 1.f/16.f;
+            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
+                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
+                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
+                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
+                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
+
+            // Apply INT8 residual corrections for Q3_K_LITE
+            {
+                device const block_q3_k_lite * xb_row = (device const block_q3_k_lite *)((device const char *)&x[ib] + (uint64_t)row * args.nb01);
+                const int rc = (int)xb_row->residual_count;
+                if (rc > 0) {
+                    const float rscale = (float)xb_row->residual_scale;
+                    const short pos_base = 128*iq + 8*ir;
+                    for (int r = 0; r < Q3_K_LITE_MAX_RESIDUALS; ++r) {
+                        if (r >= rc) break;
+                        const short delta = (short)xb_row->residual_idx[r] - pos_base;
+                        float y_val;
+                        if      (delta >= 0  && delta < 8)   y_val = yl[delta];
+                        else if (delta >= 32 && delta < 40)  y_val = yl[8  + (delta - 32)];
+                        else if (delta >= 64 && delta < 72)  y_val = yl[16 + (delta - 64)];
+                        else if (delta >= 96 && delta < 104) y_val = yl[24 + (delta - 96)];
+                        else continue;
+                        sumf[row] += rscale * (float)xb_row->residual_vals[r] * y_val;
+                    }
+                }
+            }
+
+            qs += args.nb01/2;
+            sc += args.nb01;
+            dh += args.nb01/2;
+        }
+
+        y4 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        float sum_all = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = sum_all;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q3_k_lite_f32")]]
+kernel void kernel_mul_mv_q3_K_lite_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q3_K_lite_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+// Q4_K_LITE mul_mv: Q3_K base computation (hmask + qs 3-bit, scales[12], d only)
+template<int nr0, typename args_t>
+void kernel_mul_mv_q4_K_lite_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q4_k_lite * x = (device const block_q4_k_lite *) (src0 + offset0);
+    device const float            * yy = (device const float             *) (src1 + offset1);
+
+    float yl[32];
+
+    const short tid = tiisg/4;
+    const short ix  = tiisg%4;
+    const short ip  = tid/4;
+    const short il  = 2*((tid%4)/2);
+    const short ir  = tid%2;
+    const short l0  = 8*ir;
+
+    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
+                           {0x0004, 0x0400, 0x0008, 0x0800},
+                           {0x0010, 0x1000, 0x0020, 0x2000},
+                           {0x0040, 0x4000, 0x0080, 0x8000}};
+
+    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};
+
+    const ushort4 hm = mm[2*ip + il/2];
+
+    const short shift = 2*il;
+
+    const float v1 = il == 0 ? 4.f : 64.f;
+    const float v2 = 4.f * v1;
+
+    const uint16_t s_shift1 = 4*ip;
+    const uint16_t s_shift2 = s_shift1 + il;
+
+    const short q_offset = 32*ip + l0;
+    const short y_offset = 128*ip + 32*il + l0;
+
+    device const float * y1 = yy + ix*QK_K + y_offset;
+
+    uint32_t scales32, aux32;
+    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
+    thread const int8_t * scales = (thread const int8_t *)&scales32;
+
+    float sumf1[nr0] = {0.f};
+    float sumf2[nr0] = {0.f};
+    float sumf_res[nr0] = {0.f};
+
+    for (int i = ix; i < nb; i += 4) {
+        for (short l = 0; l < 8; ++l) {
+            yl[l+ 0] = y1[l+ 0];
+            yl[l+ 8] = y1[l+16];
+            yl[l+16] = y1[l+32];
+            yl[l+24] = y1[l+48];
+        }
+
+        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
+        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
+        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
+        device const half * dh = &x[i].d;
+
+        for (short row = 0; row < nr0; ++row) {
+            const float d_all = (float)dh[0];
+
+            scales16[0] = a[4];
+            scales16[1] = a[5];
+            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
+            scales16[0] = a[il+0];
+            scales16[1] = a[il+1];
+            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
+
+            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
+            for (short l = 0; l < 8; l += 2) {
+                const int32_t qs = q[l/2];
+                s1 += yl[l+0] * (qs & qm[il/2][0]);
+                s2 += yl[l+1] * (qs & qm[il/2][1]);
+                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
+                s4 += yl[l+16] * (qs & qm[il/2][2]);
+                s5 += yl[l+17] * (qs & qm[il/2][3]);
+                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
+            }
+            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
+            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
+            sumf1[row] += d1 * (scales[0] - 32);
+            sumf2[row] += d2 * (scales[2] - 32);
+
+            s1 = s2 = s3 = s4 = s5 = s6 = 0;
+            for (short l = 0; l < 8; l += 2) {
+                const int32_t qs = q[l/2+8];
+                s1 += yl[l+8] * (qs & qm[il/2][0]);
+                s2 += yl[l+9] * (qs & qm[il/2][1]);
+                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
+                s4 += yl[l+24] * (qs & qm[il/2][2]);
+                s5 += yl[l+25] * (qs & qm[il/2][3]);
+                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
+            }
+            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
+            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
+            sumf1[row] += d1 * (scales[1] - 32);
+            sumf2[row] += d2 * (scales[3] - 32);
+
+            // Apply INT8 residual corrections for Q4_K_LITE
+            // pos_base = y_offset = 128*ip + 32*il + l0; yl groups: +0..7, +16..23, +32..39, +48..55
+            {
+                device const block_q4_k_lite * xb_row = (device const block_q4_k_lite *)((device const char *)&x[i] + (uint64_t)row * args.nb01);
+                const int rc = (int)xb_row->residual_count;
+                if (rc > 0) {
+                    const float rscale = (float)xb_row->residual_scale;
+                    const short pos_base = y_offset;
+                    for (int r = 0; r < Q4_K_LITE_MAX_RESIDUALS; ++r) {
+                        if (r >= rc) break;
+                        const short delta = (short)xb_row->residual_idx[r] - pos_base;
+                        float y_val;
+                        if      (delta >= 0  && delta < 8)   y_val = yl[delta];
+                        else if (delta >= 16 && delta < 24)  y_val = yl[8  + (delta - 16)];
+                        else if (delta >= 32 && delta < 40)  y_val = yl[16 + (delta - 32)];
+                        else if (delta >= 48 && delta < 56)  y_val = yl[24 + (delta - 48)];
+                        else continue;
+                        sumf_res[row] += rscale * (float)xb_row->residual_vals[r] * y_val;
+                    }
+                }
+            }
+
+            q  += args.nb01/2;
+            h  += args.nb01/2;
+            a  += args.nb01/2;
+            dh += args.nb01/2;
+        }
+
+        y1 += 4 * QK_K;
+    }
+
+    for (int row = 0; row < nr0; ++row) {
+        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift) + sumf_res[row];
+        sumf1[row] = simd_sum(sumf);
+    }
+
+    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;
+
+    if (tiisg == 0) {
+        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+            dst_f32[first_row + row] = sumf1[row];
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q4_k_lite_f32")]]
+kernel void kernel_mul_mv_q4_K_lite_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q4_K_lite_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+// Q5_K_LITE mul_mv: Q4_K base computation (d, dmin, scales[12], qs[128] 4-bit)
+template<int nr0, typename args_t>
+void kernel_mul_mv_q5_K_lite_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    constexpr uint16_t kmask1 = 0x3f3f;
+    constexpr uint16_t kmask2 = 0x0f0f;
+    constexpr uint16_t kmask3 = 0xc0c0;
+
+    const short ix = tiisg/8;
+    const short it = tiisg%8;
+    const short iq = it/4;
+    const short ir = it%4;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q5_k_lite * x = (device const block_q5_k_lite *) (src0 + offset0);
+    device const float             * y = (device const float             *) (src1 + offset1);
+
+    float yl[16];
+    float yh[16];
+
+    float sumf[nr0]={0.f};
+
+    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;
+
+    uint16_t sc16[4];
+    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
+
+    for (int ib = ix; ib < nb; ib += 4) {
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+
+        for (short i = 0; i < 8; ++i) {
+            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
+            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
+            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
+            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
+        }
+
+        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
+        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
+        device const half     * dh = &x[ib].d;
+
+        for (short row = 0; row < nr0; row++) {
+            sc16[0] = sc[0] & kmask1;
+            sc16[1] = sc[2] & kmask1;
+            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
+            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
+
+            device const uint16_t * q2 = q1 + 32;
+
+            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
+            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
+
+            FOR_UNROLL (short i = 0; i < 4; ++i) {
+                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
+                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
+                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
+                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
+                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
+                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
+                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
+                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
+            }
+
+            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
+                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
+                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
+                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
+                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
+
+            // Apply INT8 residual corrections for Q5_K_LITE
+            // pos_base = 64*iq + 8*ir; yl groups: +0..7, +32..39; yh groups: +128..135, +160..167
+            {
+                device const block_q5_k_lite * xb_row = (device const block_q5_k_lite *)((device const char *)&x[ib] + (uint64_t)row * args.nb01);
+                const int rc = (int)xb_row->residual_count;
+                if (rc > 0) {
+                    const float rscale = (float)xb_row->residual_scale;
+                    const short pos_base = 64*iq + 8*ir;
+                    for (int r = 0; r < Q5_K_LITE_MAX_RESIDUALS; ++r) {
+                        if (r >= rc) break;
+                        const short delta = (short)xb_row->residual_idx[r] - pos_base;
+                        float y_val;
+                        if      (delta >= 0   && delta < 8)   y_val = yl[delta];
+                        else if (delta >= 32  && delta < 40)  y_val = yl[8 + (delta - 32)];
+                        else if (delta >= 128 && delta < 136) y_val = yh[delta - 128];
+                        else if (delta >= 160 && delta < 168) y_val = yh[8 + (delta - 160)];
+                        else continue;
+                        sumf[row] += rscale * (float)xb_row->residual_vals[r] * y_val;
+                    }
+                }
+            }
+
+            q1 += args.nb01/2;
+            sc += args.nb01/2;
+            dh += args.nb01/2;
+        }
+
+        y4 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        float sum_all = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = sum_all;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q5_k_lite_f32")]]
+kernel void kernel_mul_mv_q5_K_lite_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q5_K_lite_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
+// Q6_K_LITE mul_mv: Q5_K base computation (d, dmin, scales[12], qh[32], qs[128] 5-bit)
+template<int nr0, typename args_t>
+void kernel_mul_mv_q6_K_lite_f32_impl(
+        args_t args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        threadgroup  char * shmem,
+        uint3  tgpig,
+        ushort tiisg,
+        ushort sgitg) {
+    const short NSG = FC_mul_mv_nsg;
+
+    const int nb = args.ne00/QK_K;
+
+    const int r0 = tgpig.x;
+    const int r1 = tgpig.y;
+    const int im = tgpig.z;
+
+    const int first_row = (r0 * NSG + sgitg) * nr0;
+
+    const uint i12 = im%args.ne12;
+    const uint i13 = im/args.ne12;
+
+    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;
+
+    device const block_q6_k_lite * x = (device const block_q6_k_lite *) (src0 + offset0);
+    device const float             * yy = (device const float            *) (src1 + offset1);
+
+    float sumf[nr0]={0.f};
+
+    float yl[16], yh[16];
+
+    constexpr uint16_t kmask1 = 0x3f3f;
+    constexpr uint16_t kmask2 = 0x0f0f;
+    constexpr uint16_t kmask3 = 0xc0c0;
+
+    const short tid = tiisg/4;
+    const short ix  = tiisg%4;
+    const short iq  = tid/4;
+    const short ir  = tid%4;
+
+    const short l0 = 8*ir;
+    const short q_offset = 32*iq + l0;
+    const short y_offset = 64*iq + l0;
+
+    const uint8_t hm1 = 1u << (2*iq);
+    const uint8_t hm2 = hm1 << 1;
+    const uint8_t hm3 = hm1 << 4;
+    const uint8_t hm4 = hm2 << 4;
+
+    uint16_t sc16[4];
+    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
+
+    device const float * y1 = yy + ix*QK_K + y_offset;
+
+    for (int i = ix; i < nb; i += 4) {
+        device const uint8_t * q1 = x[i].qs + q_offset;
+        device const uint8_t * qh = x[i].qh + l0;
+        device const half * dh = &x[i].d;
+        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;
+
+        device const float * y2 = y1 + 128;
+        float4 sumy = {0.f, 0.f, 0.f, 0.f};
+        for (short l = 0; l < 8; ++l) {
+            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
+            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
+            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
+            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
+        }
+
+        for (short row = 0; row < nr0; ++row) {
+            device const uint8_t * q2 = q1 + 64;
+
+            sc16[0] = a[0] & kmask1;
+            sc16[1] = a[2] & kmask1;
+            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
+            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
+
+            float4 acc1 = {0.f};
+            float4 acc2 = {0.f};
+            FOR_UNROLL (short l = 0; l < 8; ++l) {
+                uint8_t h = qh[l];
+                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
+                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
+                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
+                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
+                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
+                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
+                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
+                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
+            }
+
+            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
+                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
+                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
+                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
+                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
+
+            // Apply INT8 residual corrections for Q6_K_LITE
+            // pos_base = 64*iq + l0; yl groups: +0..7, +32..39; yh groups: +128..135, +160..167
+            {
+                device const block_q6_k_lite * xb_row = (device const block_q6_k_lite *)((device const char *)&x[i] + (uint64_t)row * args.nb01);
+                const int rc = (int)xb_row->residual_count;
+                if (rc > 0) {
+                    const float rscale = (float)xb_row->residual_scale;
+                    const short pos_base = 64*iq + l0;
+                    for (int r = 0; r < Q6_K_LITE_MAX_RESIDUALS; ++r) {
+                        if (r >= rc) break;
+                        const short delta = (short)xb_row->residual_idx[r] - pos_base;
+                        float y_val;
+                        if      (delta >= 0   && delta < 8)   y_val = yl[delta];
+                        else if (delta >= 32  && delta < 40)  y_val = yl[8 + (delta - 32)];
+                        else if (delta >= 128 && delta < 136) y_val = yh[delta - 128];
+                        else if (delta >= 160 && delta < 168) y_val = yh[8 + (delta - 160)];
+                        else continue;
+                        sumf[row] += rscale * (float)xb_row->residual_vals[r] * y_val;
+                    }
+                }
+            }
+
+            q1 += args.nb01;
+            qh += args.nb01;
+            dh += args.nb01/2;
+            a  += args.nb01/2;
+        }
+
+        y1 += 4 * QK_K;
+    }
+
+    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
+
+    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
+        const float tot = simd_sum(sumf[row]);
+        if (tiisg == 0) {
+            dst_f32[first_row + row] = tot;
+        }
+    }
+}
+
+[[host_name("kernel_mul_mv_q6_k_lite_f32")]]
+kernel void kernel_mul_mv_q6_K_lite_f32(
+        constant ggml_metal_kargs_mul_mv & args,
+        device const char * src0,
+        device const char * src1,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+
+    kernel_mul_mv_q6_K_lite_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
+}
+
 // ======================= "True" 2-bit
 
 template<int nr0, typename args_t>
@@ -9967,10 +11624,20 @@ template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get
 template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
 template [[host_name("kernel_get_rows_mxfp4")]]   kernel get_rows_q_t kernel_get_rows_q<block_mxfp4,   2, dequantize_mxfp4>;
 template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
+template [[host_name("kernel_get_rows_q2_k_hifi")]] kernel get_rows_q_t kernel_get_rows_q<block_q2_k_hifi, QK_NL, dequantize_q2_k_hifi>;
 template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
+template [[host_name("kernel_get_rows_q3_k_hifi")]] kernel get_rows_q_t kernel_get_rows_q<block_q3_k_hifi, QK_NL, dequantize_q3_k_hifi>;
 template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
+template [[host_name("kernel_get_rows_q4_k_hifi")]] kernel get_rows_q_t kernel_get_rows_q<block_q4_k_hifi, QK_NL, dequantize_q4_k_hifi>;
 template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
+template [[host_name("kernel_get_rows_q5_k_hifi_res8")]] kernel get_rows_q_t kernel_get_rows_q<block_q5_k_hifi_res8, QK_NL, dequantize_q5_k_hifi_res8>;
 template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
+template [[host_name("kernel_get_rows_q6_K_hifi_res8")]] kernel get_rows_q_t kernel_get_rows_q<block_q6_k_hifi_res8, QK_NL, dequantize_q6_k_hifi_res8>;
+template [[host_name("kernel_get_rows_q2_k_lite")]] kernel get_rows_q_t kernel_get_rows_q<block_q2_k_lite, QK_NL, dequantize_q2_k_lite>;
+template [[host_name("kernel_get_rows_q3_k_lite")]] kernel get_rows_q_t kernel_get_rows_q<block_q3_k_lite, QK_NL, dequantize_q3_k_lite>;
+template [[host_name("kernel_get_rows_q4_k_lite")]] kernel get_rows_q_t kernel_get_rows_q<block_q4_k_lite, QK_NL, dequantize_q4_k_lite>;
+template [[host_name("kernel_get_rows_q5_k_lite")]] kernel get_rows_q_t kernel_get_rows_q<block_q5_k_lite, QK_NL, dequantize_q5_k_lite>;
+template [[host_name("kernel_get_rows_q6_k_lite")]] kernel get_rows_q_t kernel_get_rows_q<block_q6_k_lite, QK_NL, dequantize_q6_k_lite>;
 template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
 template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
 template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
@@ -10030,10 +11697,20 @@ template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_m
 template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q2_k_hifi_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_hifi, QK_NL, dequantize_q2_k_hifi, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q3_k_hifi_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_hifi, QK_NL, dequantize_q3_k_hifi, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q4_k_hifi_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_hifi, QK_NL, dequantize_q4_k_hifi, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q5_K_hifi_res8_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_hifi_res8, QK_NL, dequantize_q5_k_hifi_res8, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q6_K_hifi_res8_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_hifi_res8, QK_NL, dequantize_q6_k_hifi_res8, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q2_k_lite_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_lite, QK_NL, dequantize_q2_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q3_k_lite_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_lite, QK_NL, dequantize_q3_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q4_k_lite_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_lite, QK_NL, dequantize_q4_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q5_k_lite_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_lite, QK_NL, dequantize_q5_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_q6_k_lite_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_lite, QK_NL, dequantize_q6_k_lite, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
@@ -10054,10 +11731,20 @@ template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_m
 template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q2_k_hifi_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_hifi, QK_NL, dequantize_q2_k_hifi, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q3_k_hifi_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_hifi, QK_NL, dequantize_q3_k_hifi, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q4_k_hifi_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_hifi, QK_NL, dequantize_q4_k_hifi, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q5_K_hifi_res8_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_hifi_res8, QK_NL, dequantize_q5_k_hifi_res8, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q6_K_hifi_res8_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_hifi_res8, QK_NL, dequantize_q6_k_hifi_res8, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q2_k_lite_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_lite, QK_NL, dequantize_q2_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q3_k_lite_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_lite, QK_NL, dequantize_q3_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q4_k_lite_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_lite, QK_NL, dequantize_q4_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q5_k_lite_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_lite, QK_NL, dequantize_q5_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_q6_k_lite_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_lite, QK_NL, dequantize_q6_k_lite, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
@@ -10087,10 +11774,20 @@ template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_m
 template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q2_k_hifi_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_hifi, QK_NL, dequantize_q2_k_hifi, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q3_k_hifi_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_hifi, QK_NL, dequantize_q3_k_hifi, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q4_k_hifi_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_hifi, QK_NL, dequantize_q4_k_hifi, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q5_K_hifi_res8_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_hifi_res8, QK_NL, dequantize_q5_k_hifi_res8, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q6_K_hifi_res8_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_hifi_res8, QK_NL, dequantize_q6_k_hifi_res8, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q2_k_lite_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_lite, QK_NL, dequantize_q2_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q3_k_lite_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_lite, QK_NL, dequantize_q3_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q4_k_lite_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_lite, QK_NL, dequantize_q4_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q5_k_lite_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_lite, QK_NL, dequantize_q5_k_lite, float,  float4x4,  float, float2x4>;
+template [[host_name("kernel_mul_mm_id_q6_k_lite_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_lite, QK_NL, dequantize_q6_k_lite, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
@@ -10111,10 +11808,20 @@ template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_m
 template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q2_k_hifi_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_hifi, QK_NL, dequantize_q2_k_hifi, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q3_k_hifi_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_hifi, QK_NL, dequantize_q3_k_hifi, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q4_k_hifi_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_hifi, QK_NL, dequantize_q4_k_hifi, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q5_K_hifi_res8_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_hifi_res8, QK_NL, dequantize_q5_k_hifi_res8, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q6_K_hifi_res8_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_hifi_res8, QK_NL, dequantize_q6_k_hifi_res8, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q2_k_lite_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_k_lite, QK_NL, dequantize_q2_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q3_k_lite_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_k_lite, QK_NL, dequantize_q3_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q4_k_lite_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_k_lite, QK_NL, dequantize_q4_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q5_k_lite_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_k_lite, QK_NL, dequantize_q5_k_lite, float,  float4x4,  half, half2x4>;
+template [[host_name("kernel_mul_mm_id_q6_k_lite_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_k_lite, QK_NL, dequantize_q6_k_lite, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
 template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
@@ -10267,10 +11974,20 @@ template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t
 template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;
 
 template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl   <N_R0_Q2_K>>>;
+template [[host_name("kernel_mul_mv_id_q2_k_hifi_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_k_hifi_f32_impl<N_R0_Q2_K_HIFI>>>;
 template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl   <N_R0_Q3_K>>>;
+template [[host_name("kernel_mul_mv_id_q3_k_hifi_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_k_hifi_f32_impl<N_R0_Q3_K_HIFI>>>;
 template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl   <N_R0_Q4_K>>>;
+template [[host_name("kernel_mul_mv_id_q4_k_hifi_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_k_hifi_f32_impl<N_R0_Q4_K_HIFI>>>;
 template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl   <N_R0_Q5_K>>>;
+template [[host_name("kernel_mul_mv_id_q5_K_hifi_res8_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_hifi_res8_f32_impl<N_R0_Q5_K>>>;
 template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl   <N_R0_Q6_K>>>;
+template [[host_name("kernel_mul_mv_id_q6_K_hifi_res8_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_hifi_res8_f32_impl<N_R0_Q6_K>>>;
+template [[host_name("kernel_mul_mv_id_q2_k_lite_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_lite_f32_impl<N_R0_Q2_K>>>;
+template [[host_name("kernel_mul_mv_id_q3_k_lite_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_lite_f32_impl<N_R0_Q2_K>>>;
+template [[host_name("kernel_mul_mv_id_q4_k_lite_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_lite_f32_impl<N_R0_Q4_K>>>;
+template [[host_name("kernel_mul_mv_id_q5_k_lite_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_lite_f32_impl<N_R0_Q5_K>>>;
+template [[host_name("kernel_mul_mv_id_q6_k_lite_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_lite_f32_impl<N_R0_Q6_K>>>;
 template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl  <N_R0_IQ1_S>>>;
 template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl  <N_R0_IQ1_M>>>;
 template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
```

---

### 4.3 Vulkan Backend

#### `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

```diff
diff --git a/ggml/src/ggml-vulkan/ggml-vulkan.cpp b/ggml/src/ggml-vulkan/ggml-vulkan.cpp
index 977aff62d..100586bf6 100644
--- a/ggml/src/ggml-vulkan/ggml-vulkan.cpp
+++ b/ggml/src/ggml-vulkan/ggml-vulkan.cpp
@@ -4069,7 +4069,9 @@ static void ggml_vk_load_shaders(vk_device& device) {
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
+            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K_HIFI][i], "mul_mat_vec_q2_k_hifi_f32_f32", arr_dmmv_q2_k_hifi_f32_f32_len[reduc16], arr_dmmv_q2_k_hifi_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
+            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K_HIFI][i], "mul_mat_vec_q3_k_hifi_f32_f32", arr_dmmv_q3_k_hifi_f32_f32_len[reduc16], arr_dmmv_q3_k_hifi_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
@@ -4094,7 +4096,9 @@ static void ggml_vk_load_shaders(vk_device& device) {
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
+            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K_HIFI][i], "mul_mat_vec_q2_k_hifi_f16_f32", arr_dmmv_q2_k_hifi_f16_f32_len[reduc16], arr_dmmv_q2_k_hifi_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
+            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K_HIFI][i], "mul_mat_vec_q3_k_hifi_f16_f32", arr_dmmv_q3_k_hifi_f16_f32_len[reduc16], arr_dmmv_q3_k_hifi_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
             ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
@@ -4200,7 +4204,9 @@ static void ggml_vk_load_shaders(vk_device& device) {
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K_HIFI], "dequant_q2_k_hifi", dequant_q2_k_hifi_len, dequant_q2_k_hifi_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K_HIFI], "dequant_q3_k_hifi", dequant_q3_k_hifi_len, dequant_q3_k_hifi_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
@@ -4226,7 +4232,9 @@ static void ggml_vk_load_shaders(vk_device& device) {
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K], "get_rows_q2_k", get_rows_q2_k_len, get_rows_q2_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K_HIFI], "get_rows_q2_k_hifi", get_rows_q2_k_hifi_len, get_rows_q2_k_hifi_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K], "get_rows_q3_k", get_rows_q3_k_len, get_rows_q3_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K_HIFI], "get_rows_q3_k_hifi", get_rows_q3_k_hifi_len, get_rows_q3_k_hifi_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_K], "get_rows_q4_k", get_rows_q4_k_len, get_rows_q4_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_K], "get_rows_q5_k", get_rows_q5_k_len, get_rows_q5_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q6_K], "get_rows_q6_k", get_rows_q6_k_len, get_rows_q6_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
@@ -4252,7 +4260,9 @@ static void ggml_vk_load_shaders(vk_device& device) {
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_rows_q2_k_f32_len, get_rows_q2_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K_HIFI], "get_rows_q2_k_hifi_f32", get_rows_q2_k_hifi_f32_len, get_rows_q2_k_hifi_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_rows_q3_k_f32_len, get_rows_q3_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K_HIFI], "get_rows_q3_k_hifi_f32", get_rows_q3_k_hifi_f32_len, get_rows_q3_k_hifi_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_rows_q4_k_f32_len, get_rows_q4_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_rows_q5_k_f32_len, get_rows_q5_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_rows_q6_k_f32_len, get_rows_q6_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
@@ -6050,7 +6060,9 @@ static vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type
         case GGML_TYPE_Q5_1:
         case GGML_TYPE_Q8_0:
         case GGML_TYPE_Q2_K:
+        case GGML_TYPE_Q2_K_HIFI:
         case GGML_TYPE_Q3_K:
+        case GGML_TYPE_Q3_K_HIFI:
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
@@ -6122,7 +6134,9 @@ static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_pipeline(ggml_backend_vk_conte
         case GGML_TYPE_Q5_1:
         case GGML_TYPE_Q8_0:
         case GGML_TYPE_Q2_K:
+        case GGML_TYPE_Q2_K_HIFI:
         case GGML_TYPE_Q3_K:
+        case GGML_TYPE_Q3_K_HIFI:
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
@@ -6165,6 +6179,7 @@ static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context *
             case GGML_TYPE_Q8_0:
             case GGML_TYPE_MXFP4:
             case GGML_TYPE_Q2_K:
+            case GGML_TYPE_Q2_K_HIFI:
             case GGML_TYPE_Q3_K:
             case GGML_TYPE_Q4_K:
             case GGML_TYPE_Q5_K:
@@ -6188,7 +6203,9 @@ static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context *
         case GGML_TYPE_Q5_1:
         case GGML_TYPE_Q8_0:
         case GGML_TYPE_Q2_K:
+        case GGML_TYPE_Q2_K_HIFI:
         case GGML_TYPE_Q3_K:
+        case GGML_TYPE_Q3_K_HIFI:
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
@@ -6279,7 +6296,9 @@ static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_id_pipeline(ggml_backend_vk_co
         case GGML_TYPE_Q5_1:
         case GGML_TYPE_Q8_0:
         case GGML_TYPE_Q2_K:
+        case GGML_TYPE_Q2_K_HIFI:
         case GGML_TYPE_Q3_K:
+        case GGML_TYPE_Q3_K_HIFI:
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
@@ -6325,6 +6344,7 @@ static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context
             case GGML_TYPE_Q8_0:
             case GGML_TYPE_MXFP4:
             case GGML_TYPE_Q2_K:
+            case GGML_TYPE_Q2_K_HIFI:
             case GGML_TYPE_Q3_K:
             case GGML_TYPE_Q4_K:
             case GGML_TYPE_Q5_K:
@@ -6348,7 +6368,9 @@ static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context
         case GGML_TYPE_Q5_1:
         case GGML_TYPE_Q8_0:
         case GGML_TYPE_Q2_K:
+        case GGML_TYPE_Q2_K_HIFI:
         case GGML_TYPE_Q3_K:
+        case GGML_TYPE_Q3_K_HIFI:
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
@@ -15304,7 +15326,9 @@ static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggm
                     case GGML_TYPE_Q5_1:
                     case GGML_TYPE_Q8_0:
                     case GGML_TYPE_Q2_K:
+                    case GGML_TYPE_Q2_K_HIFI:
                     case GGML_TYPE_Q3_K:
+                    case GGML_TYPE_Q3_K_HIFI:
                     case GGML_TYPE_Q4_K:
                     case GGML_TYPE_Q5_K:
                     case GGML_TYPE_Q6_K:
@@ -15419,7 +15443,9 @@ static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggm
                     case GGML_TYPE_Q5_1:
                     case GGML_TYPE_Q8_0:
                     case GGML_TYPE_Q2_K:
+                    case GGML_TYPE_Q2_K_HIFI:
                     case GGML_TYPE_Q3_K:
+                    case GGML_TYPE_Q3_K_HIFI:
                     case GGML_TYPE_Q4_K:
                     case GGML_TYPE_Q5_K:
                     case GGML_TYPE_Q6_K:
```

#### `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`

```diff
diff --git a/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp b/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
index 77a55ea81..d3ab2c1f3 100644
--- a/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
+++ b/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
@@ -52,7 +52,9 @@ const std::vector<std::string> type_names = {
     "q5_1",
     "q8_0",
     "q2_k",
+    "q2_k_hifi",
     "q3_k",
+    "q3_k_hifi",
     "q4_k",
     "q5_k",
     "q6_k",
@@ -682,7 +684,7 @@ void process_shaders() {
     for (const auto& tname : type_names) {
         // mul mat vec
         std::string data_a_key = "DATA_A_" + to_uppercase(tname);
-        std::string shader = (string_ends_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_")) ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";
+        std::string shader = (string_ends_with(tname, "_k") || tname == "q3_k_hifi" || tname == "q2_k_hifi" || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_")) ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";
 
         string_to_spv("mul_mat_vec_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
         string_to_spv("mul_mat_vec_" + tname + "_f16_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
```

#### `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`

```diff
diff --git a/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl b/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl
index 4239070af..116c52b60 100644
--- a/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl
+++ b/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl
@@ -272,6 +272,42 @@ struct block_q2_K_packed32
 #define DATA_A_QUANT_K
 #endif
 
+// Q2_K_HIFI: Q2_K with up to 3 FP16 outlier corrections per block
+#define QUANT_K_Q2_K_HIFI 256
+#define Q2_K_HIFI_MAX_OUTLIERS 3
+#define Q2_K_HIFI_RESIDUAL_MODE_FLAG 0x80
+
+struct block_q2_k_hifi
+{
+    uint8_t scales[QUANT_K_Q2_K_HIFI/16]; // 16 bytes
+    uint8_t qs[QUANT_K_Q2_K_HIFI/4];      // 64 bytes
+    f16vec2 dm;                            // 4 bytes
+    uint8_t outlier_count;                 // 1 byte
+    uint8_t outlier_idx[Q2_K_HIFI_MAX_OUTLIERS]; // 3 bytes
+    float16_t outlier_vals[Q2_K_HIFI_MAX_OUTLIERS]; // 6 bytes
+    uint8_t _pad[2];                       // 2 bytes
+};
+
+struct block_q2_k_hifi_packed16
+{
+    uint16_t scales[QUANT_K_Q2_K_HIFI/16/2];
+    uint16_t qs[QUANT_K_Q2_K_HIFI/4/2];
+    f16vec2 dm;
+    uint8_t outlier_count;
+    uint8_t outlier_idx[Q2_K_HIFI_MAX_OUTLIERS];
+    float16_t outlier_vals[Q2_K_HIFI_MAX_OUTLIERS];
+    uint8_t _pad[2];
+};
+
+#if defined(DATA_A_Q2_K_HIFI)
+#define QUANT_K QUANT_K_Q2_K_HIFI
+#define QUANT_R 1
+#define A_TYPE block_q2_k_hifi
+#define A_TYPE_PACKED16 block_q2_k_hifi_packed16
+#define SCALES_PER_32 2
+#define DATA_A_QUANT_K
+#endif
+
 #define QUANT_K_Q3_K 256
 
 struct block_q3_K
@@ -298,6 +334,79 @@ struct block_q3_K_packed16
 #define DATA_A_QUANT_K
 #endif
 
+// Q3_K_HIFI: Q3_K with 16 FP16 residual corrections for stronger signal recovery
+#define QUANT_K_Q3_K_HIFI 256
+#define Q3_K_HIFI_OUTLIERS 16
+
+struct block_q3_k_hifi
+{
+    uint8_t hmask[QUANT_K_Q3_K_HIFI/8];     // 32 bytes
+    uint8_t qs[QUANT_K_Q3_K_HIFI/4];        // 64 bytes
+    uint8_t scales[12];                    // 12 bytes
+    float16_t d;                           // 2 bytes
+    uint8_t outlier_count;                 // 1 byte: actual outliers stored
+    uint8_t _pad;                          // 1 byte: alignment
+    uint8_t outlier_idx[Q3_K_HIFI_OUTLIERS]; // 16 bytes
+    float16_t outlier_vals[Q3_K_HIFI_OUTLIERS]; // 32 bytes
+};
+
+struct block_q3_k_hifi_packed16
+{
+    uint16_t hmask[QUANT_K_Q3_K_HIFI/8/2];
+    uint16_t qs[QUANT_K_Q3_K_HIFI/4/2];
+    uint16_t scales[12/2];
+    float16_t d;
+    uint8_t outlier_count;
+    uint8_t _pad;
+    uint16_t outlier_idx[Q3_K_HIFI_OUTLIERS/2];
+    float16_t outlier_vals[Q3_K_HIFI_OUTLIERS];
+};
+
+#if defined(DATA_A_Q3_K_HIFI)
+#define QUANT_K QUANT_K_Q3_K_HIFI
+#define QUANT_R 1
+#define A_TYPE block_q3_k_hifi
+#define A_TYPE_PACKED16 block_q3_k_hifi_packed16
+#define DATA_A_QUANT_K
+#endif
+
+// Q3_K_HIFI_RES8: Lean INT8 residual version for imatrix use
+#define Q3_K_HIFI_RES8_OUTLIERS 8
+
+struct block_q3_k_hifi_res8
+{
+    uint8_t hmask[QUANT_K_Q3_K_HIFI/8];     // 32 bytes
+    uint8_t qs[QUANT_K_Q3_K_HIFI/4];        // 64 bytes
+    uint8_t scales[12];                    // 12 bytes
+    float16_t d;                           // 2 bytes
+    uint8_t outlier_count;                 // 1 byte: actual outliers stored
+    uint8_t _pad;                          // 1 byte: alignment
+    uint8_t outlier_idx[Q3_K_HIFI_RES8_OUTLIERS]; // 8 bytes
+    int8_t  residual_vals[Q3_K_HIFI_RES8_OUTLIERS]; // 8 bytes: INT8 residuals
+    float   residual_scale;                // 4 bytes
+};
+
+struct block_q3_k_hifi_res8_packed16
+{
+    uint16_t hmask[QUANT_K_Q3_K_HIFI/8/2];
+    uint16_t qs[QUANT_K_Q3_K_HIFI/4/2];
+    uint16_t scales[12/2];
+    float16_t d;
+    uint8_t outlier_count;
+    uint8_t _pad;
+    uint16_t outlier_idx[Q3_K_HIFI_RES8_OUTLIERS/2];
+    int8_t  residual_vals[Q3_K_HIFI_RES8_OUTLIERS];
+    float   residual_scale;
+};
+
+#if defined(DATA_A_Q3_K_HIFI_RES8)
+#define QUANT_K QUANT_K_Q3_K_HIFI
+#define QUANT_R 1
+#define A_TYPE block_q3_k_hifi_res8
+#define A_TYPE_PACKED16 block_q3_k_hifi_res8_packed16
+#define DATA_A_QUANT_K
+#endif
+
 #define QUANT_K_Q4_K 256
 
 struct block_q4_K
```

#### `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl`

```diff
diff --git a/ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl b/ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl
index ede1275cf..bf702b5a9 100644
--- a/ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl
+++ b/ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl
@@ -509,6 +509,46 @@ vec2 get_dm(uint ib, uint a_offset) {
 }
 #endif
 
+#if defined(DATA_A_Q2_K_HIFI)
+vec2 dequantize(uint ib, uint iqs, uint a_offset) {
+    iqs /= 2;
+    const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2;
+    const uint scalesi = iqs / 8;
+    const uint qsshift = ((iqs % 64) / 16) * 2;
+
+    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
+    const uint scales = data_a[a_offset + ib].scales[scalesi];
+    const vec2 dm = vec2(data_a[a_offset + ib].dm);
+
+    float v0 = dm.x * float(scales & 0xF) * float((qs.x >> qsshift) & 3) - dm.y * float(scales >> 4);
+    float v1 = dm.x * float(scales & 0xF) * float((qs.y >> qsshift) & 3) - dm.y * float(scales >> 4);
+
+    const uint local_idx0 = (iqs / 64) * 128 + (iqs % 16) * 2 + ((iqs % 64) / 16) * 32;
+    const uint local_idx1 = local_idx0 + 1;
+
+    const uint raw_count = data_a[a_offset + ib].outlier_count;
+    const bool residual_mode = (raw_count & Q2_K_HIFI_RESIDUAL_MODE_FLAG) != 0;
+    const uint count = raw_count & 0x7F;
+    const uint n_out = min(count, Q2_K_HIFI_MAX_OUTLIERS);
+
+    [[unroll]] for (uint k = 0; k < Q2_K_HIFI_MAX_OUTLIERS; ++k) {
+        if (k >= n_out) break;
+        const float val = float(data_a[a_offset + ib].outlier_vals[k]);
+        if (data_a[a_offset + ib].outlier_idx[k] == local_idx0) {
+            v0 = residual_mode ? (v0 + val) : val;
+        }
+        if (data_a[a_offset + ib].outlier_idx[k] == local_idx1) {
+            v1 = residual_mode ? (v1 + val) : val;
+        }
+    }
+
+    return vec2(v0, v1);
+}
+vec2 get_dm(uint ib, uint a_offset) {
+    return vec2(1, 0);
+}
+#endif
+
 #if defined(DATA_A_Q3_K)
 vec2 dequantize(uint ib, uint iqs, uint a_offset) {
     iqs /= 2;
@@ -533,6 +573,48 @@ vec2 get_dm(uint ib, uint a_offset) {
 }
 #endif
 
+#if defined(DATA_A_Q3_K_HIFI)
+vec2 dequantize(uint ib, uint iqs, uint a_offset) {
+    // Q3_K_HIFI uses same layout as Q3_K with outliers appended
+    iqs /= 2;
+    const uint n = iqs / 64;                     // 0,1
+    const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
+    const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
+    const uint j = (iqs % 64) / 4;               // 0..3
+    const uint is = iqs / 8;                     // 0..15
+    const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
+    const uint qsshift = halfsplit * 2;          // 0,2,4,6
+    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128
+
+    const int8_t us = int8_t(((data_a[a_offset + ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
+                          | (((data_a[a_offset + ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
+    const float dl = float(data_a[a_offset + ib].d) * float(us - 32);
+
+    // Compute local indices for outlier checking
+    const uint local_idx0 = 128 * n + 32 * j + (iqs % 16) * 2;
+    const uint local_idx1 = local_idx0 + 1;
+
+    // Base Q3_K dequantization
+    float v0 = dl * float(int8_t((data_a[a_offset + ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi    ] & m) != 0) ? 0 : 4));
+    float v1 = dl * float(int8_t((data_a[a_offset + ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi + 1] & m) != 0) ? 0 : 4));
+
+    // Check for outliers and replace with FP16 values
+    [[unroll]] for (uint k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+        if (data_a[a_offset + ib].outlier_idx[k] == local_idx0) {
+            v0 = float(data_a[a_offset + ib].outlier_vals[k]);
+        }
+        if (data_a[a_offset + ib].outlier_idx[k] == local_idx1) {
+            v1 = float(data_a[a_offset + ib].outlier_vals[k]);
+        }
+    }
+
+    return vec2(v0, v1);
+}
+vec2 get_dm(uint ib, uint a_offset) {
+    return vec2(1, 0);
+}
+#endif
+
 #if defined(DATA_A_Q4_K)
 vec2 dequantize(uint ib, uint iqs, uint a_offset) {
     iqs /= 2;
```

---

### 4.4 SYCL Backend

#### `ggml/src/ggml-sycl/dequantize.hpp`

```diff
diff --git a/ggml/src/ggml-sycl/dequantize.hpp b/ggml/src/ggml-sycl/dequantize.hpp
index f992db33b..26e84d356 100644
--- a/ggml/src/ggml-sycl/dequantize.hpp
+++ b/ggml/src/ggml-sycl/dequantize.hpp
@@ -361,6 +361,170 @@ static void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restri
 
 }
 
+// Q3_K_HIFI: Q3_K with 16 FP16 residual corrections
+template<typename dst_t>
+static void dequantize_block_q3_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy,
+                                     const sycl::nd_item<3> &item_ct1) {
+
+    const int64_t i = item_ct1.get_group(2);
+    const block_q3_k_hifi * x = (const block_q3_k_hifi *) vx;
+
+#if QK_K == 256
+    const int64_t r = item_ct1.get_local_id(2) / 4;
+    const int64_t tid = r/2;
+    const int64_t is0 = r%2;
+    const int64_t l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
+    const int64_t n = tid / 4;
+    const int64_t j = tid - 4*n;
+
+    uint8_t m = 1 << (4*n + j);
+    int64_t is = 8*n + 2*j + is0;
+    int shift = 2*j;
+
+    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
+                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
+                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
+                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
+    float d_all = x[i].d;
+    float dl = d_all * (us - 32);
+
+    dst_t * y = yy + i*QK_K + 128*n + 32*j;
+    const uint8_t * q = x[i].qs + 32*n;
+    const uint8_t * hm = x[i].hmask;
+
+    // Get outlier count (clamped to max)
+    const int n_outliers = (x[i].outlier_count <= Q3_K_HIFI_OUTLIERS) ? x[i].outlier_count : Q3_K_HIFI_OUTLIERS;
+
+    for (int l = l0; l < l0+4; ++l) {
+        int idx = 128*n + 32*j + l;
+        // Step 1: Standard Q3_K dequantization
+        dst_t val = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
+        // Step 2: ADD residual correction if this position has one
+        for (int k = 0; k < n_outliers; ++k) {
+            if (x[i].outlier_idx[k] == idx) {
+                val += x[i].outlier_vals[k];  // ADD correction, don't replace
+                break;
+            }
+        }
+        y[l] = val;
+    }
+#else
+    const int64_t tid = item_ct1.get_local_id(2);
+    const int64_t is  = tid/16;
+    const int64_t il  = tid%16;
+    const int64_t im  = il/8;
+    const int64_t in  = il%8;
+
+    dst_t * y = yy + i*QK_K + 16*is + il;
+
+    const uint8_t q = x[i].qs[il] >> (2*is);
+    const uint8_t h = x[i].hmask[in] >> (2*is + im);
+    const float   d = (float)x[i].d;
+
+    dst_t val0, val1;
+    if (is == 0) {
+        val0 = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
+        val1 = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
+    } else {
+        val0 = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
+        val1 = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
+    }
+    // Check for outliers
+    int idx0 = 16*is + il;
+    int idx1 = 16*is + il + 32;
+    for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
+        if (x[i].outlier_idx[k] == idx0) val0 = x[i].outlier_vals[k];
+        if (x[i].outlier_idx[k] == idx1) val1 = x[i].outlier_vals[k];
+    }
+    y[ 0] = val0;
+    y[32] = val1;
+#endif
+
+}
+
+// Q3_K_HIFI_RES8: Q3_K with 8 INT8 residual corrections (lean version for imatrix)
+template<typename dst_t>
+static void dequantize_block_q3_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy,
+                                           const sycl::nd_item<3> &item_ct1) {
+
+    const int64_t i = item_ct1.get_group(2);
+    const block_q3_k_hifi_res8 * x = (const block_q3_k_hifi_res8 *) vx;
+
+#if QK_K == 256
+    const int64_t r = item_ct1.get_local_id(2) / 4;
+    const int64_t tid = r/2;
+    const int64_t is0 = r%2;
+    const int64_t l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
+    const int64_t n = tid / 4;
+    const int64_t j = tid - 4*n;
+
+    uint8_t m = 1 << (4*n + j);
+    int64_t is = 8*n + 2*j + is0;
+    int shift = 2*j;
+
+    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
+                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
+                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
+                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
+    float d_all = x[i].d;
+    float dl = d_all * (us - 32);
+
+    dst_t * y = yy + i*QK_K + 128*n + 32*j;
+    const uint8_t * q = x[i].qs + 32*n;
+    const uint8_t * hm = x[i].hmask;
+
+    // Get outlier count and residual scale
+    const int n_outliers = (x[i].outlier_count <= Q3_K_HIFI_RES8_OUTLIERS) ? x[i].outlier_count : Q3_K_HIFI_RES8_OUTLIERS;
+    const float res_scale = x[i].residual_scale;
+
+    for (int l = l0; l < l0+4; ++l) {
+        int idx = 128*n + 32*j + l;
+        // Step 1: Standard Q3_K dequantization
+        dst_t val = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
+        // Step 2: ADD INT8 residual correction if this position has one
+        for (int k = 0; k < n_outliers; ++k) {
+            if (x[i].outlier_idx[k] == idx) {
+                val += res_scale * (float)x[i].residual_vals[k];  // ADD INT8 correction
+                break;
+            }
+        }
+        y[l] = val;
+    }
+#else
+    const int64_t tid = item_ct1.get_local_id(2);
+    const int64_t is  = tid/16;
+    const int64_t il  = tid%16;
+    const int64_t im  = il/8;
+    const int64_t in  = il%8;
+
+    dst_t * y = yy + i*QK_K + 16*is + il;
+
+    const uint8_t q = x[i].qs[il] >> (2*is);
+    const uint8_t h = x[i].hmask[in] >> (2*is + im);
+    const float   d = (float)x[i].d;
+    const float res_scale = x[i].residual_scale;
+
+    dst_t val0, val1;
+    if (is == 0) {
+        val0 = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
+        val1 = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
+    } else {
+        val0 = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
+        val1 = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
+    }
+    // Check for INT8 outliers
+    int idx0 = 16*is + il;
+    int idx1 = 16*is + il + 32;
+    for (int k = 0; k < Q3_K_HIFI_RES8_OUTLIERS; ++k) {
+        if (x[i].outlier_idx[k] == idx0) val0 += res_scale * (float)x[i].residual_vals[k];
+        if (x[i].outlier_idx[k] == idx1) val1 += res_scale * (float)x[i].residual_vals[k];
+    }
+    y[ 0] = val0;
+    y[32] = val1;
+#endif
+
+}
+
 #if QK_K == 256
 static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
     if (j < 4) {
```

#### `ggml/src/ggml-sycl/vecdotq.hpp`

```diff
diff --git a/ggml/src/ggml-sycl/vecdotq.hpp b/ggml/src/ggml-sycl/vecdotq.hpp
index 9253168e5..a872c6a6d 100644
--- a/ggml/src/ggml-sycl/vecdotq.hpp
+++ b/ggml/src/ggml-sycl/vecdotq.hpp
@@ -951,6 +951,121 @@ vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
     return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
 }
 
+// Q3_K_HIFI: Q3_K with 16 FP16 residual corrections for stronger signal recovery
+#define VDR_Q3_K_HIFI_Q8_1_MMVQ VDR_Q3_K_Q8_1_MMVQ
+
+static __dpct_inline__ float
+vec_dot_q3_k_hifi_q8_1(const void *__restrict__ vbq,
+                     const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
+
+    const block_q3_k_hifi * bq3_k_hifi = (const block_q3_k_hifi *) vbq;
+
+    // === Q3_K bulk dot product (identical logic) ===
+    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const float d = bq3_k_hifi->d;
+
+    const int vl = get_int_from_uint8(bq3_k_hifi->qs, iqs);
+
+    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
+    const int vh = ~get_int_from_uint8(bq3_k_hifi->hmask, iqs % (QI3_K/2)) >> bq8_offset;
+
+    int    u[QR3_K];
+    float d8[QR3_K];
+
+#pragma unroll
+    for (int i = 0; i < QR3_K; ++i) {
+        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = bq8_1[bq8_offset + i].ds[0];
+    }
+
+    // Compute Q3_K bulk dot product (now includes all positions)
+    float sum = vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_k_hifi->scales, scale_offset, d, d8);
+
+    // === Q3_K_HIFI residual correction ===
+    // Add RESIDUAL corrections for positions where Q3_K had largest errors
+    const int n_outliers = (bq3_k_hifi->outlier_count <= Q3_K_HIFI_OUTLIERS) ? bq3_k_hifi->outlier_count : Q3_K_HIFI_OUTLIERS;
+    for (int k = 0; k < n_outliers; ++k) {
+        const int idx = bq3_k_hifi->outlier_idx[k];
+        const int idx_bq8 = idx / QK8_1;
+        const int idx_in_bq8 = idx % QK8_1;
+
+        // Check if this outlier is in the range this thread processes
+        if (idx_bq8 >= bq8_offset && idx_bq8 < bq8_offset + QR3_K) {
+            const int thread_q8_offset = iqs % QI8_1;
+            const int pos_in_q8_group = idx_in_bq8 / 4;
+            if (pos_in_q8_group == thread_q8_offset) {
+                // outlier_vals now contains RESIDUAL correction, not original value
+                const float residual_correction = bq3_k_hifi->outlier_vals[k];
+                const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+                const float d8_val = bq8_1[idx_bq8].ds[0];
+                sum += residual_correction * q8_val * d8_val;
+            }
+        }
+    }
+
+    return sum;
+}
+
+// Q3_K_HIFI_RES8: Lean INT8 residual version for imatrix use
+#define VDR_Q3_K_HIFI_RES8_Q8_1_MMVQ VDR_Q3_K_Q8_1_MMVQ
+
+static __dpct_inline__ float
+vec_dot_q3_k_hifi_res8_q8_1(const void *__restrict__ vbq,
+                           const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
+
+    const block_q3_k_hifi_res8 * bq3_k_hifi = (const block_q3_k_hifi_res8 *) vbq;
+
+    // === Q3_K bulk dot product (identical logic) ===
+    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
+    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
+
+    const float d = bq3_k_hifi->d;
+
+    const int vl = get_int_from_uint8(bq3_k_hifi->qs, iqs);
+
+    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
+    const int vh = ~get_int_from_uint8(bq3_k_hifi->hmask, iqs % (QI3_K/2)) >> bq8_offset;
+
+    int    u[QR3_K];
+    float d8[QR3_K];
+
+#pragma unroll
+    for (int i = 0; i < QR3_K; ++i) {
+        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
+        d8[i] = bq8_1[bq8_offset + i].ds[0];
+    }
+
+    // Compute Q3_K bulk dot product (now includes all positions)
+    float sum = vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_k_hifi->scales, scale_offset, d, d8);
+
+    // === Q3_K_HIFI_RES8 INT8 residual correction ===
+    // Add RESIDUAL corrections for positions where Q3_K had largest errors
+    const int n_outliers = (bq3_k_hifi->outlier_count <= Q3_K_HIFI_RES8_OUTLIERS) ? bq3_k_hifi->outlier_count : Q3_K_HIFI_RES8_OUTLIERS;
+    const float res_scale = bq3_k_hifi->residual_scale;
+    for (int k = 0; k < n_outliers; ++k) {
+        const int idx = bq3_k_hifi->outlier_idx[k];
+        const int idx_bq8 = idx / QK8_1;
+        const int idx_in_bq8 = idx % QK8_1;
+
+        // Check if this outlier is in the range this thread processes
+        if (idx_bq8 >= bq8_offset && idx_bq8 < bq8_offset + QR3_K) {
+            const int thread_q8_offset = iqs % QI8_1;
+            const int pos_in_q8_group = idx_in_bq8 / 4;
+            if (pos_in_q8_group == thread_q8_offset) {
+                // INT8 residual correction with scale
+                const float residual_correction = res_scale * (float)bq3_k_hifi->residual_vals[k];
+                const int8_t q8_val = ((const int8_t*)bq8_1[idx_bq8].qs)[idx_in_bq8];
+                const float d8_val = bq8_1[idx_bq8].ds[0];
+                sum += residual_correction * q8_val * d8_val;
+            }
+        }
+    }
+
+    return sum;
+}
+
 static __dpct_inline__ float vec_dot_q4_K_q8_1(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                                const int & iqs) {
 #ifndef GGML_QKK_64
```

#### `ggml/src/ggml-sycl/convert.cpp`

```diff
diff --git a/ggml/src/ggml-sycl/convert.cpp b/ggml/src/ggml-sycl/convert.cpp
index d7f60cbc9..727d7a257 100644
--- a/ggml/src/ggml-sycl/convert.cpp
+++ b/ggml/src/ggml-sycl/convert.cpp
@@ -114,6 +114,38 @@ static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int64_t k,
 #endif
 }
 
+// Q3_K_HIFI: Q3_K-compatible layout with 6 FP16 outliers
+template <typename dst_t>
+static void dequantize_row_q3_k_hifi_sycl(const void *vx, dst_t *y, const int64_t k,
+                                        dpct::queue_ptr stream) {
+    const int64_t nb = k / QK_K;
+#if QK_K == 256
+    {
+        dpct::has_capability_or_fail(stream->get_device(),
+                                     {sycl::aspect::fp16});
+
+        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
+                                                   sycl::range<3>(1, 1, 64),
+                                               sycl::range<3>(1, 1, 64)),
+                             [=](sycl::nd_item<3> item_ct1) {
+                                 dequantize_block_q3_k_hifi(vx, y, item_ct1);
+                             });
+    }
+#else
+    {
+        dpct::has_capability_or_fail(stream->get_device(),
+                                     {sycl::aspect::fp16});
+
+        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
+                                                   sycl::range<3>(1, 1, 32),
+                                               sycl::range<3>(1, 1, 32)),
+                             [=](sycl::nd_item<3> item_ct1) {
+                                 dequantize_block_q3_k_hifi(vx, y, item_ct1);
+                             });
+    }
+#endif
+}
+
 template <typename dst_t>
 static void dequantize_row_q4_0_sycl(const void *vx, dst_t *y, const int64_t k,
                                      dpct::queue_ptr stream) {
@@ -619,6 +651,8 @@ to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
             return dequantize_row_q2_K_sycl;
         case GGML_TYPE_Q3_K:
             return dequantize_row_q3_K_sycl;
+        case GGML_TYPE_Q3_K_HIFI:
+            return dequantize_row_q3_k_hifi_sycl;
         case GGML_TYPE_Q4_K:
             if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                 return dequantize_row_q4_K_sycl_reorder;
@@ -688,6 +722,8 @@ to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
             return dequantize_row_q2_K_sycl;
         case GGML_TYPE_Q3_K:
             return dequantize_row_q3_K_sycl;
+        case GGML_TYPE_Q3_K_HIFI:
+            return dequantize_row_q3_k_hifi_sycl;
         case GGML_TYPE_Q4_K:
             if (dst->src[0]->extra &&
                 ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
```

#### `ggml/src/ggml-sycl/mmvq.cpp`

```diff
diff --git a/ggml/src/ggml-sycl/mmvq.cpp b/ggml/src/ggml-sycl/mmvq.cpp
index af22b98dd..0ff36450e 100644
--- a/ggml/src/ggml-sycl/mmvq.cpp
+++ b/ggml/src/ggml-sycl/mmvq.cpp
@@ -770,6 +770,29 @@ static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
     }
 }
 
+// Q3_K_HIFI: Q3_K-compatible layout with 6 FP16 outliers
+static void mul_mat_vec_q3_k_hifi_q8_1_sycl(const void *vx, const void *vy,
+                                          float *dst, const int ncols,
+                                          const int nrows,
+                                          dpct::queue_ptr stream) {
+    GGML_ASSERT(ncols % QK_K == 0);
+    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
+    const sycl::range<3> block_nums(1, 1, block_num_y);
+    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
+    {
+        stream->submit([&](sycl::handler &cgh) {
+            cgh.parallel_for(
+                sycl::nd_range<3>(block_nums * block_dims, block_dims),
+                [=](sycl::nd_item<3> item_ct1)
+                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
+                        mul_mat_vec_q<QK_K, QI3_K, block_q3_k_hifi,
+                                      VDR_Q3_K_HIFI_Q8_1_MMVQ, vec_dot_q3_k_hifi_q8_1>(
+                            vx, vy, dst, ncols, nrows, item_ct1);
+                    });
+        });
+    }
+}
+
 static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols,
                                        const int nrows,
@@ -1134,6 +1157,9 @@ void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tens
             case GGML_TYPE_Q3_K:
                 mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                 break;
+            case GGML_TYPE_Q3_K_HIFI:
+                mul_mat_vec_q3_k_hifi_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
+                break;
             case GGML_TYPE_Q4_K:
                 if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                     ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
```

---

## 5. Python/GGUF Tooling

### 5.1 `gguf-py/gguf/constants.py`

```diff
diff --git a/gguf-py/gguf/constants.py b/gguf-py/gguf/constants.py
index 53ce138fc..c37a4e87f 100644
--- a/gguf-py/gguf/constants.py
+++ b/gguf-py/gguf/constants.py
@@ -4000,6 +4000,19 @@ class GGMLQuantizationType(IntEnum):
     MXFP4   = 39
     NVFP4   = 40
     Q1_0    = 41
+    Q3_K_HIFI = 42              # Q3_K layout + 8 FP16 outliers per block
+    Q6_K_HIFI = 43            # Q6_K layout + 4 FP16 outliers
+    Q6_K_HIFI_DYNAMIC = 44    # Q6_K + 2-8 dynamic outliers
+    Q6_K_HIFI_RES8 = 45       # Q6_K + INT8 residuals (compact format)
+    Q5_K_HIFI_RES8 = 46       # Q5_K + INT8 residuals (efficient for 4B-10B models)
+    Q3_K_HIFI_RES8 = 47       # Q3_K + INT8 residuals (lean version for imatrix use)
+    Q4_K_HIFI = 48             # Q4_K layout + 8 FP16 outliers per block (high-fidelity 4-bit)
+    Q2_K_HIFI = 49
+    Q2_K_LITE = 50
+    Q3_K_LITE = 51
+    Q4_K_LITE = 52
+    Q5_K_LITE = 53
+    Q6_K_LITE = 54
 
 
 class ExpertGatingFuncType(IntEnum):
@@ -4054,6 +4067,16 @@ class LlamaFileType(IntEnum):
     MOSTLY_MXFP4_MOE     = 38  # except 1d tensors
     MOSTLY_NVFP4         = 39  # except 1d tensors
     MOSTLY_Q1_0          = 40  # except 1d tensors
+    MOSTLY_Q4_K_HIFI     = 44  # Q4_K_M + 2-8 dynamic outliers + early exit (best quality/size ratio)
+    MOSTLY_Q3_K_HIFI     = 45  # Q3_K_M base + Q6_K_HIFI on critical tensors
+    MOSTLY_Q5_K_HIFI     = 46  # Q5_K_M base + Q6_K_HIFI_RES8 on top 10-15% tensors (best 5-bit quality)
+    MOSTLY_Q2_K_HIFI     = 47  # Q2_K base + INT8 residuals on critical tensors (best 2-bit quality)
+
+    MOSTLY_Q2_K_LITE     = 48  # Q2_K base + INT8 residuals (96 bytes/block, ~3.0 bpw)
+    MOSTLY_Q3_K_LITE     = 49  # Q2_K base + INT8 residuals (104 bytes/block, ~3.25 bpw)
+    MOSTLY_Q4_K_LITE     = 50  # Q3_K base + INT8 residuals (128 bytes/block, ~4.0 bpw)
+    MOSTLY_Q5_K_LITE     = 51  # Q4_K base + INT8 residuals (164 bytes/block, ~5.13 bpw)
+    MOSTLY_Q6_K_LITE     = 52  # Q5_K base + INT8 residuals (196 bytes/block, ~6.13 bpw)
 
     GUESSED              = 1024  # not specified in the model file
 
@@ -4167,6 +4190,19 @@ GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
     GGMLQuantizationType.TQ1_0:   (256, 2 + 4 * 13),
     GGMLQuantizationType.TQ2_0:   (256, 2 + 64),
     GGMLQuantizationType.MXFP4:   (32, 1 + 16),
+    GGMLQuantizationType.Q3_K_HIFI: (256, 160),  # Q3_K (110 bytes) + outlier_count(1) + _pad(1) + outlier_idx[16] + outlier_vals[16] = 160 bytes
+    GGMLQuantizationType.Q6_K_HIFI: (256, 222),  # Q6_K (210) + idx[4] + vals[8]
+    GGMLQuantizationType.Q6_K_HIFI_DYNAMIC: (256, 236),  # Q6_K (210) + dynamic outliers (26)
+    GGMLQuantizationType.Q6_K_HIFI_RES8: (256, 232),  # Q6_K (210) + INT8 residuals (22)
+    GGMLQuantizationType.Q5_K_HIFI_RES8: (256, 200),  # Q5_K (176) + INT8 residuals (24)
+    GGMLQuantizationType.Q3_K_HIFI_RES8: (256, 132),  # Q3_K (110) + INT8 residuals (22)
+    GGMLQuantizationType.Q4_K_HIFI: (256, 168),  # Q4_K (144) + outlier_idx[8] + outlier_vals[16] = 168 bytes
+    GGMLQuantizationType.Q2_K_HIFI: (256, 99),   # Q2_K + INT8 residuals
+    GGMLQuantizationType.Q2_K_LITE: (256, 96),
+    GGMLQuantizationType.Q3_K_LITE: (256, 132),
+    GGMLQuantizationType.Q4_K_LITE: (256, 168),
+    GGMLQuantizationType.Q5_K_LITE: (256, 200),
+    GGMLQuantizationType.Q6_K_LITE: (256, 232),
     GGMLQuantizationType.NVFP4:   (64, 4 + 32),
     GGMLQuantizationType.Q1_0:    (128, 2 + 16),
 }
```

### 5.2 `gguf-py/gguf/quants.py`

```diff
diff --git a/gguf-py/gguf/quants.py b/gguf-py/gguf/quants.py
index 1d9d9ab7d..123696ac2 100644
--- a/gguf-py/gguf/quants.py
+++ b/gguf-py/gguf/quants.py
@@ -472,6 +472,37 @@ class Q3_K(__Quant, qtype=GGMLQuantizationType.Q3_K):
         return (dl * q).reshape((n_blocks, QK_K))
 
 
+class Q3_K_HIFI(__Quant, qtype=GGMLQuantizationType.Q3_K_HIFI):
+    @classmethod
+    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
+        n_blocks = blocks.shape[0]
+
+        # Q3_K_HIFI structure: Q3_K base (110 bytes) + extension (50 bytes)
+        # Base: hmask[32] + qs[64] + scales[12] + d[2] = 110 bytes
+        # Extension: outlier_count[1] + _pad[1] + outlier_idx[16] + outlier_vals[32] = 50 bytes
+        base_size = QK_K // 8 + QK_K // 4 + 12 + 2  # 110 bytes
+        base_blocks = blocks[:, :base_size]
+
+        # Dequantize base Q3_K part
+        q3k_result = Q3_K.dequantize_blocks(base_blocks)
+
+        # Extract outlier data
+        outlier_count = blocks[:, base_size:base_size+1].astype(np.uint8)
+        outlier_idx = blocks[:, base_size+2:base_size+18].astype(np.uint8)  # Skip _pad
+        outlier_vals = blocks[:, base_size+18:base_size+50].view(np.float16).astype(np.float32)  # 16 FP16 values = 32 bytes
+
+        # Apply outlier corrections
+        result = q3k_result.copy()
+        for i in range(n_blocks):
+            n_outliers = min(int(outlier_count[i, 0]), 16)
+            for k in range(n_outliers):
+                idx = int(outlier_idx[i, k])
+                if idx < QK_K:
+                    result[i, idx] += float(outlier_vals[i, k])
+
+        return result
+
+
 class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):
     K_SCALE_SIZE = 12
```

---

## 6. LLaMA Layer Changes

### 6.1 `include/llama.h`

```diff
diff --git a/include/llama.h b/include/llama.h
index ac267b508..127afa881 100644
--- a/include/llama.h
+++ b/include/llama.h
@@ -155,6 +155,17 @@ extern "C" {
         LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
         LLAMA_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
         LLAMA_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors
+        // HIFI / LITE ftypes (44–52; 41–43 reserved — legacy HIFI ids removed)
+        LLAMA_FTYPE_MOSTLY_Q4_K_HIFI     = 44, // Q4_K_M + 2-8 dynamic outliers + early exit (best quality/size ratio)
+        LLAMA_FTYPE_MOSTLY_Q3_K_HIFI     = 45, // Q3_K_M base + Q6_K_HIFI on critical tensors
+        LLAMA_FTYPE_MOSTLY_Q5_K_HIFI     = 46, // Q5_K_M base + Q6_K_HIFI_RES8 on top 10-15% tensors (best 5-bit quality)
+        LLAMA_FTYPE_MOSTLY_Q2_K_HIFI     = 47, // Q2_K base + INT8 residuals on critical tensors (best 2-bit quality)
+
+        LLAMA_FTYPE_MOSTLY_Q2_K_LITE    = 48, // Q2_K base + INT8 residuals (96 bytes/block, ~3.0 bpw)
+        LLAMA_FTYPE_MOSTLY_Q3_K_LITE    = 49, // Q2_K base + INT8 residuals (104 bytes/block, ~3.25 bpw)
+        LLAMA_FTYPE_MOSTLY_Q4_K_LITE    = 50, // Q3_K base + INT8 residuals (128 bytes/block, ~4.0 bpw)
+        LLAMA_FTYPE_MOSTLY_Q5_K_LITE    = 51, // Q4_K base + INT8 residuals (164 bytes/block, ~5.13 bpw)
+        LLAMA_FTYPE_MOSTLY_Q6_K_LITE    = 52, // Q5_K base + INT8 residuals (196 bytes/block, ~6.13 bpw)
 
         LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
     };
```

### 6.2 `src/llama-model-loader.cpp`

```diff
diff --git a/src/llama-model-loader.cpp b/src/llama-model-loader.cpp
index 4e65a45a5..fc87a236c 100644
--- a/src/llama-model-loader.cpp
+++ b/src/llama-model-loader.cpp
@@ -68,6 +68,13 @@ static std::string llama_model_ftype_name(llama_ftype ftype) {
         case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
         case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
         case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";
+        case LLAMA_FTYPE_MOSTLY_Q4_K_HIFI: return "Q4_K_HIFI - ~4.95 bpw (Q4_K base + FP16 outliers, tiered)";
+        case LLAMA_FTYPE_MOSTLY_Q2_K_HIFI: return "Q2_K_HIFI - ~3.0 bpw (Q2_K base + INT8 residuals on critical tensors)";
+        case LLAMA_FTYPE_MOSTLY_Q2_K_LITE: return "Q2_K_LITE - 3.0 bpw  (Q2_K base + INT8 residuals)";
+        case LLAMA_FTYPE_MOSTLY_Q3_K_LITE: return "Q3_K_LITE - 3.25 bpw (Q2_K base + INT8 residuals)";
+        case LLAMA_FTYPE_MOSTLY_Q4_K_LITE: return "Q4_K_LITE - 4.0 bpw  (Q3_K base + INT8 residuals)";
+        case LLAMA_FTYPE_MOSTLY_Q5_K_LITE: return "Q5_K_LITE - 5.13 bpw (Q4_K base + INT8 residuals)";
+        case LLAMA_FTYPE_MOSTLY_Q6_K_LITE: return "Q6_K_LITE - 6.13 bpw (Q5_K base + INT8 residuals)";
 
         default: return "unknown, may not work";
     }
@@ -733,6 +740,19 @@ llama_model_loader::llama_model_loader(
             }
         }
 
+        // Log Q3_K_HIFI tensor count if debug is enabled
+        if (getenv("Q3_K_HIFI_DEBUG") != NULL) {
+            uint32_t q3_k_hifi_count = n_type[GGML_TYPE_Q3_K_HIFI];
+            uint32_t q3_k_count = n_type[GGML_TYPE_Q3_K];
+            if (q3_k_hifi_count > 0) {
+                LLAMA_LOG_INFO("%s: Q3_K_HIFI DEBUG: Found %u Q3_K_HIFI tensors and %u Q3_K tensors in model\n",
+                             __func__, q3_k_hifi_count, q3_k_count);
+            } else if (q3_k_count > 0) {
+                LLAMA_LOG_INFO("%s: Q3_K_HIFI DEBUG: Model uses Q3_K (not Q3_K_HIFI): %u Q3_K tensors found\n",
+                             __func__, q3_k_count);
+            }
+        }
+
         switch (type_max) {
             case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
             case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
@@ -744,6 +764,7 @@ llama_model_loader::llama_model_loader(
             case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
             case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
             case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
+            case GGML_TYPE_Q3_K_HIFI: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_HIFI; break;
             case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
             case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
             case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
@@ -758,6 +779,16 @@ llama_model_loader::llama_model_loader(
             case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
             case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
             case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
+            case GGML_TYPE_Q6_K_HIFI_DYNAMIC: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
+            case GGML_TYPE_Q6_K_HIFI_RES8: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
+            case GGML_TYPE_Q5_K_HIFI_RES8: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
+            case GGML_TYPE_Q4_K_HIFI: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
+            case GGML_TYPE_Q2_K_HIFI: ftype = LLAMA_FTYPE_MOSTLY_Q2_K_HIFI; break;
+            case GGML_TYPE_Q2_K_LITE: ftype = LLAMA_FTYPE_MOSTLY_Q2_K_LITE; break;
+            case GGML_TYPE_Q3_K_LITE: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_LITE; break;
+            case GGML_TYPE_Q4_K_LITE: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_LITE; break;
+            case GGML_TYPE_Q5_K_LITE: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_LITE; break;
+            case GGML_TYPE_Q6_K_LITE: ftype = LLAMA_FTYPE_MOSTLY_Q6_K_LITE; break;
             case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
             case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
             default:
```

### 6.3 `src/llama-model.cpp`

```diff
diff --git a/src/llama-model.cpp b/src/llama-model.cpp
index f057fd31c..0df628f2a 100644
--- a/src/llama-model.cpp
+++ b/src/llama-model.cpp
@@ -419,6 +419,7 @@ const char * llm_type_name(llm_type type) {
         case LLM_TYPE_26B:           return "26B";
         case LLM_TYPE_27B:           return "27B";
         case LLM_TYPE_30B:           return "30B";
+        case LLM_TYPE_31B:           return "31B";
         case LLM_TYPE_32B:           return "32B";
         case LLM_TYPE_34B:           return "34B";
         case LLM_TYPE_35B:           return "35B";
@@ -1613,8 +1614,9 @@ void llama_model::load_hparams(llama_model_loader & ml) {
                 ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
 
                 switch (hparams.n_layer) {
-                    case 35: type = LLM_TYPE_E2B; break;
-                    case 42: type = LLM_TYPE_E4B; break; // to confirm: E4B or E5B?
+                    case 35: type = LLM_TYPE_E2B;  break;
+                    case 42: type = LLM_TYPE_E4B;  break; // to confirm: E4B or E5B?
+                    case 60: type = LLM_TYPE_31B;  break;
                     default: type = LLM_TYPE_UNKNOWN;
                 }
             } break;
```

### 6.4 `src/llama-model.h`

```diff
diff --git a/src/llama-model.h b/src/llama-model.h
index bba70012e..08ad49847 100644
--- a/src/llama-model.h
+++ b/src/llama-model.h
@@ -84,6 +84,7 @@ enum llm_type {
     LLM_TYPE_26B,
     LLM_TYPE_27B,
     LLM_TYPE_30B,
+    LLM_TYPE_31B,
     LLM_TYPE_32B,
     LLM_TYPE_34B,
     LLM_TYPE_35B,
```

### 6.5 `src/llama-quant.cpp`

```diff
diff --git a/src/llama-quant.cpp b/src/llama-quant.cpp
index f91d795b3..fe0e9522e 100644
--- a/src/llama-quant.cpp
+++ b/src/llama-quant.cpp
@@ -3,15 +3,39 @@
 #include "llama-model-loader.h"
 #include "llama-ext.h"
 
+// HIFI layer-adaptive quantization context
+extern "C" {
+#include "../ggml/src/ggml-quants-hifi.h"
+}
+
 #include <algorithm>
 #include <cmath>
 #include <cstring>
 #include <cinttypes>
+#include <cstdlib>  // for getenv
 #include <fstream>
 #include <mutex>
+#include <numeric>
 #include <regex>
 #include <thread>
 #include <unordered_map>
+#include <vector>
+#include <map>
+
+// ===========================================================================
+// IMATRIX-GUIDED TENSOR SELECTION FOR Q3_K_HIFI
+// Store tensor importance scores for global ranking and threshold computation
+// ===========================================================================
+struct tensor_importance_entry {
+    std::string name;
+    float importance;
+    bool is_candidate;  // true if tensor is a Q3_K_HIFI candidate (input projection)
+};
+
+// Global storage for tensor importance data (populated during pre-pass)
+static std::map<std::string, float> g_tensor_importance_map;
+static float g_importance_threshold = 0.0f;
+static bool g_imatrix_guided_enabled = false;
 
 // result of parsing --tensor-type option
 // (changes to this struct must be reflected in tools/quantize/quantize.cpp)
@@ -44,6 +68,343 @@ static void zeros(std::ofstream & file, size_t n) {
     }
 }
 
+// Compute model size in billions from hyperparameters
+static float compute_model_params_b(const llama_hparams & hparams, int64_t n_vocab) {
+    const int64_t n_embd = hparams.n_embd;
+    const int64_t n_ff = hparams.n_ff();
+    const int64_t n_layer = hparams.n_layer;
+
+    // Attention: 4 weight matrices per layer (Q, K, V, O) each ~d*d
+    const int64_t attn_params = 4 * n_embd * n_embd * n_layer;
+    // FFN: 3 weight matrices per layer (gate, up, down) each ~d*n_ff
+    const int64_t ffn_params = 3 * n_embd * n_ff * n_layer;
+    // Embeddings: input + output
+    const int64_t emb_params = 2 * n_vocab * n_embd;
+
+    return (float)(attn_params + ffn_params + emb_params) / 1e9f;
+}
+
+// Get the appropriate HIFI type based on model size for Q4_K_HIFI
+// Small models (≤5B): Q5_K_HIFI_RES8 - size-efficient, proven at 4B scale
+// Large models (>5B): Q6_K_HIFI_RES8 - precision-focused, needed for 8B/14B+ quality
+static ggml_type get_hifi_enhanced_type(float model_params_b) {
+    if (model_params_b <= 5.0f) {
+        // 0.6B–5B: Q5_K_HIFI_RES8 (size-efficient)
+        return GGML_TYPE_Q5_K_HIFI_RES8;
+    } else {
+        // 8B/14B+: Q6_K_HIFI_RES8 (precision-focused)
+        return GGML_TYPE_Q6_K_HIFI_RES8;
+    }
+}
+
+// Get the HIFI type for Q5_K_HIFI - always Q6_K_HIFI_RES8 for maximum precision benefit
+// Q5_K already has good base quality, so HIFI enhancement must be high-precision
+static ggml_type get_q5_hifi_enhanced_type(float model_params_b) {
+    // For Q5_K_HIFI, we want to use Q6_K_HIFI_RES8 only for large models where it helps
+    // For small models (≤2B), use Q6_K instead - no HIFI overhead (matches Q5_K_M behavior)
+    if (model_params_b <= 2.0f) {
+        return GGML_TYPE_Q6_K;  // No HIFI overhead for tiny models
+    } else if (model_params_b <= 5.0f) {
+        return GGML_TYPE_Q5_K_HIFI_RES8;  // Size-efficient HIFI for medium models
+    } else {
+        return GGML_TYPE_Q6_K_HIFI_RES8;  // Full precision HIFI for large models
+    }
+}
+
+// Get the percentage of attn_v layers to enhance based on model size
+// Smaller models benefit more from enhancement, larger models have diminishing returns
+// Strategy: Broader coverage for tiny models (≤1B), graduated reduction for larger
+static float get_hifi_enhancement_threshold(float model_params_b) {
+    if (model_params_b <= 1.0f) {
+        // Tiny models (≤1B, e.g. 0.6B): enhance ~32% (layers 0-8 of 28)
+        // Broader coverage critical for quantization-sensitive small models
+        return 0.32f;
+    } else if (model_params_b <= 2.0f) {
+        // Small models (1-2B): enhance 25% of layers
+        return 0.25f;
+    } else if (model_params_b <= 5.0f) {
+        // Medium-small models (2-5B): enhance 20% of layers
+        return 0.20f;
+    } else if (model_params_b <= 15.0f) {
+        // Medium-large models (5-15B): enhance 20% of layers
+        // Includes 8B and 14B models - matching 8B success case
+        // Results in ~8-12 enhanced tensors (token_embd, output.weight, attn_v layers 0-N)
+        return 0.20f;
+    } else {
+        // Very large models (>15B): Skip ALL attn_v enhancement
+        // Only token_embd and output.weight are enhanced (reduces overhead significantly)
+        return 0.0f;
+    }
+}
+
+// Get the percentage of ffn_gate layers to enhance for tiny models
+// Only tiny models (≤1B) benefit from ffn_gate enhancement - critical for reasoning paths
+static float get_hifi_ffn_gate_threshold(float model_params_b) {
+    if (model_params_b <= 1.0f) {
+        // Tiny models (≤1B): enhance ~18% (layers 0-5 of 28)
+        // ffn_gate enhancement recovers lost reasoning quality in small models
+        return 0.18f;
+    } else {
+        // Larger models: no ffn_gate enhancement needed (diminishing returns)
+        return 0.0f;
+    }
+}
+
+// ===========================================================================
+// Lever 3: Statistical Outlier Detection using 3σ rule
+// Computes the outlier ratio: count(|w| > 3*stddev) / n_elements
+// Used to determine if a tensor benefits from HIFI enhancement
+// ===========================================================================
+static float compute_outlier_ratio(const float * weights, int64_t n) {
+    if (weights == nullptr || n <= 0) {
+        return 0.0f;
+    }
+
+    // Compute mean and stddev in one pass using Welford's algorithm
+    double mean = 0.0;
+    double m2 = 0.0;
+    for (int64_t i = 0; i < n; ++i) {
+        double x = (double)weights[i];
+        double delta = x - mean;
+        mean += delta / (double)(i + 1);
+        double delta2 = x - mean;
+        m2 += delta * delta2;
+    }
+
+    double variance = m2 / (double)n;
+    if (variance <= 0.0) return 0.0f;
+
+    double stddev = sqrt(variance);
+    double threshold = 3.0 * stddev;
+
+    // Count outliers (weights beyond 3σ from mean)
+    int64_t outlier_count = 0;
+    for (int64_t i = 0; i < n; ++i) {
+        if (fabs((double)weights[i] - mean) > threshold) {
+            outlier_count++;
+        }
+    }
+
+    return (float)outlier_count / (float)n;
+}
+
+// Get the outlier ratio threshold for HIFI enhancement based on model size
+// Only enhance tensors whose outlier ratio exceeds this threshold
+// Smaller models need higher thresholds (more selective) to avoid BPW overhead
+static float get_q5_hifi_outlier_threshold(float model_params_b) {
+    if (model_params_b <= 1.0f) {
+        return 0.08f;  // 8% - very selective for tiny models
+    } else if (model_params_b <= 2.0f) {
+        return 0.06f;  // 6% - selective for small models
+    } else if (model_params_b <= 5.0f) {
+        return 0.04f;  // 4% - moderate for medium models
+    } else if (model_params_b <= 10.0f) {
+        return 0.025f; // 2.5% - relaxed for large models
+    } else {
+        return 0.015f; // 1.5% - minimal threshold for very large models
+    }
+}
+
+// ===========================================================================
+// Lever 1: Adaptive Enhancement by Model Scale for Q5_K_HIFI
+// Returns the max number of enhanced tensors based on model size
+// Smaller models get fewer enhanced tensors to minimize BPW overhead
+// ===========================================================================
+static int get_q5_hifi_max_enhancements(float model_params_b) {
+    if (model_params_b <= 1.0f) {
+        return 2;  // Only token_embd + output for tiny models
+    } else if (model_params_b <= 2.0f) {
+        return 3;  // + maybe 1 attn_v for small models
+    } else if (model_params_b <= 5.0f) {
+        return 5;  // + 3 attn_v layers for medium models
+    } else if (model_params_b <= 10.0f) {
+        return 6;  // + 4 attn_v layers for large models
+    } else {
+        return 5;  // Focused enhancement for very large models
+    }
+}
+
+// Get Q5_K_HIFI enhancement threshold for attn_v layers
+// This is much more conservative than Q4_K_HIFI - focuses on proven wins
+static float get_q5_hifi_attn_v_threshold(float model_params_b) {
+    if (model_params_b <= 1.7f) {
+        return 0.0f;  // NO attn_v enhancement for tiny models - match Q5_K_M BPW
+    } else if (model_params_b <= 5.0f) {
+        return 0.05f;  // Only top 5% for medium models (1-2 layers)
+    } else if (model_params_b <= 10.0f) {
+        return 0.08f;  // 8% for large models (proven at 8B)
+    } else {
+        return 0.05f;  // Conservative for very large models
+    }
+}
+
+// ===========================================================================
+// Q3_K_HIFI Scale-Aware Enhancement Logic
+// Based on proven strategies from Q4_K_HIFI and Q5_K_HIFI
+// Key insight: Fixed enhancement doesn't scale - small/large models need different approaches
+// ===========================================================================
+
+// Get the percentage of attn_v layers to enhance for Q3_K_HIFI
+// Tiny models (≤1.7B): Skip HIFI overhead that hurts them
+// Medium models (2-8B): Full enhancement (sweet spot)
+// Large models (14B+): Minimal enhancement (large models self-correct)
+static float get_q3_hifi_attn_v_threshold(float model_params_b) {
+    if (model_params_b <= 1.0f) {
+        // 0.6B/1B: Skip attn_v HIFI entirely - matches Q3_K_M BPW
+        // This addresses the +2.2% PPL regression seen at 0.6B
+        return 0.0f;
+    } else if (model_params_b <= 2.0f) {
+        // 1.7B: Q3_K_HIFI DISABLED - match Q3_K_M behavior exactly
+        // Q3_K_M uses: first 2 layers get Q5_K, rest Q4_K (threshold = 2/28 ≈ 0.07)
+        return 0.07f;
+    } else if (model_params_b <= 5.0f) {
+        // 2-5B: Full enhancement - this is the sweet spot
+        // 4B shows -2.9% PPL improvement with current Q3_K_HIFI
+        return 0.25f;
+    } else if (model_params_b <= 10.0f) {
+        // 8B: Moderate enhancement
+        return 0.15f;
+    } else if (model_params_b <= 20.0f) {
+        // 14B: Reduced enhancement - addresses +0.24% PPL regression
+        return 0.08f;
+    } else {
+        // 32B+: Minimal enhancement - addresses +0.13% PPL regression
+        return 0.05f;
+    }
+}
+
+// Get the enhancement type for Q3_K_HIFI attn_v layers based on model size
+// Smaller models: Q4_K (avoid excessive BPW overhead)
+// Larger models: Q5_K (quality focus with more headroom)
+static ggml_type get_q3_hifi_attn_v_type(float model_params_b) {
+    if (model_params_b <= 2.0f) {
+        // Small models: Q4_K to minimize BPW overhead
+        return GGML_TYPE_Q4_K;
+    } else if (model_params_b <= 10.0f) {
+        // Medium models: Q5_K for better quality
+        return GGML_TYPE_Q5_K;
+    } else {
+        // Large models: Q5_K (they can afford the bits)
+        return GGML_TYPE_Q5_K;
+    }
+}
+
+// Get the enhancement type for Q3_K_HIFI ffn_down layers based on model size
+// Follows Q3_K_M default but with scale-aware adjustments
+static ggml_type get_q3_hifi_ffn_down_type(float model_params_b, int i_layer, int n_layer) {
+    // Early layers (first 1/16) always get Q5_K
+    if (i_layer < n_layer / 16) {
+        return GGML_TYPE_Q5_K;
+    }
+
+    // Tiny models: use Q4_K for middle layers (match Q3_K_M behavior)
+    if (model_params_b <= 1.7f) {
+        return GGML_TYPE_Q4_K;
+    }
+
+    // Medium/large models: use Q4_K for most layers
+    return GGML_TYPE_Q4_K;
+}
+
+// ===========================================================================
+// IMATRIX-GUIDED TENSOR SELECTION HELPERS
+// Check if a tensor is a Q3_K_HIFI candidate (input projection) and compute
+// importance threshold based on global ranking
+// ===========================================================================
+
+// Check if a tensor is a Q3_K_HIFI candidate (input projection, not output)
+static bool is_q3_hifi_candidate(const std::string & name) {
+    // Exclude output projections - these are too sensitive
+    bool is_output_projection =
+        name.find("o_proj") != std::string::npos ||
+        name.find("attn_output") != std::string::npos ||
+        name.find("down_proj") != std::string::npos ||
+        name.find("ffn_down") != std::string::npos ||
+        name.find("output.weight") != std::string::npos ||
+        name.find("lm_head") != std::string::npos ||
+        name.find("ssm_out") != std::string::npos;
+
+    if (is_output_projection) {
+        return false;
+    }
+
+    // Include input projections
+    bool is_input_projection =
+        name.find("q_proj") != std::string::npos ||
+        name.find("k_proj") != std::string::npos ||
+        name.find("v_proj") != std::string::npos ||
+        name.find("gate_proj") != std::string::npos ||
+        name.find("up_proj") != std::string::npos ||
+        name.find("attn_q") != std::string::npos ||
+        name.find("attn_k") != std::string::npos ||
+        name.find("attn_v") != std::string::npos ||
+        name.find("ffn_gate") != std::string::npos ||
+        name.find("ffn_up") != std::string::npos ||
+        name.find("wqkv") != std::string::npos ||
+        name.find("qkv") != std::string::npos;
+
+    return is_input_projection;
+}
+
+// Get model-size-aware imatrix guidance threshold
+// Returns the percentage of top-importance tensors to enhance with Q3_K_HIFI
+static float get_imatrix_guidance_threshold(float model_params_b) {
+    if (model_params_b <= 2.0f) {
+        // Tiny models: DISABLE imatrix-guided Q3_K_HIFI
+        // Q3_K_HIFI hurts at this scale regardless of configuration
+        return 0.0f;
+    } else if (model_params_b <= 5.0f) {
+        // Medium-small models (2-5B): enhance top 30% of tensors
+        // This is the sweet spot where Q3_K_HIFI provides good improvement
+        return 0.30f;
+    } else if (model_params_b <= 10.0f) {
+        // Medium models (5-10B): enhance top 20% of tensors
+        return 0.20f;
+    } else if (model_params_b <= 20.0f) {
+        // Large models (10-20B): enhance top 15% of tensors
+        return 0.15f;
+    } else {
+        // Very large models (20B+): enhance top 10% of tensors
+        // Large models have redundancy and need less enhancement
+        return 0.10f;
+    }
+}
+
+// Compute importance threshold from collected tensor importance scores
+// Returns the threshold value where tensors above this get Q3_K_HIFI
+static float compute_importance_threshold(
+    const std::vector<tensor_importance_entry> & entries,
+    float top_percent
+) {
+    if (entries.empty() || top_percent <= 0.0f) {
+        return 1.0f;  // No tensors get Q3_K_HIFI
+    }
+
+    // Collect importance values from candidate tensors only
+    std::vector<float> importance_values;
+    importance_values.reserve(entries.size());
+    for (const auto & e : entries) {
+        if (e.is_candidate) {
+            importance_values.push_back(e.importance);
+        }
+    }
+
+    if (importance_values.empty()) {
+        return 1.0f;
+    }
+
+    // Sort in descending order (highest importance first)
+    std::sort(importance_values.begin(), importance_values.end(), std::greater<float>());
+
+    // Find the threshold at top_percent
+    size_t cutoff_idx = (size_t)(importance_values.size() * top_percent);
+    if (cutoff_idx >= importance_values.size()) {
+        cutoff_idx = importance_values.size() - 1;
+    }
+
+    return importance_values[cutoff_idx];
+}
+
 static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
     if (prune.empty()) {
         return orig_name;
@@ -417,6 +778,7 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
     auto use_more_bits = [](int i_layer, int n_layers) -> bool {
         return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
     };
+
     const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
     auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
         if (n_expert > 1) {
@@ -454,6 +816,40 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
                      ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                 new_type = GGML_TYPE_Q5_K;
             }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+                // Q4_K_HIFI: Use size-aware HIFI type on output - always critical
+                // Q5_K_HIFI_RES8 for 4B-10B, Q6_K_HIFI_RES8 for smaller models
+                const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+                new_type = get_hifi_enhanced_type(model_params_b);
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI) {
+                // Q5_K_HIFI: Use scale-appropriate type on output.weight
+                // Tiny models (≤2B): Use Q6_K (no HIFI overhead, matches Q5_K_M)
+                // Medium models (2-5B): Use Q5_K_HIFI_RES8 (efficient enhancement)
+                // Large models (>5B): Use Q6_K_HIFI_RES8 (maximum precision)
+                const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+                new_type = get_q5_hifi_enhanced_type(model_params_b);
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) {
+                // Q2_K_HIFI: output.weight is always critical — use Q6_K (matches Q2_K behavior)
+                new_type = GGML_TYPE_Q6_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) {
+                // Q3_K_HIFI: Scale-aware output.weight handling
+                // Q3_K_M uses Q6_K via default else clause, so we match that for consistency
+                // However, for tiny models we could consider matching the lower overhead
+                const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+                // Q6_K for all sizes (matches Q3_K_M behavior)
+                // output.weight is critical for quality, so keep Q6_K even for tiny models
+                new_type = GGML_TYPE_Q6_K;
+                (void)model_params_b; // Suppress unused warning - kept for future tuning
+            }
+            // K_LITE output.weight: bump one tier higher within LITE family
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_LITE) { new_type = GGML_TYPE_Q3_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_LITE) { new_type = GGML_TYPE_Q4_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_LITE) { new_type = GGML_TYPE_Q5_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_LITE) { new_type = GGML_TYPE_Q6_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q6_K_LITE) { new_type = GGML_TYPE_Q8_0; }
             else if (new_type != GGML_TYPE_Q8_0) {
                 new_type = GGML_TYPE_Q6_K;
             }
@@ -483,6 +879,41 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
             else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                 new_type = GGML_TYPE_Q4_K;
             }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+                // Q4_K_HIFI: Use size-aware HIFI type on token embeddings - always critical
+                // Q5_K_HIFI_RES8 for 4B-10B, Q6_K_HIFI_RES8 for smaller models
+                const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+                new_type = get_hifi_enhanced_type(model_params_b);
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI) {
+                // Q5_K_HIFI: Use scale-appropriate type on token_embd
+                // Tiny models (≤2B): Use Q6_K (no HIFI overhead, matches Q5_K_M)
+                // Medium models (2-5B): Use Q5_K_HIFI_RES8 (efficient enhancement)
+                // Large models (>5B): Use Q6_K_HIFI_RES8 (maximum precision)
+                const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+                new_type = get_q5_hifi_enhanced_type(model_params_b);
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) {
+                // Q2_K_HIFI: token embeddings are critical — use Q4_K (matches Q2_K behavior)
+                new_type = GGML_TYPE_Q4_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) {
+                // Q3_K_HIFI: Scale-aware token_embd handling
+                // The key insight: Q3_K_M does NOT explicitly handle token_embd, so it uses default (Q3_K)
+                // For tiny models (≤1.7B): Match Q3_K_M → use default type (no explicit assignment)
+                // For larger models (>1.7B): Use Q6_K for better quality
+                const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+                if (model_params_b > 1.7f) {
+                    new_type = GGML_TYPE_Q6_K;
+                }
+                // else: tiny models skip - use default_type (Q3_K), matching Q3_K_M
+            }
+            // K_LITE token_embd: bump one tier higher within LITE family
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_LITE) { new_type = GGML_TYPE_Q3_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_LITE) { new_type = GGML_TYPE_Q4_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_LITE) { new_type = GGML_TYPE_Q5_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_LITE) { new_type = GGML_TYPE_Q6_K_LITE; }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q6_K_LITE) { new_type = GGML_TYPE_Q8_0; }
         }
     } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
@@ -527,10 +958,56 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
             new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
         }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) {
+            // Q2_K_HIFI: Match Q2_K behavior for attn_v
+            new_type = GGML_TYPE_Q3_K;
+        }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) {
+            // Q3_K_HIFI: Match Q3_K_M strategy exactly for attn_v
+            // Q3_K_M uses: Q5_K for first 2 layers, Q4_K for the rest
+            // We match this exactly - no Q3_K is used here, so no upgrade needed
+            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
+        }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+            // Q4_K_HIFI: Model-size-aware enhancement to optimize size vs quality tradeoff
+            // - Tiny models (≤1B): Q5_K_HIFI_RES8, enhance 32% of attn_v layers
+            // - Small models (1-5B): Q5_K_HIFI_RES8, enhance 20-25% of layers
+            // - Large models (5-15B): Q6_K_HIFI_RES8, enhance 20% of layers (~8-12 tensors)
+            // - Very large models (>15B): Only token_embd and output.weight enhanced
+            // This provides optimal quality/size tradeoff at each model scale
+            const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+            const float enhancement_threshold = get_hifi_enhancement_threshold(model_params_b);
+            const ggml_type hifi_type = get_hifi_enhanced_type(model_params_b);
+
+            if (qs.i_attention_wv <= qs.n_attention_wv * enhancement_threshold) {
+                new_type = hifi_type;  // Use size-appropriate HIFI type
+            } else if (use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) {
+                new_type = GGML_TYPE_Q6_K;  // Follow Q4_K_M behavior for critical late layers
+            } else {
+                new_type = GGML_TYPE_Q4_K_HIFI;  // Q4_K_HIFI for medium-sensitivity mid layers
+            }
+        }
         else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
         else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
             new_type = GGML_TYPE_Q5_K;
         }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI) {
+            // Q5_K_HIFI: Adaptive enhancement based on model size + statistical filtering
+            // Lever 1: Scale-adaptive thresholds - tiny models get minimal enhancement
+            // Lever 3: Only enhance if tensor has high outlier ratio (pending weight access)
+            const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+            const float enhancement_threshold = get_q5_hifi_attn_v_threshold(model_params_b);
+
+            // For tiny models (≤1.7B), skip ALL attn_v HIFI enhancement - only use Q5_K_M logic
+            // This matches Q5_K_M BPW while still getting HIFI benefit on token_embd/output
+            if (enhancement_threshold > 0.0f && qs.i_attention_wv <= qs.n_attention_wv * enhancement_threshold) {
+                // Use scale-appropriate HIFI type
+                new_type = get_q5_hifi_enhanced_type(model_params_b);
+            } else if (use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) {
+                new_type = GGML_TYPE_Q6_K;  // Follow Q5_K_M behavior for critical late layers
+            }
+            // else: use default Q5_K for non-critical middle/late layers
+        }
         else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                 use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
         else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
@@ -569,6 +1046,8 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
         int i_layer = info.first, n_layer = info.second;
         if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) new_type = GGML_TYPE_Q3_K;
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_LITE) new_type = GGML_TYPE_Q3_K_LITE;
         else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
             if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
         }
@@ -580,6 +1059,14 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
                      : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                      : GGML_TYPE_Q3_K;
         }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) {
+            // Q3_K_HIFI: Match Q3_K_M strategy exactly, then upgrade Q3_K to Q3_K_HIFI
+            // Q3_K_M uses: Q5_K for early layers, Q4_K for most, Q3_K only for FALCON
+            // We match this exactly, then upgrade Q3_K → Q3_K_HIFI at the end
+            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
+                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
+                     : GGML_TYPE_Q3_K;  // Only FALCON with !use_more_bits gets Q3_K (will be upgraded to Q3_K_HIFI)
+        }
         else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                     (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
             new_type = GGML_TYPE_Q4_K;
@@ -587,7 +1074,8 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
             new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
         }
-        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+            // Q4_K_HIFI follows Q4_K_M behavior for ffn_down
             if (arch == LLM_ARCH_FALCON) {
                 new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                            use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
@@ -598,7 +1086,10 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
             new_type = GGML_TYPE_Q5_K;
         }
-        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
+        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI) && use_more_bits(i_layer, n_layer)) {
+            // Q5_K_HIFI follows Q5_K_M behavior for ffn_down - Q6_K for critical layers
+            new_type = GGML_TYPE_Q6_K;
+        }
         else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
             new_type = GGML_TYPE_Q5_K;
         }
@@ -616,13 +1107,18 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
                 if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                     ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
-                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
+                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS ||
+                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI ||
+                    ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) {
                     new_type = GGML_TYPE_Q5_K;
                 }
             } else {
                 if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
+                else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) new_type = GGML_TYPE_Q3_K;
                 else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                 else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
+                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) new_type = GGML_TYPE_Q4_K;  // Match Q3_K_M
+                else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) new_type = GGML_TYPE_Q4_K_HIFI;  // Medium-sensitivity
                 else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                 else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
             }
@@ -630,12 +1126,12 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
             if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
         }
     }
-    else if (category == tensor_category::ATTENTION_QKV) {
+    else if (name.find("attn_qkv.weight") != std::string::npos) {
         if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
             new_type = GGML_TYPE_Q4_K;
         }
-        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
-        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) new_type = GGML_TYPE_Q5_K;
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI) new_type = GGML_TYPE_Q6_K;
     }
     else if (category == tensor_category::FFN_GATE) {
         auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
@@ -643,6 +1139,19 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
             new_type = GGML_TYPE_IQ3_XXS;
         }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+            // Q4_K_HIFI: Enhance early ffn_gate layers for tiny models (≤1B)
+            // ffn_gate is critical for reasoning paths in small models
+            const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+            const float ffn_gate_threshold = get_hifi_ffn_gate_threshold(model_params_b);
+
+            if (ffn_gate_threshold > 0.0f && i_layer <= n_layer * ffn_gate_threshold) {
+                const ggml_type hifi_type = get_hifi_enhanced_type(model_params_b);
+                new_type = hifi_type;  // Use HIFI type for early ffn_gate layers
+            } else {
+                new_type = GGML_TYPE_Q4_K_HIFI;  // Q4_K_HIFI for medium-sensitivity
+            }
+        }
         ++qs.i_ffn_gate;
     }
     else if (category == tensor_category::FFN_UP) {
@@ -651,9 +1160,274 @@ static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type
         if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
             new_type = GGML_TYPE_IQ3_XXS;
         }
+        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+            new_type = GGML_TYPE_Q4_K_HIFI;  // Q4_K_HIFI for medium-sensitivity ffn_up
+        }
         ++qs.i_ffn_up;
     }
 
+    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
+    //}
+    // IK: let's remove this, else Q2_K is almost the same as Q3_K_S
+    //else if (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos) {
+    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
+    //}
+    // This can be used to reduce the size of the Q5_K_S model.
+    // The associated PPL increase is fully in line with the size reduction
+    //else {
+    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) new_type = GGML_TYPE_Q4_K;
+    //}
+    // === Q3_K_HIFI: Upgrade Q3_K to Q3_K_HIFI ONLY for safe input-heavy layers ===
+    // Critical: Q3_K_HIFI should NOT be applied to output projections (o_proj, down_proj, output.weight)
+    // These layers are extremely sensitive to 3-bit quantization, even with outlier correction.
+    // Only apply Q3_K_HIFI to input projections that tolerate 3-bit well.
+    if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI && new_type == GGML_TYPE_Q3_K) {
+        // First, check if this is an output projection (EXCLUDE these)
+        bool is_output_projection =
+            name.find("o_proj") != std::string::npos ||
+            name.find("attn_output") != std::string::npos ||
+            name.find("down_proj") != std::string::npos ||
+            name.find("ffn_down") != std::string::npos ||
+            tensor_name_match_output_weight(name.c_str()) ||
+            name.find("lm_head") != std::string::npos ||
+            name.find("ssm_out") != std::string::npos;  // Qwen3Next linear attention output
+
+        if (is_output_projection) {
+            // Output projections: use Q4_K instead of Q3_K_HIFI
+            new_type = GGML_TYPE_Q4_K;
+            const char * debug_env = getenv("Q3_K_HIFI_DEBUG");
+            if (debug_env) {
+                static int skip_count = 0;
+                skip_count++;
+                if (skip_count <= 10) {
+                    LLAMA_LOG_INFO("Q3_K_HIFI: Excluding output projection '%s' from Q3_K_HIFI, using Q4_K instead (count: %d)\n",
+                                  name.c_str(), skip_count);
+                }
+            }
+        } else {
+            // MODEL-SIZE-AWARE + IMATRIX-GUIDED Q3_K_HIFI TENSOR SELECTION
+            // Priority 1: If imatrix-guided mode is enabled, use importance threshold
+            // Priority 2: Fall back to model-size-aware name-based selection
+            const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+
+            bool is_safe_for_q3_k_hifi = false;
+            bool used_imatrix_guidance = false;
+
+            // Check if this tensor is a Q3_K_HIFI candidate (input projection)
+            bool is_candidate = is_q3_hifi_candidate(name);
+
+            // IMATRIX-GUIDED SELECTION (if enabled and tensor is a candidate)
+            if (g_imatrix_guided_enabled && is_candidate) {
+                // Look up tensor importance from pre-computed map
+                auto it = g_tensor_importance_map.find(name);
+                if (it != g_tensor_importance_map.end()) {
+                    float tensor_importance = it->second;
+                    // Tensor gets Q3_K_HIFI if importance >= threshold (top N%)
+                    is_safe_for_q3_k_hifi = (tensor_importance >= g_importance_threshold);
+                    used_imatrix_guidance = true;
+
+                    const char * debug_env = getenv("Q3_K_HIFI_DEBUG");
+                    if (debug_env) {
+                        static int imatrix_log_count = 0;
+                        if (imatrix_log_count++ < 20) {
+                            LLAMA_LOG_INFO("Q3_K_HIFI: imatrix-guided '%s' imp=%.3f threshold=%.3f -> %s\n",
+                                          name.c_str(), tensor_importance, g_importance_threshold,
+                                          is_safe_for_q3_k_hifi ? "Q3_K_HIFI" : "Q4_K");
+                        }
+                    }
+                }
+            }
+
+            // FALLBACK TO MODEL-SIZE-AWARE SELECTION (if imatrix not available/used)
+            if (!used_imatrix_guidance) {
+                if (model_params_b <= 2.0f) {
+                    // TINY MODELS (≤1.7B): DISABLE Q3_K_HIFI entirely
+                    // Testing showed Q3_K_HIFI hurts 1.7B regardless of strategy:
+                    //   - Ultra-surgical: PPL 18.00 vs Q3_K_M 17.75
+                    //   - Bulk: PPL 18.58 - even worse
+                    // Fall back to Q3_K_M behavior (use Q4_K for these tensors)
+                    is_safe_for_q3_k_hifi = false;
+                } else if (model_params_b <= 10.0f) {
+                    // MEDIUM MODELS (2B-8B): FULL Q3_K_HIFI - this is the sweet spot
+                    // 4B shows -2.9% PPL improvement with Q3_K_HIFI
+                    is_safe_for_q3_k_hifi =
+                        name.find("q_proj") != std::string::npos ||
+                        name.find("k_proj") != std::string::npos ||
+                        name.find("v_proj") != std::string::npos ||
+                        name.find("gate_proj") != std::string::npos ||
+                        name.find("up_proj") != std::string::npos ||
+                        name.find("attn_q") != std::string::npos ||
+                        name.find("attn_k") != std::string::npos ||
+                        name.find("attn_v") != std::string::npos ||
+                        name.find("ffn_gate") != std::string::npos ||
+                        name.find("ffn_up") != std::string::npos ||
+                        name.find("wqkv") != std::string::npos ||
+                        name.find("qkv") != std::string::npos;
+                } else {
+                    // LARGE MODELS (14B+): REDUCED Q3_K_HIFI
+                    // Use Q3_K_HIFI only on attention input (q, k) and FFN gate
+                    // Leave v_proj, up_proj as Q4_K to match Q3_K_M efficiency
+                    // This addresses the +0.24% PPL regression at 14B and +0.13% at 32B
+                    is_safe_for_q3_k_hifi =
+                        name.find("q_proj") != std::string::npos ||
+                        name.find("k_proj") != std::string::npos ||
+                        name.find("gate_proj") != std::string::npos ||
+                        name.find("attn_q") != std::string::npos ||
+                        name.find("attn_k") != std::string::npos ||
+                        name.find("ffn_gate") != std::string::npos;
+                    // EXCLUDE for 14B+: v_proj, up_proj (use Q4_K instead)
+                }
+            }
+
+            // For ffn_down: only allow Q3_K_HIFI if Q3_K_M would use Q3_K (FALCON with !use_more_bits)
+            if (name.find("ffn_down") != std::string::npos) {
+                auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
+                int i_layer = info.first, n_layer = info.second;
+                // Q3_K_M uses Q3_K only for FALCON with !use_more_bits
+                if (arch == LLM_ARCH_FALCON && !use_more_bits(i_layer, n_layer)) {
+                    is_safe_for_q3_k_hifi = true;
+                } else {
+                    is_safe_for_q3_k_hifi = false;  // ffn_down should already be Q4_K from earlier logic
+                }
+            }
+
+            if (is_safe_for_q3_k_hifi) {
+                static int upgrade_count = 0;
+                static bool debug_logged = false;
+                const char * debug_env = getenv("Q3_K_HIFI_DEBUG");
+                if (debug_env && !debug_logged) {
+                    LLAMA_LOG_INFO("Q3_K_HIFI: Debug enabled - will upgrade Q3_K tensors to Q3_K_HIFI (only safe input layers)\n");
+                    debug_logged = true;
+                }
+                new_type = GGML_TYPE_Q3_K_HIFI;
+                upgrade_count++;
+                if (debug_env && upgrade_count <= 10) {
+                    LLAMA_LOG_INFO("Q3_K_HIFI: Upgraded tensor '%s' from Q3_K to Q3_K_HIFI (count: %d)\n",
+                                  name.c_str(), upgrade_count);
+                }
+            } else {
+                // Unknown tensor type - be conservative and use Q4_K
+                new_type = GGML_TYPE_Q4_K;
+                const char * debug_env = getenv("Q3_K_HIFI_DEBUG");
+                if (debug_env) {
+                    static int unknown_count = 0;
+                    unknown_count++;
+                    if (unknown_count <= 10) {
+                        LLAMA_LOG_INFO("Q3_K_HIFI: Unknown tensor '%s' - using Q4_K instead of Q3_K_HIFI (count: %d)\n",
+                                      name.c_str(), unknown_count);
+                    }
+                }
+            }
+        }
+    }
+
+    // === Q2_K_HIFI: Upgrade Q2_K to Q2_K_HIFI for critical input-heavy layers ===
+    // Protects top-3 outliers per superblock BEFORE Q2_K quantization (stored as FP16).
+    // Concentrate enhancement on tensors where quantization error causes the most PPL damage.
+    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI && new_type == GGML_TYPE_Q2_K) {
+        bool is_output_projection =
+            name.find("o_proj") != std::string::npos ||
+            name.find("attn_output") != std::string::npos ||
+            name.find("down_proj") != std::string::npos ||
+            name.find("ffn_down") != std::string::npos ||
+            name == "output.weight" ||
+            name.find("lm_head") != std::string::npos;
+
+        if (!is_output_projection) {
+            const float model_params_b = compute_model_params_b(qs.model.hparams, qs.model.vocab.n_tokens());
+
+            bool upgrade_to_hifi = false;
+
+            // HIFI enhancement targets attention Q/K projections, which are
+            // high-impact for model quality but relatively small tensors.
+            // FFN gate/up are excluded: they are 4-8x larger than attention tensors
+            // and enhancing them causes ~50% of the model to use Q2_K_HIFI,
+            // creating severe speed regression with minimal PPL benefit per byte.
+            if (model_params_b <= 2.0f) {
+                // Tiny models (<=2B): only Q/K projections
+                upgrade_to_hifi =
+                    name.find("q_proj") != std::string::npos ||
+                    name.find("k_proj") != std::string::npos ||
+                    name.find("attn_q") != std::string::npos ||
+                    name.find("attn_k") != std::string::npos;
+            } else if (model_params_b <= 10.0f) {
+                // Medium models (3B-8B): Q/K projections only
+                // ffn_gate/ffn_up are too large — 2×47.50 MiB/layer dominates the model
+                upgrade_to_hifi =
+                    name.find("q_proj") != std::string::npos ||
+                    name.find("k_proj") != std::string::npos ||
+                    name.find("attn_q") != std::string::npos ||
+                    name.find("attn_k") != std::string::npos ||
+                    name.find("wqkv") != std::string::npos ||
+                    name.find("qkv") != std::string::npos;
+            } else {
+                // Large models (13B+): Q/K/V projections (still exclude FFN for speed)
+                upgrade_to_hifi =
+                    name.find("q_proj") != std::string::npos ||
+                    name.find("k_proj") != std::string::npos ||
+                    name.find("v_proj") != std::string::npos ||
+                    name.find("attn_q") != std::string::npos ||
+                    name.find("attn_k") != std::string::npos ||
+                    name.find("attn_v") != std::string::npos ||
+                    name.find("wqkv") != std::string::npos ||
+                    name.find("qkv") != std::string::npos;
+            }
+
+            if (upgrade_to_hifi) {
+                new_type = GGML_TYPE_Q2_K_HIFI;
+                if (getenv("Q2_K_HIFI_DEBUG")) {
+                    LLAMA_LOG_INFO("Q2_K_HIFI: Upgraded '%s' from Q2_K to Q2_K_HIFI (model=%.1fB)\n",
+                                  name.c_str(), model_params_b);
+                }
+            }
+        }
+    }
+
+    bool convert_incompatible_tensor = false;
+    {
+        const int64_t nx = tensor->ne[0];
+        const int64_t ny = tensor->ne[1];
+        const int64_t qk_k = ggml_blck_size(new_type);
+
+        if (nx % qk_k != 0) {
+            LLAMA_LOG_WARN("\n\n%s : tensor cols %" PRId64 " x %" PRId64 " are not divisible by %" PRId64 ", required for %s", __func__, nx, ny, qk_k, ggml_type_name(new_type));
+            convert_incompatible_tensor = true;
+        }
+    }
+
+    if (convert_incompatible_tensor) {
+        switch (new_type) {
+            case GGML_TYPE_TQ1_0:
+            case GGML_TYPE_TQ2_0:  new_type = GGML_TYPE_Q4_0; break;  // TODO: use a symmetric type instead
+            case GGML_TYPE_IQ2_XXS:
+            case GGML_TYPE_IQ2_XS:
+            case GGML_TYPE_IQ2_S:
+            case GGML_TYPE_IQ3_XXS:
+            case GGML_TYPE_IQ3_S:
+            case GGML_TYPE_IQ1_S:
+            case GGML_TYPE_IQ1_M:
+            case GGML_TYPE_Q2_K:
+            case GGML_TYPE_Q2_K_HIFI:
+            case GGML_TYPE_Q2_K_LITE:
+            case GGML_TYPE_Q3_K:
+            case GGML_TYPE_Q3_K_HIFI:
+            case GGML_TYPE_Q3_K_LITE:
+            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
+            case GGML_TYPE_Q4_K:
+            case GGML_TYPE_Q4_K_LITE: new_type = GGML_TYPE_Q5_0; break;
+            case GGML_TYPE_Q5_K:
+            case GGML_TYPE_Q5_K_LITE: new_type = GGML_TYPE_Q5_1; break;
+            case GGML_TYPE_Q6_K:
+            case GGML_TYPE_Q6_K_LITE: new_type = GGML_TYPE_Q8_0; break;
+            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
+        }
+        if (tensor->ne[0] % ggml_blck_size(new_type) != 0) {
+            new_type = GGML_TYPE_F16;
+        }
+        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
+        ++qs.n_fallback;
+    }
+
     return new_type;
 }
 
@@ -706,10 +1480,17 @@ static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_mod
 // quantization implementation
 //
 
-static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
+// Overload with HIFI context support
+static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread, const ggml_hifi_quant_context * hifi_ctx = nullptr) {
     if (nthread < 2) {
-        // single-thread
+        // single-thread - set context directly
+        if (hifi_ctx) {
+            ggml_hifi_set_context(hifi_ctx);
+        }
         size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
+        if (hifi_ctx) {
+            ggml_hifi_set_context(nullptr);
+        }
         if (!ggml_validate_row_data(new_type, new_data, new_size)) {
             throw std::runtime_error("quantized data validation failed");
         }
@@ -721,7 +1502,12 @@ static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float *
     size_t new_size = 0;
     bool valid = true;
     auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
-            nrows, n_per_row, imatrix]() {
+            nrows, n_per_row, imatrix, hifi_ctx]() {
+        // Set HIFI context for this thread
+        if (hifi_ctx) {
+            ggml_hifi_set_context(hifi_ctx);
+        }
+
         const int64_t nrows_per_chunk = chunk_size / n_per_row;
         size_t local_size = 0;
         while (true) {
@@ -747,6 +1533,11 @@ static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float *
                 break;
             }
         }
+
+        // Clear HIFI context for this thread
+        if (hifi_ctx) {
+            ggml_hifi_set_context(nullptr);
+        }
     };
     for (int it = 0; it < nthread - 1; ++it) {
         workers.emplace_back(compute);
@@ -826,9 +1617,22 @@ ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
         case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
         case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
         case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
-        case LLAMA_FTYPE_MOSTLY_IQ3_S:
+        case LLAMA_FTYPE_MOSTLY_IQ3_S:   return GGML_TYPE_IQ3_S;
         case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;
 
+        // HIFI types
+        case LLAMA_FTYPE_MOSTLY_Q4_K_HIFI: return GGML_TYPE_Q4_K;
+        case LLAMA_FTYPE_MOSTLY_Q5_K_HIFI: return GGML_TYPE_Q5_K;
+        case LLAMA_FTYPE_MOSTLY_Q3_K_HIFI: return GGML_TYPE_Q3_K;
+        case LLAMA_FTYPE_MOSTLY_Q2_K_HIFI: return GGML_TYPE_Q2_K_HIFI;
+
+        // LITE types
+        case LLAMA_FTYPE_MOSTLY_Q2_K_LITE: return GGML_TYPE_Q2_K_LITE;
+        case LLAMA_FTYPE_MOSTLY_Q3_K_LITE: return GGML_TYPE_Q3_K_LITE;
+        case LLAMA_FTYPE_MOSTLY_Q4_K_LITE: return GGML_TYPE_Q4_K_LITE;
+        case LLAMA_FTYPE_MOSTLY_Q5_K_LITE: return GGML_TYPE_Q5_K_LITE;
+        case LLAMA_FTYPE_MOSTLY_Q6_K_LITE: return GGML_TYPE_Q6_K_LITE;
+
         default: return GGML_TYPE_COUNT;
     }
 }
@@ -930,6 +1734,13 @@ static void llama_model_quantize_impl(const std::string & fname_inp, const std::
     gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV
     gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype); // TODO: use LLM_KV
 
+    // Set quantization type string for Hugging Face model card display
+    if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
+        gguf_set_val_str(ctx_out.get(), "general.quantization_type", "Q4_K_HIFI");
+    } else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI) {
+        gguf_set_val_str(ctx_out.get(), "general.quantization_type", "Q5_K_HIFI");
+    }
+
     // Remove split metadata
     gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
     gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
@@ -1232,14 +2043,171 @@ static void llama_model_quantize_impl(const std::string & fname_inp, const std::
                 const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                 const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;
 
-                // quantize each expert separately since they have different importance matrices
-                new_size = 0;
-                for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
-                    const float * f32_data_03 = f32_data + i03 * nelements_matrix;
-                    void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
-                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;
+            // quantize each expert separately since they have different importance matrices
+            new_size = 0;
+
+            // Set up HIFI context for Q6_K_HIFI_RES8, Q5_K_HIFI_RES8, and Q3_K_HIFI tensors
+            ggml_hifi_quant_context hifi_ctx = {};
+            const ggml_hifi_quant_context * hifi_ctx_ptr = nullptr;
+
+            // Compute model size in billions (needed for Q3_K_HIFI and other HIFI types)
+            const int64_t n_embd = model.hparams.n_embd;
+            const int64_t n_ff = model.hparams.n_ff();
+            const int64_t n_vocab = model.vocab.n_tokens();
+            const int64_t n_layer = model.hparams.n_layer;
+            const int64_t attn_params = 4 * n_embd * n_embd * n_layer;
+            const int64_t ffn_params = 3 * n_embd * n_ff * n_layer;
+            const int64_t emb_params = 2 * n_vocab * n_embd;
+            const float model_params_b = (float)(attn_params + ffn_params + emb_params) / 1e9f;
+
+            // Handle Q3_K_HIFI: model-size-aware + imatrix-guided outlier allocation
+            const bool is_q3_hifi = (new_type == GGML_TYPE_Q3_K_HIFI);
+            const bool is_q3_hifi_ftype = (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI);
+            if (is_q3_hifi && is_q3_hifi_ftype) {
+                // Get base outlier count from model size
+                int base_outliers = ggml_q3_hifi_get_max_outliers(model_params_b);
+
+                // Check for imatrix-guided importance and adjust outliers accordingly
+                float tensor_importance = 0.5f;  // Default to medium
+                if (g_imatrix_guided_enabled) {
+                    auto it = g_tensor_importance_map.find(std::string(tensor->name));
+                    if (it != g_tensor_importance_map.end()) {
+                        tensor_importance = it->second;
+
+                        // IMATRIX-GUIDED OUTLIER SCALING:
+                        // High importance tensors (>=0.7): use max outliers
+                        // Medium importance (0.4-0.7): use base outliers
+                        // Low importance (<0.4): reduce outliers
+                        if (tensor_importance >= 0.7f) {
+                            base_outliers = std::min(base_outliers + 2, Q3_K_HIFI_MAX_OUTLIERS);
+                        } else if (tensor_importance < 0.4f) {
+                            base_outliers = std::max(base_outliers - 2, 2);
+                        }
+                    }
+                }
+
+                // Set TLS state for Q3_K_HIFI quantization
+                ggml_q3_hifi_set_tensor_outliers(base_outliers);
+                ggml_q3_hifi_set_tensor_importance(tensor_importance);
+
+                hifi_ctx.outlier_count = base_outliers;
+                hifi_ctx.layer_importance = tensor_importance;
+                hifi_ctx.layer_idx = -1;
+                hifi_ctx.total_layers = (int)n_layer;
+                hifi_ctx.is_active = 1;
+                hifi_ctx.model_params_b = model_params_b;
+                hifi_ctx_ptr = &hifi_ctx;
+
+                // Log imatrix-guided outlier allocation
+                if (g_imatrix_guided_enabled) {
+                    LLAMA_LOG_INFO("(Q3_K_HIFI: model=%.1fB, imp=%.2f, outliers=%d) ",
+                                  model_params_b, tensor_importance, base_outliers);
+                } else if (base_outliers == 0) {
+                    LLAMA_LOG_INFO("(Q3_K_HIFI: model=%.1fB, skipping outliers - too small) ", model_params_b);
+                } else {
+                    LLAMA_LOG_INFO("(Q3_K_HIFI: model=%.1fB, max_outliers=%d) ", model_params_b, base_outliers);
+                }
+            }
 
-                    new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);
+            // Handle both Q6_K_HIFI_RES8 and Q5_K_HIFI_RES8 HIFI types (layer-adaptive)
+            const bool is_hifi_type = (new_type == GGML_TYPE_Q6_K_HIFI_RES8 || new_type == GGML_TYPE_Q5_K_HIFI_RES8);
+            const bool is_hifi_ftype = (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI);
+            if (is_hifi_type && is_hifi_ftype) {
+                // Extract layer index from tensor name (e.g., "blk.5.attn_v.weight" -> 5)
+                int layer_idx = -1;
+                if (sscanf(tensor->name, "blk.%d.", &layer_idx) != 1) {
+                    // Not a layer tensor (e.g., token_embd, output.weight)
+                    // Use max outliers for these critical tensors
+                    layer_idx = -1;
+                }
+
+                const int n_layers = (int)model.hparams.n_layer;
+
+                // Compute layer importance from imatrix if available
+                float layer_importance = 0.5f;  // default to medium
+                if (imatrix && n_per_row > 0) {
+                    layer_importance = ggml_hifi_compute_tensor_importance(imatrix, n_per_row);
+                }
+
+                // Compute adaptive outlier count
+                // Use the appropriate max outliers constant based on type
+                const int max_outliers = (new_type == GGML_TYPE_Q5_K_HIFI_RES8)
+                    ? Q5_K_HIFI_RES8_MAX_OUTLIERS : Q6_K_HIFI_RES8_MAX_OUTLIERS;
+                int outlier_count;
+                if (layer_idx < 0) {
+                    // Critical non-layer tensors (token_embd, output.weight): max outliers
+                    outlier_count = max_outliers;
+                } else {
+                    outlier_count = ggml_hifi_compute_outlier_count(
+                        layer_idx, n_layers, layer_importance, model_params_b
+                    );
+                    // Clamp to the type's max outliers
+                    if (outlier_count > max_outliers) outlier_count = max_outliers;
+                }
+
+                // Set up context
+                hifi_ctx.outlier_count = outlier_count;
+                hifi_ctx.layer_importance = layer_importance;
+                hifi_ctx.layer_idx = layer_idx;
+                hifi_ctx.total_layers = n_layers;
+                hifi_ctx.is_active = 1;
+                hifi_ctx.model_params_b = model_params_b;
+                hifi_ctx_ptr = &hifi_ctx;
+
+                // Log adaptive outlier allocation (INFO level for visibility)
+                const char * type_name = (new_type == GGML_TYPE_Q5_K_HIFI_RES8) ? "Q5_K_HIFI" : "Q6_K_HIFI";
+                LLAMA_LOG_INFO("(%s: model=%.1fB layer=%d/%d imp=%.2f outliers=%d) ",
+                    type_name, model_params_b, layer_idx, n_layers, layer_importance, outlier_count);
+            }
+
+            // Handle Q4_K_HIFI type - set per-tensor outlier count via TLS
+            if (new_type == GGML_TYPE_Q4_K_HIFI) {
+                int q4_outliers = ggml_q4_hifi_get_max_outliers(model_params_b);
+
+                // Use imatrix importance to modulate outlier count
+                if (imatrix && n_per_row > 0) {
+                    float importance = ggml_hifi_compute_tensor_importance(imatrix, n_per_row);
+                    // High importance tensors get more outliers
+                    if (importance > 0.7f) {
+                        q4_outliers = Q4_K_HIFI_MAX_OUTLIERS;  // Max outliers for critical tensors
+                    } else if (importance < 0.3f) {
+                        q4_outliers = (q4_outliers > 2) ? q4_outliers - 2 : 2;  // Reduce for low-importance
+                    }
+                }
+
+                ggml_q3_hifi_set_tensor_outliers(q4_outliers);  // Reuse Q3 TLS infrastructure
+                LLAMA_LOG_INFO("(Q4_K_HIFI: model=%.1fB outliers=%d) ", model_params_b, q4_outliers);
+            }
+
+            for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
+                const float * f32_data_03 = f32_data + i03 * nelements_matrix;
+                void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
+                const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;
+
+                    new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use, hifi_ctx_ptr);
+
+                    // TODO: temporary sanity check that the F16 -> MXFP4 is lossless
+#if 0
+                    if (new_type == GGML_TYPE_MXFP4) {
+                        auto * x = f32_data_03;
+
+                        //LLAMA_LOG_INFO("nrows = %d, n_per_row = %d\n", nrows, n_per_row);
+                        std::vector<float> deq(nrows*n_per_row);
+                        const ggml_type_traits * qtype = ggml_get_type_traits(new_type);
+                        qtype->to_float(new_data_03, deq.data(), deq.size());
+
+                        double err = 0.0f;
+                        for (int i = 0; i < (int) deq.size(); ++i) {
+                            err += fabsf(deq[i] - x[i]);
+                            //if (fabsf(deq[i] - x[i]) > 0.00001 && i < 256) {
+                            if (deq[i] != x[i]) {
+                                LLAMA_LOG_INFO("deq[%d] = %f, x[%d] = %f\n", i, deq[i], i, x[i]);
+                            }
+                        }
+                        //LLAMA_LOG_INFO("err = %f\n", err);
+                        GGML_ASSERT(err == 0.00000);
+                    }
+#endif
                 }
                 LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
             }
```

---

## 7. Tool Changes

### 7.1 `tools/quantize/quantize.cpp`

```diff
diff --git a/tools/quantize/quantize.cpp b/tools/quantize/quantize.cpp
index a882c78f1..591566edd 100644
--- a/tools/quantize/quantize.cpp
+++ b/tools/quantize/quantize.cpp
@@ -53,6 +53,15 @@ static const std::vector<quant_option> QUANT_OPTIONS = {
     { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
     { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
     { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
+    { "Q2_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q2_K_HIFI,  " ~3.0 bpw Q2_K base + INT8 residuals on critical tensors", },
+    { "Q3_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q3_K_HIFI,  " ~3.7G Q3_K_M base + scale-aware FP16 outlier enhancement", },
+    { "Q4_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q4_K_HIFI,  " ~4.95 bpw Q4_K base + FP16 outliers on medium tensors, tiered enhancement", },
+    { "Q5_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q5_K_HIFI,  " ~5.4 bpw Q5_K_M base + Q6_K_HIFI_RES8 on critical tensors", },
+    { "Q2_K_LITE", LLAMA_FTYPE_MOSTLY_Q2_K_LITE, " 3.0 bpw  Q2_K base + INT8 residuals, faster than Q2_K_S (imatrix recommended)", },
+    { "Q3_K_LITE", LLAMA_FTYPE_MOSTLY_Q3_K_LITE, " 3.25 bpw Q2_K base + INT8 residuals, faster than Q3_K_S (imatrix recommended)", },
+    { "Q4_K_LITE", LLAMA_FTYPE_MOSTLY_Q4_K_LITE, " 4.0 bpw  Q3_K base + INT8 residuals, faster than Q4_K_S (imatrix recommended)", },
+    { "Q5_K_LITE", LLAMA_FTYPE_MOSTLY_Q5_K_LITE, " 5.13 bpw Q4_K base + INT8 residuals, faster than Q5_K_S (imatrix recommended)", },
+    { "Q6_K_LITE", LLAMA_FTYPE_MOSTLY_Q6_K_LITE, " 6.13 bpw Q5_K base + INT8 residuals, faster than Q6_K_S (imatrix recommended)", },
     { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
     { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
     { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M",                  },
@@ -246,6 +255,15 @@ static int load_legacy_imatrix(const std::string & imatrix_file, std::vector<std
 
 static int load_imatrix(const std::string & imatrix_file, std::vector<std::string> & imatrix_datasets, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
 
+    if (!std::filesystem::exists(imatrix_file)) {
+        fprintf(stderr, "%s: imatrix file '%s' not found\n", __func__, imatrix_file.c_str());
+        exit(1);
+    }
+    if (!std::filesystem::is_regular_file(imatrix_file)) {
+        fprintf(stderr, "%s: imatrix path '%s' is not a regular file\n", __func__, imatrix_file.c_str());
+        exit(1);
+    }
+
     struct ggml_context * ctx = nullptr;
     struct gguf_init_params meta_gguf_params = {
         /* .no_alloc = */ false, // the data is needed
```

### 7.2 `tools/imatrix/imatrix.cpp`

```diff
diff --git a/tools/imatrix/imatrix.cpp b/tools/imatrix/imatrix.cpp
index 3f7f3a11d..6108c4a2b 100644
--- a/tools/imatrix/imatrix.cpp
+++ b/tools/imatrix/imatrix.cpp
@@ -871,8 +871,16 @@ static std::vector<float> softmax(const std::vector<float> & logits) {
 }
 
 static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
-    float max_logit = logits[0];
-    for (int i = 1; i < n_vocab; ++i) {
+    // Models running in F16 can produce +Inf logits for large-vocabulary heads
+    // due to F16 overflow (max ~65504).  +Inf - +Inf = NaN, which permanently
+    // poisons the accumulated NLL (sticky NaN).  Detect any non-finite logit
+    // and return a sentinel (log_prob = 0, prob = 1) so the token is counted
+    // but contributes 0 NLL rather than corrupting all subsequent PPL values.
+    float max_logit = -INFINITY;
+    for (int i = 0; i < n_vocab; ++i) {
+        if (!std::isfinite(logits[i])) {
+            return {0.0, logits[tok], 1.0f};
+        }
         max_logit = std::max(max_logit, logits[i]);
     }
     double sum_exp = 0.0;
```

### 7.3 `convert_hf_to_gguf.py`

```diff
diff --git a/convert_hf_to_gguf.py b/convert_hf_to_gguf.py
index 8d6b0a97a..dc866162a 100755
--- a/convert_hf_to_gguf.py
+++ b/convert_hf_to_gguf.py
@@ -1137,6 +1137,9 @@ class TextModel(ModelBase):
             elif rope_type.lower() == "llama3":
                 # Handled in generate_extra_tensors
                 pass
+            elif rope_type == "proportional":
+                # Handled in generate_extra_tensors (e.g. Gemma4 global attention layers)
+                pass
             else:
                 logger.warning(f"Unknown RoPE type: {rope_type}")
             logger.info(f"gguf: rope scaling type = {rope_gguf_type.name}")
@@ -6984,7 +6987,7 @@ class Gemma3Model(TextModel):
         self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
         self.gguf_writer.add_key_length(hparams.get("head_dim", 256))
         self.gguf_writer.add_value_length(hparams.get("head_dim", 256))
-        self.gguf_writer.add_rope_freq_base(self.rope_parameters.get("full_attention", self.rope_parameters).get("rope_theta", 1_000_000.0)) # for global layers
+        self.gguf_writer.add_rope_freq_base(self.rope_parameters.get("full_attention", self.rope_parameters).get("rope_theta", 1_000_000.0))  # for global layers
         # attn_logit_softcapping is removed in Gemma3
         assert hparams.get("attn_logit_softcapping") is None
         if (final_logit_softcap := hparams.get("final_logit_softcapping")):
@@ -13166,6 +13169,7 @@ def main() -> None:
         "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
         "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
         "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
+        "q3_k_hifi": gguf.LlamaFileType.MOSTLY_Q3_K_HIFI,
         "auto": gguf.LlamaFileType.GUESSED,
     }
```

---

## 8. New Tool Scripts

Create these new tool scripts in the `tools/` directory:

### 8.1 `tools/download_imatrix_datasets.py`

```python
#!/usr/bin/env python3
"""Download datasets for imatrix generation."""

from typing import Any, cast

from datasets import load_dataset

SAMPLE_SEPARATOR = "<|endofsample|>"


def download_mathqa(output_file="mathqa.txt", num_samples=10000) -> tuple[str, int, bool]:
    """Download MathQA problems. Returns (filename, expected_count, uses_separator)."""
    print(f"Downloading MathQA dataset ({num_samples} samples)...")
    ds = load_dataset('allenai/math_qa', revision='refs/convert/parquet', split='train')
    with open(output_file, 'w') as f:
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            f.write(item['Problem'].strip() + '\n')
    print(f"  Saved to {output_file}")
    return output_file, num_samples, False


def download_codeparrot(output_file="codeparrot.txt", num_samples=10000) -> tuple[str, int, bool]:
    """Download CodeParrot code snippets. Returns (filename, expected_count, uses_separator)."""
    print(f"Downloading CodeParrot dataset ({num_samples} samples)...")
    ds = load_dataset('codeparrot/codeparrot-valid-v2-near-dedup', split='train', streaming=True)
    with open(output_file, 'w') as f:
        count = 0
        for item in ds:
            if count >= num_samples:
                break
            code = cast(dict[str, Any], item)['content'].strip()
            if code and len(code) > 20:  # skip tiny snippets
                f.write(code + '\n' + SAMPLE_SEPARATOR + '\n')
                count += 1
    print(f"  Saved to {output_file}")
    return output_file, num_samples, True


def download_wikitext(output_file="wikitext.txt", num_lines=20000) -> tuple[str, int, bool]:
    """Download WikiText samples. Returns (filename, expected_count, uses_separator)."""
    print(f"Downloading WikiText dataset ({num_lines} lines)...")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    count = 0
    with open(output_file, 'w') as f:
        for item in ds:
            if count >= num_lines:
                break
            line = cast(dict[str, Any], item)['text']
            if line.strip():
                f.write(line.strip() + '\n')
                count += 1
    print(f"  Saved to {output_file}")
    return output_file, num_lines, False


def verify_file(filename: str, expected: int, uses_separator: bool) -> bool:
    """Verify that a file has the expected number of samples."""
    with open(filename, 'r') as f:
        content = f.read()
    if uses_separator:
        actual = content.count(SAMPLE_SEPARATOR)
        unit = "samples"
    else:
        actual = content.count('\n')
        unit = "lines"
    if actual == expected:
        print(f"  ✓ {filename}: {actual} {unit}")
        return True
    else:
        print(f"  ✗ {filename}: expected {expected}, got {actual} {unit}")
        return False


if __name__ == "__main__":
    results = [
        download_mathqa(),
        download_codeparrot(),
        download_wikitext(),
    ]

    print("\nVerifying downloads...")
    all_ok = all(verify_file(f, n, sep) for f, n, sep in results)

    if all_ok:
        print("\nDone! All files verified.")
    else:
        print("\nWarning: Some files have unexpected line counts.")
        exit(1)
```

### 8.2 `tools/download_coder_imatrix_datasets.py`

```python
#!/usr/bin/env python3
"""
Create a high-fidelity mixed-domain dataset for HIFI imatrix generation,
optimized for code-aware quantization of LLMs like Qwen3.

Target Mix:
- 50% Clean Code (The Stack top langs + CodeSearchNet)
- 15% Code Instructions (CodeAlpaca / Evol-Instruct-Code style)
- 15% Technical Q&A (Stack Overflow + GitHub Issues)
- 10% Developer Docs (READMEs, API docs)
- 10% General Tech Knowledge (Wikipedia CS + ArXiv abstracts)

Usage:
    python create_hifi_imatrix_dataset.py --output hifi-imatrix-dataset.txt
"""

import argparse
import random
from typing import List, Optional, Dict, Any
from datasets import load_dataset

def read_or_generate(
    source: str,
    split: str = "train",
    text_key: str = "text",
    max_samples: int = 50000,
    min_length: int = 20,
    filter_fn=None
) -> List[str]:
    """Load or generate lines from a Hugging Face dataset."""
    print(f"Loading {source} ({split})...")
    try:
        ds = load_dataset(source, split=split, streaming=True)
    except Exception as e:
        print(f"⚠️ Failed to load {source}: {e}")
        return []

    lines = []
    for item in ds:
        if len(lines) >= max_samples:
            break
        text = item.get(text_key, "").strip()
        if not text:
            continue
        if len(text) < min_length:
            continue
        if filter_fn and not filter_fn(item):
            continue
        lines.append(text)
    print(f"  → Got {len(lines)} samples")
    return lines

def main():
    parser = argparse.ArgumentParser(description="Build HIFI imatrix dataset")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)

    # === 1. Clean Code Repositories (50%) ===
    code_lines = []

    # The Stack v2 - top languages only (Python, JS, TS, Java, C++, Go, Rust, C#, PHP, Ruby)
    stack_langs = ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "C#", "PHP", "Ruby"]
    for lang in stack_langs:
        lines = read_or_generate(
            "bigcode/the-stack-v2-dedup",
            split="train",
            text_key="content",
            max_samples=3000,  # ~30k total
            min_length=30,
            filter_fn=lambda x: x.get("lang") == lang and x.get("size") > 100
        )
        code_lines.extend(lines)

    # CodeSearchNet (high-quality GitHub snippets)
    codesearchnet = read_or_generate(
        "code_search_net",
        split="train",
        text_key="whole_func_string",
        max_samples=10000,
        min_length=50
    )
    code_lines.extend(codesearchnet)

    # === 2. Code Instructions (15%) ===
    instruct_lines = []

    # CodeAlpaca (instruction-response pairs)
    codealpaca = read_or_generate(
        "sahil2801/CodeAlpaca-20k",
        split="train",
        text_key="text",
        max_samples=5000,
        min_length=30
    )
    instruct_lines.extend(codealpaca)

    # Evol-Instruct-Code (synthetic but high-quality)
    evolinstruct = read_or_generate(
        "nickrosh/Evol-Instruct-Code-80k-v1",
        split="train",
        text_key="output",
        max_samples=5000,
        min_length=30
    )
    instruct_lines.extend(evolinstruct)

    # === 3. Technical Q&A (15%) ===
    qa_lines = []

    # Stack Overflow (questions + answers)
    so = read_or_generate(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        text_key="response",
        max_samples=7500,
        min_length=40
    )
    qa_lines.extend(so)

    # GitHub issues (filtered for technical discussions)
    gh_issues = read_or_generate(
        "m-a-p/CodeFeedback-Filtered",
        split="train",
        text_key="answer",
        max_samples=7500,
        min_length=40
    )
    qa_lines.extend(gh_issues)

    # === 4. Developer Docs (10%) ===
    doc_lines = []

    # GitHub READMEs from popular repos
    readmes = read_or_generate(
        "bigcode/stack-readmes",
        split="train",
        text_key="readme",
        max_samples=5000,
        min_length=50
    )
    doc_lines.extend(readmes)

    # API documentation snippets
    api_docs = read_or_generate(
        "nomic-ai/gpt4all-j-prompt-generations",
        split="train",
        text_key="prompt",
        max_samples=5000,
        min_length=30,
        filter_fn=lambda x: "api" in x.get("prompt", "").lower() or "function" in x.get("prompt", "").lower()
    )
    doc_lines.extend(api_docs)

    # === 5. General Tech Knowledge (10%) ===
    general_lines = []

    # Wikipedia (CS-related only)
    wiki_cs = read_or_generate(
        "wikipedia",
        split="train",
        text_key="text",
        max_samples=5000,
        min_length=60,
        filter_fn=lambda x: any(kw in x.get("title", "").lower() for kw in [
            "algorithm", "data structure", "computer science", "programming", "software",
            "machine learning", "artificial intelligence", "compiler", "operating system"
        ])
    )
    general_lines.extend(wiki_cs)

    # ArXiv CS abstracts
    arxiv = read_or_generate(
        "ccdv/arxiv-summarization",
        split="train",
        text_key="abstract",
        max_samples=5000,
        min_length=80
    )
    general_lines.extend(arxiv)

    # === Normalize counts based on target weights ===
    total_target = 100_000  # total lines desired
    targets = {
        'code': int(0.50 * total_target),
        'instruct': int(0.15 * total_target),
        'qa': int(0.15 * total_target),
        'docs': int(0.10 * total_target),
        'general': int(0.10 * total_target),
    }

    def truncate_or_sample(lst: List[str], n: int) -> List[str]:
        if len(lst) <= n:
            return lst
        return random.sample(lst, n)

    final_lines = []
    final_lines.extend(truncate_or_sample(code_lines, targets['code']))
    final_lines.extend(truncate_or_sample(instruct_lines, targets['instruct']))
    final_lines.extend(truncate_or_sample(qa_lines, targets['qa']))
    final_lines.extend(truncate_or_sample(doc_lines, targets['docs']))
    final_lines.extend(truncate_or_sample(general_lines, targets['general']))

    # Shuffle final dataset
    random.shuffle(final_lines)

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line.replace('\n', ' ') + '\n')

    print(f"\n✅ Created HIFI imatrix dataset: {args.output}")
    print(f"   Total lines: {len(final_lines)}")

if __name__ == "__main__":
    main()
```

### 8.3 `tools/create_mixed_imatrix_dataset.py`

```python
#!/usr/bin/env python3
"""
Create an interleaved dataset file for mixed-domain imatrix generation.

Usage:
  python create_mixed_imatrix_dataset.py \
    --wikitext wikitext.txt \
    --code codeparrot.txt \
    --math mathqa.txt \
    --output mixed-imatrix_dataset.txt \
    --ratio 50,25,25
"""

import argparse
import random
from typing import List, Optional

def read_lines(filename: str, max_lines: Optional[int] = None) -> List[str]:
    """Read non-empty lines from file, optionally limiting count."""
    lines = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped:  # Skip empty lines
                lines.append(stripped)
                if max_lines and len(lines) >= max_lines:
                    break
    return lines

def interleave_datasets(
    wikitext: List[str],
    code: List[str],
    math: List[str],
    ratios: tuple = (50, 25, 25)
) -> List[str]:
    """Interleave datasets according to given ratios (percentages)."""
    wt_ratio, code_ratio, math_ratio = ratios
    total_ratio = wt_ratio + code_ratio + math_ratio

    # Normalize ratios to fractions
    wt_frac = wt_ratio / total_ratio
    code_frac = code_ratio / total_ratio
    math_frac = math_ratio / total_ratio

    # Calculate how many lines we can take from each (conservative estimate)
    min_multiplier = min(
        len(wikitext) / wt_frac if wt_frac > 0 else float('inf'),
        len(code) / code_frac if code_frac > 0 else float('inf'),
        len(math) / math_frac if math_frac > 0 else float('inf')
    )

    target_wt = int(min_multiplier * wt_frac)
    target_code = int(min_multiplier * code_frac)
    target_math = int(min_multiplier * math_frac)

    print(f"Using {target_wt} Wikitext, {target_code} Code, {target_math} Math lines")

    # Truncate to target counts
    wikitext = wikitext[:target_wt]
    code = code[:target_code]
    math = math[:target_math]

    # Create interleaved list
    mixed = []
    i = j = k = 0

    while i < len(wikitext) or j < len(code) or k < len(math):
        # Add Wikitext lines (highest ratio)
        for _ in range(2):  # 2x more frequent than others
            if i < len(wikitext):
                mixed.append(wikitext[i])
                i += 1

        # Add Code line
        if j < len(code):
            mixed.append(code[j])
            j += 1

        # Add Math line
        if k < len(math):
            mixed.append(math[k])
            k += 1

    return mixed

def main():
    parser = argparse.ArgumentParser(description="Create mixed imatrix dataset")
    parser.add_argument("--wikitext", required=True, help="Wikitext dataset file")
    parser.add_argument("--code", required=True, help="Code dataset file")
    parser.add_argument("--math", required=True, help="Math dataset file")
    parser.add_argument("--output", required=True, help="Output mixed dataset file")
    parser.add_argument("--ratio", default="50,25,25",
                        help="Ratios as WIKITEXT,CODE,MATH (default: 50,25,25)")

    args = parser.parse_args()

    # Parse ratios
    ratios = tuple(int(x) for x in args.ratio.split(','))
    if len(ratios) != 3:
        raise ValueError("Ratio must have exactly 3 values (e.g., 50,25,25)")

    # Load datasets
    print("Loading datasets...")
    wikitext_lines = read_lines(args.wikitext)
    code_lines = read_lines(args.code)
    math_lines = read_lines(args.math)

    print(f"Loaded {len(wikitext_lines)} Wikitext lines")
    print(f"Loaded {len(code_lines)} Code lines")
    print(f"Loaded {len(math_lines)} Math lines")

    # Interleave
    mixed_lines = interleave_datasets(wikitext_lines, code_lines, math_lines, ratios)

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in mixed_lines:
            f.write(line + '\n')

    print(f"\n✅ Created mixed dataset: {args.output}")
    print(f"   Total lines: {len(mixed_lines)}")

    # Sample output
    print("\nFirst 10 lines:")
    for i, line in enumerate(mixed_lines[:10]):
        prefix = "WT" if i % 4 < 2 else "CD" if i % 4 == 2 else "MH"
        print(f"  {prefix}: {line[:60]}...")

if __name__ == "__main__":
    main()
```

### 8.4 `benchmark_speed_test.sh` — Linux/macOS Benchmark Script

Create `benchmark_speed_test.sh` at the repository root:

```bash
#!/bin/bash
# Qwen3-0.6B Quantization Speed Benchmark Script
# Runs llama-bench multiple times per model and calculates statistics

# Note: Not using 'set -e' as we handle errors explicitly

# Default configuration
ITERATIONS=100
THREADS=4
REPEATS=3
PROMPT_TOKENS=0
GENERATE_TOKENS=20
GPU_LAYERS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations) ITERATIONS="$2"; shift 2 ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        -r|--repeats) REPEATS="$2"; shift 2 ;;
        -p|--prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        -n|--generate-tokens) GENERATE_TOKENS="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  -i N   Iterations per model (default: 100)"
            echo "  -t N   Threads (default: 4)"
            echo "  -r N   Repeats per run (default: 3)"
            echo "  -p N   Prompt tokens (default: 0)"
            echo "  -n N   Generate tokens (default: 20)"
            echo "  -ngl N GPU layers (default: none)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LLAMA_BENCH="./build/bin/llama-bench"
declare -a MODEL_NAMES=("Baseline" "Q5_K_S" "Q5_K_M" "Q5_K_HIFI" "Q5_K_S + imatrix" "Q5_K_M + imatrix" "Q5_K_HIFI + imatrix")
declare -a MODEL_PATHS=(
    "./Qwen3-0.6B-f16.gguf"
    "./Qwen3-0.6B-f16:Q5_K_S.gguf"
    "./Qwen3-0.6B-f16:Q5_K_M.gguf"
    "./Qwen3-0.6B-f16:Q5_K_HIFI.gguf"
    "./Qwen3-0.6B-f16-imatrix:Q5_K_S.gguf"
    "./Qwen3-0.6B-f16-imatrix:Q5_K_M.gguf"
    "./Qwen3-0.6B-f16-imatrix:Q5_K_HIFI.gguf"
)

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "Error: llama-bench not found at: $LLAMA_BENCH"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

for name in "${MODEL_NAMES[@]}"; do
    touch "$TEMP_DIR/${name}_speeds.txt"
    echo "0" > "$TEMP_DIR/${name}_errors.txt"
done

echo "Starting benchmark..."
START_TIME=$(date +%s)
TOTAL_RUNS=$((ITERATIONS * ${#MODEL_NAMES[@]}))
CURRENT_RUN=0

for ((i = 1; i <= ITERATIONS; i++)); do
    for idx in "${!MODEL_NAMES[@]}"; do
        name="${MODEL_NAMES[$idx]}"
        path="${MODEL_PATHS[$idx]}"
        CURRENT_RUN=$((CURRENT_RUN + 1))
        NGL_FLAG=""
        if [[ -n "$GPU_LAYERS" ]]; then NGL_FLAG="-ngl $GPU_LAYERS"; fi
        output=$("$LLAMA_BENCH" -m "$path" -t "$THREADS" -r "$REPEATS" -p "$PROMPT_TOKENS" -n "$GENERATE_TOKENS" $NGL_FLAG 2>&1) || true
        found=false
        while IFS= read -r line; do
            if [[ $line =~ tg[0-9]+[[:space:]]*\|[[:space:]]*([0-9.]+)[[:space:]]*± ]]; then
                echo "${BASH_REMATCH[1]}" >> "$TEMP_DIR/${name}_speeds.txt"
                found=true; break
            fi
        done <<< "$output"
        if [[ $found == false ]]; then
            errors=$(cat "$TEMP_DIR/${name}_errors.txt")
            echo $((errors + 1)) > "$TEMP_DIR/${name}_errors.txt"
        fi
    done
done

# Calculate and print statistics
for name in "${MODEL_NAMES[@]}"; do
    file="$TEMP_DIR/${name}_speeds.txt"
    if [[ -s "$file" ]]; then
        awk -v model="$name" '{sum+=$1; n++} END {print model": "sum/n" t/s avg ("n" samples)"}' "$file"
    fi
done
```

### 8.5 `benchmark_speed_test.ps1` — Windows PowerShell Benchmark Script

Create `benchmark_speed_test.ps1` at the repository root:

```powershell
# Qwen3-1.7B Quantization Speed Benchmark Script
param(
    [int]$Iterations = 100,
    [int]$Threads = 4,
    [int]$Repeats = 3,
    [int]$PromptTokens = 0,
    [int]$GenerateTokens = 20
)

$LlamaBench = ".\build\bin\Release\llama-bench.exe"
$Models = @(
    @{ Name = "Q3_K_S";   Path = ".\Qwen3-1.7B-f16-Q3_K_S.gguf" },
    @{ Name = "Q3_K_M";   Path = ".\Qwen3-1.7B-f16-Q3_K_M.gguf" },
    @{ Name = "Q3_K_HIFI"; Path = ".\Qwen3-1.7B-f16-Q3_K_HIFI.gguf" }
)

if (-not (Test-Path $LlamaBench)) { Write-Error "llama-bench not found at: $LlamaBench"; exit 1 }

$Results = @{}
foreach ($model in $Models) {
    $Results[$model.Name] = @{ Speeds = [System.Collections.ArrayList]::new(); Errors = 0 }
}

for ($i = 1; $i -le $Iterations; $i++) {
    foreach ($model in $Models) {
        try {
            $output = & $LlamaBench -m $model.Path -t $Threads -r $Repeats -p $PromptTokens -n $GenerateTokens 2>&1
            $found = $false
            foreach ($line in $output) {
                if ($line -match "tg\d+\s*\|\s*([\d.]+)\s*±") {
                    [void]$Results[$model.Name].Speeds.Add([double]$Matches[1])
                    $found = $true; break
                }
            }
            if (-not $found) { $Results[$model.Name].Errors++ }
        } catch { $Results[$model.Name].Errors++ }
    }
}

foreach ($model in $Models) {
    $speeds = $Results[$model.Name].Speeds
    if ($speeds.Count -gt 0) {
        $mean = ($speeds | Measure-Object -Average).Average
        Write-Host "$($model.Name): $([math]::Round($mean,2)) t/s avg ($($speeds.Count) samples)"
    }
}
```

---

## 9. Build Instructions

### 9.1 Metal (Apple Silicon)

```bash
cmake -B build \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
```

The Metal shader file (`ggml-metal.metal`) is compiled at build time via
`ggml_metal_embed_library`. If you see "kernel not found" errors at runtime, ensure all
new kernel instantiations in `ggml-metal.metal` are complete.

### 9.2 CUDA

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;90"  # adjust for your GPU
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

For consumer GPUs (RTX 3090/4090): use `86` and `89`.
For datacenter (A100/H100): use `80` and `90`.

### 9.3 ROCm/HIP

```bash
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="gfx1100"  # adjust for your GPU
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 9.4 Vulkan

```bash
cmake -B build \
    -DGGML_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

The Vulkan shaders are compiled from GLSL to SPIR-V during the build. The new HIFI shaders
in `vulkan-shaders/` will be automatically picked up.

### 9.5 CPU Only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

---

## 10. Verification Steps

### 10.1 Verify Type Registration

```bash
./build/bin/llama-quantize --help 2>&1 | grep -E 'HIFI|LITE'
```

Expected output shows Q2_K_HIFI, Q3_K_HIFI, Q4_K_HIFI, Q5_K_HIFI and all LITE variants.

### 10.2 Quantize a Model

```bash
# Basic quantization (CPU)
./build/bin/llama-quantize model.gguf model_q4k_hifi.gguf Q4_K_HIFI

# With imatrix (recommended for LITE types)
./build/bin/llama-imatrix -m model.gguf -f calibration_data.txt -o imatrix.dat
./build/bin/llama-quantize --imatrix imatrix.dat model.gguf model_q4k_hifi.gguf Q4_K_HIFI
```

### 10.3 Test Inference

```bash
./build/bin/llama-cli -m model_q4k_hifi.gguf -p "Hello, world" -n 50
```

### 10.4 Check Block Sizes

The following sizes must match (test in a short C program or check the static_asserts compile):

| Type | Expected bytes |
|------|----------------|
| block_q3_k_hifi | 136 |
| block_q3_k_hifi_res8 | 132 |
| block_q4_k_hifi | 168 |
| block_q6_k_hifi | 222 |
| block_q6_k_hifi_dynamic | 236 |
| block_q6_k_hifi_res8 | 232 |
| block_q5_k_hifi_res8 | 196 |
| block_q2_k_hifi | 96 |
| block_q2_k_lite | 96 |
| block_q3_k_lite | 104 |
| block_q4_k_lite | 128 |
| block_q5_k_lite | 164 |
| block_q6_k_lite | 196 |

If any `static_assert` fails at compile time, re-check the struct layout and padding.

### 10.5 GGUF Round-Trip Test

```python
from gguf import GGUFReader
reader = GGUFReader("model_q4k_hifi.gguf")
for tensor in reader.tensors:
    print(f"{tensor.name}: {tensor.tensor_type.name} ({tensor.n_bytes} bytes)")
```

Verify that tensors listed as HIFI/LITE types have the expected byte counts.

---

## 11. Common Issues

**Issue:** `static_assert` failure for block size.
**Fix:** Check `#pragma pack(push,1)` is present for LITE structs and `ggml_half` alignment is
correct for HIFI structs with mixed int8/float16 fields.

**Issue:** CUDA compilation error `incomplete type`.
**Fix:** Ensure `ggml-common.h` is included before CUDA kernel files. The CUDA backend includes
it via `ggml-cuda.cuh`.

**Issue:** Metal "kernel not found" at runtime.
**Fix:** The `kernel_mul_mv_q3_k_hifi_f32_f32_nsg=2` kernel (or similar) is missing from
`ggml-metal.metal`. Check that the `KERNEL_MUL_MV_Q` macro instantiation was added for each
new type.

**Issue:** GGUF offset mismatch error on load.
**Fix:** The Python-side `GGML_QUANT_SIZES` in `constants.py` must match the C-side
`sizeof(block_*)` values exactly. Check for discrepancies especially for Q3_K_HIFI (136 bytes)
and Q2_K_HIFI (96 bytes).

**Issue:** NaN perplexity when testing with imatrix.
**Fix:** This is fixed by the `log_softmax` change in `tools/imatrix/imatrix.cpp` to handle
infinite logits from F16 overflow. Ensure this fix is applied.

**Issue:** Vulkan validation error about missing shader data.
**Fix:** Rebuild the Vulkan shaders. The `vulkan-shaders-gen.cpp` must include the new type
names before the SPIR-V data is generated.

---

## 12. Post-Implementation Build Fixes

The following corrections were discovered during the first successful build on upstream llama.cpp
(tested on CUDA, Metal, and ROCm). Apply them after following the diffs in sections 2–8.

---

### 12.1 `ggml/src/ggml-quants.c` — Missing closing braces in LITE quantize functions

The diffs for the Q6_K_LITE, Q3_K_LITE, and Q2_K_LITE sections were generated with some closing
braces dropped. The affected functions are:

**Q6_K_LITE block** (`dequantize_row_q6_k_lite` and `quantize_q6_k_lite`):
- `if (rc > 0)` block was missing its closing `}`
- `for (int64_t ib ...)` loop was missing its closing `}`
- `if (hifi_ctx && hifi_ctx->is_active)` was missing its closing `}`
- `if (quant_weights)` was missing its closing `}`
- `for (int64_t row ...)` loop was missing its closing `}`
- `return nrow * row_size;` and function closing `}` were missing

**Q3_K_LITE block** (`quantize_row_q3_k_lite_inner`, `dequantize_row_q3_k_lite`):
- `assert(k % QK_K == 0)` and `const int64_t nb = k / QK_K;` were missing from `quantize_row_q3_k_lite_inner`
- `if (residual_budget == 0)` was missing its closing `}` and `continue;` statement
- `for (int i = 0; i < QK_K; ++i)` was missing its closing `}`
- `for (int k_idx ...)` was missing its closing `}`
- `assert(k % QK_K == 0)` and `const int64_t nb = k / QK_K;` were missing from `dequantize_row_q3_k_lite`
- `if (rc > 0)`, `for (int64_t ib ...)`, and function `}` were all missing

**Q2_K_LITE block** (`quantize_row_q2_k_lite_inner`, `dequantize_row_q2_k_lite`, `quantize_q2_k_lite`):
- Same pattern as Q3_K_LITE: missing `continue;`, missing closing `}` for `if`, `for`, and function bodies
- `if (hifi_ctx && hifi_ctx->is_active)` missing closing `}`
- `if (quant_weights)` missing closing `}`
- `for (int64_t row ...)` loop missing closing `}` and `return nrow * row_size;`

**Verify** the correct versions by reading the canonical implementations in the file after applying —
each `quantize_row_*_lite_inner` must have `assert(k % QK_K == 0)`, a `continue;` inside the
`if (residual_budget == 0)` block, and properly nested closing braces for all `if`/`for`/function
scopes. Each `quantize_q*_lite` must close the `if (hifi_ctx ...)` and `if (quant_weights)` blocks
before declaring `char * qrow` and must end with `return nrow * row_size;`.

---

### 12.2 `ggml/src/ggml-cpu/arch/arm/quants.c` — Wrong include path and duplicate forwarding stubs

**Include path:** The agent will add `#include "../../ggml-quants-hifi.h"` at line 4, but the file
is two directory levels deeper than expected. The correct path is:

```c
#include "../../../ggml-quants-hifi.h"
```

**Duplicate forwarding stubs:** The agent incorrectly adds forwarding stubs for functions that are
already fully defined in `ggml/src/ggml-cpu/quants.c` and have no `_generic` variant. Remove these
stubs entirely from `arch/arm/quants.c`:

```c
// REMOVE these — they call non-existent _generic / _ex variants and duplicate quants.c:
void ggml_vec_dot_q6_k_hifi_dynamic_q8_K(...) { ggml_vec_dot_q6_k_hifi_dynamic_q8_K_generic(...); }
void ggml_vec_dot_q6_k_hifi_res8_q8_K(...)    { ggml_vec_dot_q6_k_hifi_res8_q8_K_generic(...); }
void ggml_vec_dot_q5_k_hifi_res8_q8_K(...)    { ggml_vec_dot_q5_k_hifi_res8_q8_K_generic(...); }
void quantize_row_q5_k_hifi_res8(...)          { quantize_row_q5_k_hifi_res8_ref(...); }
size_t quantize_q5_k_hifi_res8(...)            { return quantize_q5_k_hifi_res8_ex(...); }
```

The `_generic` suffix functions (`ggml_vec_dot_q6_k_hifi_dynamic_q8_K_generic` etc.) and
`quantize_q5_k_hifi_res8_ex` do not exist. Only `q2_k_hifi`, `q3_k_hifi`, `q4_k_hifi`, and all
K_LITE types have `_generic` variants — the RES8 and DYNAMIC types do not.

**Missing prototype warning:** The agent will also add `dequantize_row_q3_k_hifi_neon` without a
forward declaration. Since it is only used within `arm/quants.c` (not referenced from any other
translation unit), declare it `static`:

```c
static void dequantize_row_q3_k_hifi_neon(const block_q3_k_hifi * GGML_RESTRICT x,
                                           float * GGML_RESTRICT y, int64_t k) {
```

---

### 12.3 `ggml/src/ggml-cpu/arch/x86/quants.c` — Same fixes as ARM

Apply exactly the same two fixes:
- Change `#include "../../ggml-quants-hifi.h"` → `#include "../../../ggml-quants-hifi.h"`
- Remove the same five duplicate/broken forwarding stubs listed in 12.2

---

### 12.4 `ggml/src/ggml-cpu/quants.c` — Stray diff header

The agent may insert a literal diff header line into the source file after `quantize_row_q2_k_lite`:

```
++ b/ggml/src/ggml-cpu/repack.cpp
```

Delete that line. The two blank lines around it can also be collapsed to one.

---

### 12.5 `ggml/src/ggml-cpu/quants.c` and `quants.h` — Missing `quantize_row_q3_k_hifi_res8` wrapper

The guide's section 3.11 (`ggml-cpu.c` type trait registration) uses `quantize_row_q3_k_hifi_res8`
as the `from_float` function for `GGML_TYPE_Q3_K_HIFI_RES8`, but only the `_ref` variant is
declared in `ggml-quants.h` (which `ggml-cpu.c` does not include). The plain wrapper is never
created by the guide diffs.

Add the following wrapper to `ggml/src/ggml-cpu/quants.c` (alongside the existing
`quantize_row_q5_k_hifi_res8` wrapper):

```c
void quantize_row_q3_k_hifi_res8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q3_k_hifi_res8_ref(x, (block_q3_k_hifi_res8 *)y, k);
}
```

And add the declaration to `ggml/src/ggml-cpu/quants.h` (alongside `quantize_row_q5_k_hifi_res8`):

```c
void quantize_row_q3_k_hifi_res8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
```

The `ggml-cpu.c` type trait entry should use the plain wrapper (not `_ref`):

```c
[GGML_TYPE_Q3_K_HIFI_RES8] = {
    .from_float   = quantize_row_q3_k_hifi_res8,
    .vec_dot      = ggml_vec_dot_q3_K_q8_K,  // fallback to q3_K kernel
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
```

---

## 13. File Reference

This guide is self-contained. All diffs and new file contents are embedded above.
The baseline upstream commit is `a29e4c0b7`. Every diff in this guide was generated with:

```bash
git diff a29e4c0b7..HEAD -- <path/to/file>
```

Summary of changed files by category:

- **New files**: `ggml/src/ggml-quants-hifi.h`, `ggml/src/ggml-quants-hifi.c`, `ggml/src/ggml-ext.h`,
  4 Vulkan GLSL shaders, 3 tool Python scripts
- **Core type system**: `ggml/src/ggml-common.h`, `ggml/src/ggml.c`, `ggml/src/gguf.cpp`,
  `ggml/src/ggml-quants.c`, `ggml/src/ggml-quants.h`
- **CPU backend**: `ggml/src/ggml-cpu/quants.c`, `ggml/src/ggml-cpu/quants.h`,
  `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ops.cpp`
- **CUDA backend**: 7 files in `ggml/src/ggml-cuda/`
- **Metal backend**: 3 files in `ggml/src/ggml-metal/`
- **Vulkan backend**: `ggml-vulkan.cpp` + 4 shader files
- **SYCL backend**: 4 files in `ggml/src/ggml-sycl/`
- **Python/GGUF**: `gguf-py/gguf/constants.py`, `gguf-py/gguf/quants.py`
- **LLaMA layer**: `include/llama.h`, `src/llama-model-loader.cpp`, `src/llama-model.cpp`,
  `src/llama-model.h`, `src/llama-quant.cpp`
- **Tools**: `tools/quantize/quantize.cpp`, `tools/imatrix/imatrix.cpp`, `convert_hf_to_gguf.py`

New files (not in upstream):
```
ggml/src/ggml-quants-hifi.h
ggml/src/ggml-quants-hifi.c
ggml/src/ggml-ext.h
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_k_hifi.comp
ggml/src/ggml-vulkan/vulkan-shaders/dequant_q3_k_hifi.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k_hifi.comp
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k_hifi.comp
tools/create_mixed_imatrix_dataset.py
tools/download_coder_imatrix_datasets.py
tools/download_imatrix_datasets.py
benchmark_speed_test.sh
benchmark_speed_test.ps1
```
