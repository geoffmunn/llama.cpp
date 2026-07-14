#pragma once

#include "../ggml/src/ggml-quants-hifi.h"
#include "gguf.h"

#ifndef GGUF_QUANT_H
#define GGUF_QUANT_H

/// High-importance threshold: tensors above this get full outlier budget
#ifndef HIGH_IMPORTANCE_THRESHOLD
#define HIGH_IMPORTANCE_THRESHOLD 0.75f
#endif

/// Compute per-tensor outlier count for Q4_K_HIFI based on model parameters
/// and optional importance matrix.
///
/// \param model_params_b  Model size in billions of parameters.
/// \param imatrix         Optional importance matrix data (per-element).
/// \param n_elements      Number of elements in the tensor.
/// \return                Maximum number of outliers for this tensor.
static inline int ggml_q4_k_get_max_outliers(float model_params_b,
                                             const float * imatrix,
                                             int64_t n_elements) {
    int q4_outliers = ggml_q4_hifi_get_max_outliers(model_params_b);

    if (imatrix != nullptr) {
        float importance = ggml_hifi_compute_tensor_importance(imatrix, n_elements);
        if (importance > HIGH_IMPORTANCE_THRESHOLD) {
            q4_outliers = Q4_K_HIFI_MAX_OUTLIERS;
        }
    }

    return q4_outliers;
}

/// Simpler overload without importance matrix — just model-size based.
static inline int ggml_q4_k_get_max_outliers(float model_params_b) {
    return ggml_q4_hifi_get_max_outliers(model_params_b);
}

#endif // GGUF_QUANT_H
