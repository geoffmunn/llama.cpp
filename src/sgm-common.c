// sgm-common.c — Shared quantization model-size utilities

#include <stddef.h>

// ------------------------------------------------------------------
// Model size categories (in billions of parameters)
// ------------------------------------------------------------------

typedef enum {
    MODEL_SIZE_TINY   = 0,  // < 2B
    MODEL_SIZE_MEDIUM = 1,  // 2B – 8B
    MODEL_SIZE_LARGE  = 2,  // >= 14B
} model_size_category;

// ------------------------------------------------------------------
// Classify a model by total parameter count (in billions).
// ------------------------------------------------------------------

static model_size_category get_model_size_category(float model_params_b) {
    if (model_params_b < 2.0f) {
        return MODEL_SIZE_TINY;
    } else if (model_params_b <= 8.0f) {
        return MODEL_SIZE_MEDIUM;
    } else {
        return MODEL_SIZE_LARGE;
    }
}

// ------------------------------------------------------------------
// Maximum outlier count for the given model size.
//
// Mapping (guide §7):
//   TINY (<2B)   -> 2  outliers  (flat weight distributions,
//                              extra outliers hurt accuracy)
//   MEDIUM (2-8B)-> 8  outliers  (genuine outliers exist)
//   LARGE (>=14B)-> 6  outliers  (outliers rarer per-block at scale)
// ------------------------------------------------------------------

int get_max_outliers_by_model_size(float model_params_b) {
    model_size_category cat = get_model_size_category(model_params_b);

    switch (cat) {
        case MODEL_SIZE_TINY:
            return 2;
        case MODEL_SIZE_MEDIUM:
            return 8;
        case MODEL_SIZE_LARGE:
            return 6;
        default:
            return 8;  // safe fallback for edge cases (9B–13B gap)
    }
}
