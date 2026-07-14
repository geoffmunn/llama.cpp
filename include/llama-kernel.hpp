#pragma once

#include <cstdint>

// -----------------------------------------------------------------
// BY_Q8 accessor / descriptor helpers
// -----------------------------------------------------------------
// Outlier loops are terminated by a FP16-zero sentinel rather than
// an explicit outlier_count field.  This keeps the on-disk layout
// stable while removing the need to track a separate count.
// -----------------------------------------------------------------

typedef uint16_t ggml_half;

/// Return true when the given ggml_half value represents the
/// FP16 zero sentinel that terminates an outlier list.
inline bool is_outlier_sentinel(ggml_half v) {
    return v == static_cast<ggml_half>(0);
}

/// Iterate over the outlier arrays of a single HIFI block.
///
/// @param idx_arr   Pre-sorted outlier index array  (uint8_t *)
/// @param val_arr   FP16 outlier value array         (ggml_half *)
/// @param max       Maximum entries to consider (array length)
/// @param callback  Invoked for each valid outlier before the
///                  sentinel.  Signature: void(int pos, float val)
template <typename Callback>
inline void for_each_outlier(const uint8_t *idx_arr,
                             const ggml_half *val_arr,
                             int max,
                            Callback &&callback) {
    for (int j = 0; j < max; ++j) {
        if (is_outlier_sentinel(val_arr[j])) break;
        callback(static_cast<int>(idx_arr[j]),
                 GGML_FP16_TO_FP32(val_arr[j]));
    }
}
