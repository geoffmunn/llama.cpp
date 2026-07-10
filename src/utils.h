#ifndef LLAMA_UTILS_H
#define LLAMA_UTILS_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

/**
 * @brief Create a temporary copy of a float row with outlier positions zeroed out.
 *
 * This routine allocates a new buffer containing a copy of the source row,
 * then sets the elements at each outlier index to zero.  The caller owns
 * the returned pointer and must free it with `free()` (or `utils_temp_buf_free`).
 *
 * Typical usage in the quantization pipeline:
 *   1. Identify outliers via `llama_find_top_outliers()`.
 *   2. Call `utils_create_temp_zeroed_outliers()` to get a zeroed copy.
 *   3. Quantize the zeroed copy with the standard base function.
 *   4. Store the original FP16 outlier values alongside the quantized block.
 *   5. Free the temporary buffer.
 *
 * @param src         pointer to the source weight row (n_per_row floats)
 * @param n_per_row   number of elements in the row
 * @param outlier_idx sorted vector of outlier element indices
 * @return            heap-allocated copy with outlier positions set to 0;
 *                    NULL on allocation failure or invalid input
 */
static inline float * utils_create_temp_zeroed_outliers(const float * src,
                                                        int64_t n_per_row,
                                                        const std::vector<int> & outlier_idx) {
    if (src == nullptr || n_per_row <= 0) {
        return nullptr;
    }

    float * tmp = static_cast<float *>(std::malloc(n_per_row * sizeof(float)));
    if (tmp == nullptr) {
        return nullptr;
    }

    // Copy the entire row
    std::memcpy(tmp, src, n_per_row * sizeof(float));

    // Zero out each outlier position
    for (int idx : outlier_idx) {
        if (idx >= 0 && idx < n_per_row) {
            tmp[idx] = 0.0f;
        }
    }

    return tmp;
}

/**
 * @brief Free a temporary buffer allocated by `utils_create_temp_zeroed_outliers`.
 *
 * A thin wrapper around `free()` for clarity at call sites.
 * Accepts NULL safely.
 *
 * @param buf pointer returned by `utils_create_temp_zeroed_outliers`, or NULL
 */
static inline void utils_temp_buf_free(float * buf) {
    std::free(buf);
}

#endif // LLAMA_UTILS_H
