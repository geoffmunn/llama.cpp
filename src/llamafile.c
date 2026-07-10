#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"

#include "ggml-quants.h"
#include <string.h>
#include <math.h>

// ------------------------------------------------------------------
// Q4_K_HIFI dot product with Q8_K
// ------------------------------------------------------------------

void ggml_vec_dot_q4_k_hifi_q8_K(int n, float * GGML_RESTRICT s,
                                  const void * GGML_RESTRICT vx,
                                  const void * GGML_RESTRICT vy) {
    const int nb = n / QK_K;
    const block_q4_k_hifi * x = vx;
    const block_q8_K       * y = vy;

    float total = 0.0f;
    for (int i = 0; i < nb; i++) {
        // Dequantize the embedded Q4_K block to a float buffer
        float w[QK_K];
        dequantize_row_q4_K((const block_q4_K *)x[i].q4_k_data, w, QK_K);

        const float dy = y[i].d;
        const int8_t * q8 = y[i].qs;

        float sum = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            sum += w[j] * (float)q8[j] * dy;
        }

        // Outlier correction: replace Q4K contribution with true FP16 value.
        // Break on FP16-zero sentinel (unused slots store outliers=0x0000).
        for (int k = 0; k < Q4_K_HIFI_OUTLIERS; k++) {
            ggml_half oval = x[i].outliers[k];
            if (oval == 0) break;
            int pos = (int)x[i].outlier_idx[k];
            sum += (GGML_FP16_TO_FP32(oval) - w[pos]) * (float)q8[pos] * dy;
        }

        total += sum;
    }
    *s = total;
}
