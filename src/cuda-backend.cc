// CUDA Backend — accumulation logic for HIFI/LITE quantization types
//
// Mirrors the CPU backend loop pattern:
//   1. Dequantize one HIFI/LITE block to float
//   2. Dequantize the corresponding Q8_K block to float
//   3. Accumulate the float dot product
//   4. Repeat for all blocks in the vector
//
// Optimized path for Q3_K_HIFI reuses the existing SIMD vec_dot on the
// embedded q3_k_data field, then adds outlier FMA corrections.

#include "backend-cpu.h"
#include <cassert>
#include <cstddef>

// -----------------------------------------------------------------------
// Instantiations — one vec_dot per HIFI/LITE type
// -----------------------------------------------------------------------

extern "C" {

// Q3_K_HIFI — uses the optimized path (reuses ggml_vec_dot_q3_K_q8_K
// on the embedded q3_k_data, then adds FP16 outlier FMA corrections)
void ggml_vec_dot_q3_k_hifi_q8_K(int n,
                                  float * GGML_RESTRICT s,
                                  size_t bs,
                                  const void * GGML_RESTRICT vx,
                                  size_t bx,
                                  const void * GGML_RESTRICT vy,
                                  size_t by,
                                  int nrc) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const block_q3_k_hifi * bx_hifi = (const block_q3_k_hifi *)vx;
    const block_q8_K       * by_q8  = (const block_q8_K *)vy;
    float result = 0.0f;
    for (int i = 0; i < nb; i++) {
        float block_dot = 0.0f;
        ggml_vec_dot_q3_K_q8_K(QK_K, &block_dot, sizeof(float),
                                bx_hifi[i].q3_k_data, sizeof(block_q3_K),
                                &by_q8[i], sizeof(block_q8_K), 1);
        result += block_dot;
        // block_q3_k_hifi has no outlier_count. Break on FP16-zero sentinel.
        const float q8d = by_q8[i].d;
        for (int j = 0; j < Q3_K_HIFI_OUTLIERS; j++) {
            ggml_half oval = bx_hifi[i].outliers[j];
            if (oval == 0) break;  // FP16-zero sentinel (ggml_half = uint16_t on CPU)
            const int pos = (int)bx_hifi[i].outlier_idx[j];
            result += GGML_FP16_TO_FP32(oval)
                    * (q8d * (float)by_q8[i].qs[pos]);
        }
    }
    *s = result;
}

// Q4_K_HIFI — generic dequantize + accumulate
HIFI_VEC_DOT_Q8K(q4_k_hifi, block_q4_k_hifi, dequantize_row_q4_k_hifi)

} // extern "C"
