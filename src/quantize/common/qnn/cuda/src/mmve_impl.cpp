// mmve_impl.cpp — device function for Q3_K_HIFI dot product with Q8_1

#include "ggml-cuda/common.cuh"
#include "ggml-cpu/ggml-common.h"

// VDR: same as base Q3_K
#define VDR_Q3_K_HIFI_Q8_1_MMVQ VDR_Q3_K_Q8_1_MMVQ

static inline float vec_dot_q3_k_hifi_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_q3_k_hifi * bq = (const block_q3_k_hifi *) vbq + kbx;
    const block_q3_K * q3k = (const block_q3_K *)bq->q3_k_data;

    // --- Q3_K base dot product ---
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const float d = q3k->d;

    const int vl = get_int_b2(q3k->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(q3k->hmask, iqs % (QI3_K / 2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    float sum = vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, q3k->scales, scale_offset, d, d8);

    // --- Outlier correction (additive; Q3K zero-level identity means base = 0 at outlier pos) ---
    for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
        const int idx = bq->outlier_idx[k];
        // sorted indices: once past this thread's range, done
        if (idx / QK8_1 >= bq8_offset + QR3_K) break;
        if (idx / QK8_1 < bq8_offset) continue;
        const float outlier_val = __half2float(bq->outliers[k]);
        // outlier_val==0 -> contribution 0, harmless for sentinel slots
        const int8_t q8_val = ((const int8_t*)bq8_1[idx / QK8_1].qs)[idx % QK8_1];
        sum += outlier_val * q8_val * __low2float(bq8_1[idx / QK8_1].ds);
    }

    return sum;
}
