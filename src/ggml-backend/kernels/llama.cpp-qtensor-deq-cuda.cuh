// llama.cpp-qtensor-deq-cuda.cuh
// CUDA dequantize kernels for Q3_K_HIFI / Q4_K_HIFI tensor types

#pragma once

#include <cuda_runtime.h>
#include <half.h>

template<typename dst_t>
static __global__ void dequantize_block_q3_k_hifi(const void * __restrict__ vx,
                                                   dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const block_q3_k_hifi * x = (const block_q3_k_hifi *) vx;

    // Cast embedded Q3_K data to block_q3_K for standard 64-thread dequant
    const block_q3_K * q3k = (const block_q3_K *)x[i].q3_k_data;

    // --- Standard Q3_K 64-thread dequant ---
    const int tid = threadIdx.x;

    // Unpack scales
    static alignas(16) uint8_t ru[32];
    for (int j = 0; j < 16; ++j) {
        const int c = j / 4;
        const int bit = 3 * (j % 4);
        ru[32 * c + 6 * j + 0] = (q3k->m[c] >> bit) & 0x7;
        ru[32 * c + 6 * j + 1] = (q3k->m[c] >> bit) & 0x7;
        if (ru[32 * c + 6 * j + 0] > 4) ru[32 * c + 6 * j + 0] -= 8;
        if (ru[32 * c + 6 * j + 1] > 4) ru[32 * c + 6 * j + 1] -= 8;
    }

    // Dequantize and accumulate
    float sumf = 0.0f;
    for (int i1 = 0; i1 < 4; ++i1) {
        const int f = tid + 4 * i1;
        const int c = f / 8;
        const int j = f % 8;
        const int b = j / 2;
        const int bit = 6 * b;

        const int x0 = ((q3k->ql[2 * c * 16 + 8 * (j % 8) + (b & 1)] >> bit) & 0x3f);
        const int x1 = ((q3k->qh[f] >> 4) & 0xf) | (((x0 >> 2) & 0x3) << 4);
        const int x2 = ((q3k->qh[f] >> 0) & 0xf) | (((x0 >> 0) & 0x3) << 4);

        const float v0 = (x0 - 32) * ru[32 * c + 6 * j + 0];
        const float v1 = (x1 - 32) * ru[32 * c + 6 * j + 1];
        const float v2 = (x2 - 32) * ru[32 * c + 6 * (j + 8) + 0];

        yy[i * QK_K + 64 * i1 + tid] = dst_t(v0);
        yy[i * QK_K + 64 * i1 + 64 + tid] = dst_t(v1);
        yy[i * QK_K + 64 * i1 + 128 + tid] = dst_t(v2);

        sumf += v0 + v1 + v2;
    }
    __syncthreads();

    // Thread 0 replaces outlier positions with exact FP16 values.
    // Sentinel: unused slots have outliers[k]=FP16(0)=0x0000. Cannot use
    // idx>=256 as guard: sentinel idx=int(255) is < 256, so that check never fires.
    if (tid == 0) {
        dst_t * yb = yy + i * QK_K;
        for (int k = 0; k < Q3_K_HIFI_OUTLIERS; ++k) {
            if (__half_as_ushort(x[i].outliers[k]) == 0) break;  // FP16-zero sentinel
            const int idx = x[i].outlier_idx[k];
            yb[idx] = __half2float(x[i].outliers[k]);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q3_k_hifi_cuda(const void * vx, dst_t * y,
                                           const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
}
