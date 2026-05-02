#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

// HIFI dequant kernels — use byte-pointer arithmetic since GGML_COMMON_DECL_CUDA
// suppresses the HIFI struct definitions in CUDA translation units.

// block_q3_k_hifi byte layout (136 bytes):
//   [0..109]  = q3_k_data   (embedded block_q3_K, 110 bytes)
//   [110..117]= outlier_idx (uint8_t × 8)
//   [118..133]= outliers    (__half × 8)
//   [134]     = outlier_count
//   [135]     = _pad
static constexpr int Q3_K_HIFI_GPU_STRIDE    = 136;
static constexpr int Q3_K_HIFI_GPU_IDX_OFF   = 110;
static constexpr int Q3_K_HIFI_GPU_VALS_OFF  = 118;
static constexpr int Q3_K_HIFI_GPU_COUNT_OFF = 134;
static constexpr int Q3_K_HIFI_GPU_MAX_OUT   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q3_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q3_K_HIFI_GPU_STRIDE;
    const block_q3_K * x = (const block_q3_K *) blk;  // first 110 bytes = base Q3_K

    const int64_t r   = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0  = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n   = tid / 4;
    const int64_t j   = tid - 4*n;
    uint8_t m  = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift  = 2*j;
    int8_t us = is <  4 ? (x->scales[is-0] & 0xF) | (((x->scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x->scales[is-0] & 0xF) | (((x->scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x->scales[is-8] >>  4) | (((x->scales[is+0] >> 4) & 3) << 4) :
                          (x->scales[is-8] >>  4) | (((x->scales[is-4] >> 6) & 3) << 4);
    const float d_all = x->d;
    const float dl    = d_all * (us - 32);
    dst_t * y  = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q  = x->qs + 32*n;
    const uint8_t * hm = x->hmask;
    for (int l = l0; l < l0+4; ++l) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint8_t * idx_ptr = blk + Q3_K_HIFI_GPU_IDX_OFF;
        const __half  * val_ptr = (const __half *)(blk + Q3_K_HIFI_GPU_VALS_OFF);
        int nc = (int)(blk[Q3_K_HIFI_GPU_COUNT_OFF]);
        if (nc < 0 || nc > Q3_K_HIFI_GPU_MAX_OUT) nc = Q3_K_HIFI_GPU_MAX_OUT;
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            out[(int)idx_ptr[k]] = (dst_t)__half2float(val_ptr[k]);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q3_k_hifi_cuda(const void * vx, dst_t * y,
                                           const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
}

// block_q4_k_hifi byte layout (168 bytes):
//   [0..143]  = q4_k_data   (embedded block_q4_K, 144 bytes)
//   [144..151]= outlier_idx (uint8_t × 8)
//   [152..167]= outliers    (__half × 8)
// No outlier_count — uses FP16-zero sentinel.
static constexpr int Q4_K_HIFI_GPU_STRIDE   = 168;
static constexpr int Q4_K_HIFI_GPU_IDX_OFF  = 144;
static constexpr int Q4_K_HIFI_GPU_VALS_OFF = 152;
static constexpr int Q4_K_HIFI_GPU_MAX_OUT  = 8;

template<typename dst_t>
static __global__ void dequantize_block_q4_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q4_K_HIFI_GPU_STRIDE;
    const block_q4_K * x = (const block_q4_K *) blk;  // first 144 bytes = base Q4_K

    // Replicate dequantize_block_q4_K logic (32 threads)
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;
    dst_t * y = yy + i*QK_K + 64*il + n*ir;
    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * q = x->qs + 32*il + n*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = (dst_t)(d1 * (q[l] & 0xF) - m1);
        y[l + n] = (dst_t)(d2 * (q[l] >>  4) - m2);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint8_t * idx_ptr = blk + Q4_K_HIFI_GPU_IDX_OFF;
        const __half  * val_ptr = (const __half *)(blk + Q4_K_HIFI_GPU_VALS_OFF);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < Q4_K_HIFI_GPU_MAX_OUT; k++) {
            const __half v = val_ptr[k];
            // FP16-zero sentinel: unused slots store 0x0000
            if (__half_as_ushort(v) == 0) break;
            out[(int)idx_ptr[k]] = (dst_t)__half2float(v);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q4_k_hifi_cuda(const void * vx, dst_t * y,
                                           const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_k_hifi<<<nb, 32, 0, stream>>>(vx, y);
}

// E4M3FN FP8 → float (used for Q5_K_HIFI_RES8 residual scale)
__device__ static float e4m3fn_to_float(uint8_t b) {
    if ((b & 0x7F) == 0x7F) return 0.0f;  // NaN
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    float v = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
    return s ? -v : v;
}

// Q5_K_HIFI_RES8 — Q5_K base + INT8 residuals with E4M3 scale (196 bytes)
static constexpr int Q5_K_HIFI_RES8_GPU_STRIDE    = 196;
static constexpr int Q5_K_HIFI_RES8_GPU_COUNT_OFF = 176;
static constexpr int Q5_K_HIFI_RES8_GPU_IDX_OFF   = 177;
static constexpr int Q5_K_HIFI_RES8_GPU_VALS_OFF  = 185;
static constexpr int Q5_K_HIFI_RES8_GPU_E4M3_OFF  = 193;
static constexpr int Q5_K_HIFI_RES8_GPU_MAX_RES   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q5_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q5_K_HIFI_RES8_GPU_STRIDE;
    const block_q5_K * x = (const block_q5_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;
    const int64_t ir  = tid%16;
    const int64_t is  = 2*il;
    dst_t * y = yy + i*QK_K + 64*il + 2*ir;
    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * ql = x->qs + 32*il + 2*ir;
    const uint8_t * qh = x->qh + 2*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    uint8_t hm = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q5_K_HIFI_RES8_GPU_COUNT_OFF];
        if (nc > Q5_K_HIFI_RES8_GPU_MAX_RES) nc = Q5_K_HIFI_RES8_GPU_MAX_RES;
        const float rscale = e4m3fn_to_float(blk[Q5_K_HIFI_RES8_GPU_E4M3_OFF]);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q5_K_HIFI_RES8_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q5_K_HIFI_RES8_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q5_k_hifi_res8_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_k_hifi_res8<<<nb, 64, 0, stream>>>(vx, y);
}

// Q6_K_HIFI_RES8 — Q6_K base + INT8 residuals with float32 scale (232 bytes)
//   [210] count  [211..218] idx[8]  [219..226] vals[8]  [227] pad  [228..231] scale_f32
static constexpr int Q6_K_HIFI_RES8_GPU_STRIDE    = 232;
static constexpr int Q6_K_HIFI_RES8_GPU_COUNT_OFF = 210;
static constexpr int Q6_K_HIFI_RES8_GPU_IDX_OFF   = 211;
static constexpr int Q6_K_HIFI_RES8_GPU_VALS_OFF  = 219;
static constexpr int Q6_K_HIFI_RES8_GPU_SCALE_OFF = 228;
static constexpr int Q6_K_HIFI_RES8_GPU_MAX_RES   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q6_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q6_K_HIFI_RES8_GPU_STRIDE;
    const block_q6_K * x = (const block_q6_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;
    const int64_t il  = tid - 32*ip;
    const int64_t is  = 8*ip + il/16;
    dst_t * y = yy + i*QK_K + 128*ip + il;
    const float d = x->d;
    const uint8_t * ql = x->ql + 64*ip + il;
    const uint8_t   qh = x->qh[32*ip + il];
    const int8_t  * sc = x->scales + is;
    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q6_K_HIFI_RES8_GPU_COUNT_OFF];
        if (nc > Q6_K_HIFI_RES8_GPU_MAX_RES) nc = Q6_K_HIFI_RES8_GPU_MAX_RES;
        const float rscale = *(const float *)(blk + Q6_K_HIFI_RES8_GPU_SCALE_OFF);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q6_K_HIFI_RES8_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q6_K_HIFI_RES8_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q6_k_hifi_res8_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_k_hifi_res8<<<nb, 64, 0, stream>>>(vx, y);
}

// Q6_K_HIFI — Q6_K base + 4 FP16 outliers, zero-sentinel (222 bytes)
//   [210..213] idx[4]  [214..221] vals_f16[4]
static constexpr int Q6_K_HIFI_GPU_STRIDE   = 222;
static constexpr int Q6_K_HIFI_GPU_IDX_OFF  = 210;
static constexpr int Q6_K_HIFI_GPU_VALS_OFF = 214;
static constexpr int Q6_K_HIFI_GPU_MAX_OUT  = 4;

template<typename dst_t>
static __global__ void dequantize_block_q6_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q6_K_HIFI_GPU_STRIDE;
    const block_q6_K * x = (const block_q6_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;
    const int64_t il  = tid - 32*ip;
    const int64_t is  = 8*ip + il/16;
    dst_t * y = yy + i*QK_K + 128*ip + il;
    const float d = x->d;
    const uint8_t * ql = x->ql + 64*ip + il;
    const uint8_t   qh = x->qh[32*ip + il];
    const int8_t  * sc = x->scales + is;
    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);

    __syncthreads();
    if (threadIdx.x == 0) {
        const uint8_t * idx_ptr = blk + Q6_K_HIFI_GPU_IDX_OFF;
        const __half  * val_ptr = (const __half *)(blk + Q6_K_HIFI_GPU_VALS_OFF);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < Q6_K_HIFI_GPU_MAX_OUT; k++) {
            const __half v = val_ptr[k];
            if (__half_as_ushort(v) == 0) break;  // zero-sentinel
            out[(int)idx_ptr[k]] = (dst_t)__half2float(v);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q6_k_hifi_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
}

// Q6_K_HIFI_DYNAMIC — Q6_K base + 2-8 FP16 outliers with count (236 bytes)
//   [210] count  [211..218] idx[8]  [219] pad  [220..235] vals_f16[8]
static constexpr int Q6_K_HIFI_DYN_GPU_STRIDE    = 236;
static constexpr int Q6_K_HIFI_DYN_GPU_COUNT_OFF = 210;
static constexpr int Q6_K_HIFI_DYN_GPU_IDX_OFF   = 211;
static constexpr int Q6_K_HIFI_DYN_GPU_VALS_OFF  = 220;
static constexpr int Q6_K_HIFI_DYN_GPU_MAX_OUT   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q6_k_hifi_dynamic(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q6_K_HIFI_DYN_GPU_STRIDE;
    const block_q6_K * x = (const block_q6_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;
    const int64_t il  = tid - 32*ip;
    const int64_t is  = 8*ip + il/16;
    dst_t * y = yy + i*QK_K + 128*ip + il;
    const float d = x->d;
    const uint8_t * ql = x->ql + 64*ip + il;
    const uint8_t   qh = x->qh[32*ip + il];
    const int8_t  * sc = x->scales + is;
    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q6_K_HIFI_DYN_GPU_COUNT_OFF];
        if (nc > Q6_K_HIFI_DYN_GPU_MAX_OUT) nc = Q6_K_HIFI_DYN_GPU_MAX_OUT;
        const uint8_t * idx_ptr = blk + Q6_K_HIFI_DYN_GPU_IDX_OFF;
        const __half  * val_ptr = (const __half *)(blk + Q6_K_HIFI_DYN_GPU_VALS_OFF);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            out[(int)idx_ptr[k]] = (dst_t)__half2float(val_ptr[k]);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q6_k_hifi_dynamic_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_k_hifi_dynamic<<<nb, 64, 0, stream>>>(vx, y);
}

// Q3_K_HIFI_RES8 — Q3_K base + INT8 residuals with float32 scale (132 bytes)
//   [110] count  [111] pad  [112..119] idx[8]  [120..127] vals[8]  [128..131] scale_f32
static constexpr int Q3_K_HIFI_RES8_GPU_STRIDE    = 132;
static constexpr int Q3_K_HIFI_RES8_GPU_COUNT_OFF = 110;
static constexpr int Q3_K_HIFI_RES8_GPU_IDX_OFF   = 112;
static constexpr int Q3_K_HIFI_RES8_GPU_VALS_OFF  = 120;
static constexpr int Q3_K_HIFI_RES8_GPU_SCALE_OFF = 128;
static constexpr int Q3_K_HIFI_RES8_GPU_MAX_RES   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q3_k_hifi_res8(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q3_K_HIFI_RES8_GPU_STRIDE;
    const block_q3_K * x = (const block_q3_K *)blk;

    const int64_t r   = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0  = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n   = tid / 4;
    const int64_t j   = tid - 4*n;
    uint8_t m  = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift  = 2*j;
    int8_t us = is <  4 ? (x->scales[is-0] & 0xF) | (((x->scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x->scales[is-0] & 0xF) | (((x->scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x->scales[is-8] >>  4) | (((x->scales[is+0] >> 4) & 3) << 4) :
                          (x->scales[is-8] >>  4) | (((x->scales[is-4] >> 6) & 3) << 4);
    const float d_all = x->d;
    const float dl    = d_all * (us - 32);
    dst_t * y  = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q  = x->qs + 32*n;
    const uint8_t * hm = x->hmask;
    for (int l = l0; l < l0+4; ++l) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q3_K_HIFI_RES8_GPU_COUNT_OFF];
        if (nc > Q3_K_HIFI_RES8_GPU_MAX_RES) nc = Q3_K_HIFI_RES8_GPU_MAX_RES;
        const float rscale = *(const float *)(blk + Q3_K_HIFI_RES8_GPU_SCALE_OFF);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q3_K_HIFI_RES8_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q3_K_HIFI_RES8_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q3_k_hifi_res8_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_k_hifi_res8<<<nb, 64, 0, stream>>>(vx, y);
}

// Q2_K_HIFI — Q2_K base + 0-3 FP16 outliers with count (96 bytes, pack(1))
//   [84] count  [85..87] idx[3]  [88..93] vals_f16[3]  [94..95] pad
static constexpr int Q2_K_HIFI_GPU_STRIDE    = 96;
static constexpr int Q2_K_HIFI_GPU_COUNT_OFF = 84;
static constexpr int Q2_K_HIFI_GPU_IDX_OFF   = 85;
static constexpr int Q2_K_HIFI_GPU_VALS_OFF  = 88;
static constexpr int Q2_K_HIFI_GPU_MAX_OUT   = 3;

template<typename dst_t>
static __global__ void dequantize_block_q2_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q2_K_HIFI_GPU_STRIDE;
    const block_q2_K * x = (const block_q2_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;
    const uint8_t q = x->qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;
    float dall = __low2half(x->dm);
    float dmin = __high2half(x->dm);
    y[l+ 0] = dall * (x->scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x->scales[is+0] >> 4);
    y[l+32] = dall * (x->scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x->scales[is+2] >> 4);
    y[l+64] = dall * (x->scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x->scales[is+4] >> 4);
    y[l+96] = dall * (x->scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x->scales[is+6] >> 4);

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q2_K_HIFI_GPU_COUNT_OFF];
        if (nc > Q2_K_HIFI_GPU_MAX_OUT) nc = Q2_K_HIFI_GPU_MAX_OUT;
        const uint8_t * idx_ptr = blk + Q2_K_HIFI_GPU_IDX_OFF;
        const __half  * val_ptr = (const __half *)(blk + Q2_K_HIFI_GPU_VALS_OFF);
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            out[(int)idx_ptr[k]] = (dst_t)__half2float(val_ptr[k]);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q2_k_hifi_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
}

// Q2_K_LITE — Q2_K base + INT8 residuals with FP16 scale (96 bytes, pack(1))
//   [84] count  [85..88] idx[4]  [89..92] vals[4]  [93] pad  [94..95] scale_f16
static constexpr int Q2_K_LITE_GPU_STRIDE    = 96;
static constexpr int Q2_K_LITE_GPU_COUNT_OFF = 84;
static constexpr int Q2_K_LITE_GPU_IDX_OFF   = 85;
static constexpr int Q2_K_LITE_GPU_VALS_OFF  = 89;
static constexpr int Q2_K_LITE_GPU_SCALE_OFF = 94;
static constexpr int Q2_K_LITE_GPU_MAX_RES   = 4;

template<typename dst_t>
static __global__ void dequantize_block_q2_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q2_K_LITE_GPU_STRIDE;
    const block_q2_K * x = (const block_q2_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;
    const uint8_t q = x->qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;
    float dall = __low2half(x->dm);
    float dmin = __high2half(x->dm);
    y[l+ 0] = dall * (x->scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x->scales[is+0] >> 4);
    y[l+32] = dall * (x->scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x->scales[is+2] >> 4);
    y[l+64] = dall * (x->scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x->scales[is+4] >> 4);
    y[l+96] = dall * (x->scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x->scales[is+6] >> 4);

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q2_K_LITE_GPU_COUNT_OFF];
        if (nc > Q2_K_LITE_GPU_MAX_RES) nc = Q2_K_LITE_GPU_MAX_RES;
        const float rscale = __half2float(*(const __half *)(blk + Q2_K_LITE_GPU_SCALE_OFF));
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q2_K_LITE_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q2_K_LITE_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q2_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_k_lite<<<nb, 64, 0, stream>>>(vx, y);
}

// Q3_K_LITE — Q2_K base + INT8 residuals with FP16 scale (104 bytes, pack(1))
//   [84] count  [85..92] idx[8]  [93..100] vals[8]  [101] pad  [102..103] scale_f16
static constexpr int Q3_K_LITE_GPU_STRIDE    = 104;
static constexpr int Q3_K_LITE_GPU_COUNT_OFF = 84;
static constexpr int Q3_K_LITE_GPU_IDX_OFF   = 85;
static constexpr int Q3_K_LITE_GPU_VALS_OFF  = 93;
static constexpr int Q3_K_LITE_GPU_SCALE_OFF = 102;
static constexpr int Q3_K_LITE_GPU_MAX_RES   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q3_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q3_K_LITE_GPU_STRIDE;
    const block_q2_K * x = (const block_q2_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;
    const uint8_t q = x->qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;
    float dall = __low2half(x->dm);
    float dmin = __high2half(x->dm);
    y[l+ 0] = dall * (x->scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x->scales[is+0] >> 4);
    y[l+32] = dall * (x->scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x->scales[is+2] >> 4);
    y[l+64] = dall * (x->scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x->scales[is+4] >> 4);
    y[l+96] = dall * (x->scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x->scales[is+6] >> 4);

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q3_K_LITE_GPU_COUNT_OFF];
        if (nc > Q3_K_LITE_GPU_MAX_RES) nc = Q3_K_LITE_GPU_MAX_RES;
        const float rscale = __half2float(*(const __half *)(blk + Q3_K_LITE_GPU_SCALE_OFF));
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q3_K_LITE_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q3_K_LITE_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q3_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_k_lite<<<nb, 64, 0, stream>>>(vx, y);
}

// Q4_K_LITE — Q3_K base + INT8 residuals with FP16 scale (128 bytes, pack(1))
//   [110] count  [111..117] idx[7]  [118..124] vals[7]  [125] pad  [126..127] scale_f16
static constexpr int Q4_K_LITE_GPU_STRIDE    = 128;
static constexpr int Q4_K_LITE_GPU_COUNT_OFF = 110;
static constexpr int Q4_K_LITE_GPU_IDX_OFF   = 111;
static constexpr int Q4_K_LITE_GPU_VALS_OFF  = 118;
static constexpr int Q4_K_LITE_GPU_SCALE_OFF = 126;
static constexpr int Q4_K_LITE_GPU_MAX_RES   = 7;

template<typename dst_t>
static __global__ void dequantize_block_q4_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q4_K_LITE_GPU_STRIDE;
    const block_q3_K * x = (const block_q3_K *)blk;

    const int64_t r   = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0  = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n   = tid / 4;
    const int64_t j   = tid - 4*n;
    uint8_t m  = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift  = 2*j;
    int8_t us = is <  4 ? (x->scales[is-0] & 0xF) | (((x->scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x->scales[is-0] & 0xF) | (((x->scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x->scales[is-8] >>  4) | (((x->scales[is+0] >> 4) & 3) << 4) :
                          (x->scales[is-8] >>  4) | (((x->scales[is-4] >> 6) & 3) << 4);
    const float d_all = x->d;
    const float dl    = d_all * (us - 32);
    dst_t * y  = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q  = x->qs + 32*n;
    const uint8_t * hm = x->hmask;
    for (int l = l0; l < l0+4; ++l) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q4_K_LITE_GPU_COUNT_OFF];
        if (nc > Q4_K_LITE_GPU_MAX_RES) nc = Q4_K_LITE_GPU_MAX_RES;
        const float rscale = __half2float(*(const __half *)(blk + Q4_K_LITE_GPU_SCALE_OFF));
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q4_K_LITE_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q4_K_LITE_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q4_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_k_lite<<<nb, 64, 0, stream>>>(vx, y);
}

// Q5_K_LITE — Q4_K base + INT8 residuals with FP16 scale (164 bytes, pack(1))
//   [144] count  [145..152] idx[8]  [153..160] vals[8]  [161] pad  [162..163] scale_f16
static constexpr int Q5_K_LITE_GPU_STRIDE    = 164;
static constexpr int Q5_K_LITE_GPU_COUNT_OFF = 144;
static constexpr int Q5_K_LITE_GPU_IDX_OFF   = 145;
static constexpr int Q5_K_LITE_GPU_VALS_OFF  = 153;
static constexpr int Q5_K_LITE_GPU_SCALE_OFF = 162;
static constexpr int Q5_K_LITE_GPU_MAX_RES   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q5_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q5_K_LITE_GPU_STRIDE;
    const block_q4_K * x = (const block_q4_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;
    dst_t * y = yy + i*QK_K + 64*il + n*ir;
    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * q = x->qs + 32*il + n*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = (dst_t)(d1 * (q[l] & 0xF) - m1);
        y[l +32] = (dst_t)(d2 * (q[l] >>  4) - m2);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q5_K_LITE_GPU_COUNT_OFF];
        if (nc > Q5_K_LITE_GPU_MAX_RES) nc = Q5_K_LITE_GPU_MAX_RES;
        const float rscale = __half2float(*(const __half *)(blk + Q5_K_LITE_GPU_SCALE_OFF));
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q5_K_LITE_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q5_K_LITE_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q5_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_k_lite<<<nb, 32, 0, stream>>>(vx, y);
}

// Q6_K_LITE — Q5_K base + INT8 residuals with FP16 scale (196 bytes, pack(1))
//   [176] count  [177..184] idx[8]  [185..192] vals[8]  [193] pad  [194..195] scale_f16
static constexpr int Q6_K_LITE_GPU_STRIDE    = 196;
static constexpr int Q6_K_LITE_GPU_COUNT_OFF = 176;
static constexpr int Q6_K_LITE_GPU_IDX_OFF   = 177;
static constexpr int Q6_K_LITE_GPU_VALS_OFF  = 185;
static constexpr int Q6_K_LITE_GPU_SCALE_OFF = 194;
static constexpr int Q6_K_LITE_GPU_MAX_RES   = 8;

template<typename dst_t>
static __global__ void dequantize_block_q6_k_lite(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q6_K_LITE_GPU_STRIDE;
    const block_q5_K * x = (const block_q5_K *)blk;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;
    const int64_t ir  = tid%16;
    const int64_t is  = 2*il;
    dst_t * y = yy + i*QK_K + 64*il + 2*ir;
    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * ql = x->qs + 32*il + 2*ir;
    const uint8_t * qh = x->qh + 2*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    uint8_t hm = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;

    __syncthreads();
    if (threadIdx.x == 0) {
        int nc = (int)blk[Q6_K_LITE_GPU_COUNT_OFF];
        if (nc > Q6_K_LITE_GPU_MAX_RES) nc = Q6_K_LITE_GPU_MAX_RES;
        const float rscale = __half2float(*(const __half *)(blk + Q6_K_LITE_GPU_SCALE_OFF));
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            const int pos = (int)blk[Q6_K_LITE_GPU_IDX_OFF + k];
            out[pos] = (dst_t)((float)out[pos] + (float)(int8_t)blk[Q6_K_LITE_GPU_VALS_OFF + k] * rscale);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q6_k_lite_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_k_lite<<<nb, 64, 0, stream>>>(vx, y);
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_Q3_K_HIFI:
            return dequantize_row_q3_k_hifi_cuda;
        case GGML_TYPE_Q4_K_HIFI:
            return dequantize_row_q4_k_hifi_cuda;
        case GGML_TYPE_Q5_K_HIFI_RES8:
            return dequantize_row_q5_k_hifi_res8_cuda;
        case GGML_TYPE_Q6_K_HIFI_RES8:
            return dequantize_row_q6_k_hifi_res8_cuda;
        case GGML_TYPE_Q6_K_HIFI:
            return dequantize_row_q6_k_hifi_cuda;
        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
            return dequantize_row_q6_k_hifi_dynamic_cuda;
        case GGML_TYPE_Q3_K_HIFI_RES8:
            return dequantize_row_q3_k_hifi_res8_cuda;
        case GGML_TYPE_Q2_K_HIFI:
            return dequantize_row_q2_k_hifi_cuda;
        case GGML_TYPE_Q2_K_LITE:
            return dequantize_row_q2_k_lite_cuda;
        case GGML_TYPE_Q3_K_LITE:
            return dequantize_row_q3_k_lite_cuda;
        case GGML_TYPE_Q4_K_LITE:
            return dequantize_row_q4_k_lite_cuda;
        case GGML_TYPE_Q5_K_LITE:
            return dequantize_row_q5_k_lite_cuda;
        case GGML_TYPE_Q6_K_LITE:
            return dequantize_row_q6_k_lite_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_Q3_K_HIFI:
            return dequantize_row_q3_k_hifi_cuda;
        case GGML_TYPE_Q4_K_HIFI:
            return dequantize_row_q4_k_hifi_cuda;
        case GGML_TYPE_Q5_K_HIFI_RES8:
            return dequantize_row_q5_k_hifi_res8_cuda;
        case GGML_TYPE_Q6_K_HIFI_RES8:
            return dequantize_row_q6_k_hifi_res8_cuda;
        case GGML_TYPE_Q6_K_HIFI:
            return dequantize_row_q6_k_hifi_cuda;
        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
            return dequantize_row_q6_k_hifi_dynamic_cuda;
        case GGML_TYPE_Q3_K_HIFI_RES8:
            return dequantize_row_q3_k_hifi_res8_cuda;
        case GGML_TYPE_Q2_K_HIFI:
            return dequantize_row_q2_k_hifi_cuda;
        case GGML_TYPE_Q2_K_LITE:
            return dequantize_row_q2_k_lite_cuda;
        case GGML_TYPE_Q3_K_LITE:
            return dequantize_row_q3_k_lite_cuda;
        case GGML_TYPE_Q4_K_LITE:
            return dequantize_row_q4_k_lite_cuda;
        case GGML_TYPE_Q5_K_LITE:
            return dequantize_row_q5_k_lite_cuda;
        case GGML_TYPE_Q6_K_LITE:
            return dequantize_row_q6_k_lite_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
