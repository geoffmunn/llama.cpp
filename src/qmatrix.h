#pragma once

#include <cstdint>

// Q6_K_HIFI — 222 bytes  (4 outliers; used for critical tensors in Q4_K_HIFI ftype)

#define Q6_K_HIFI_OUTLIERS 4

typedef struct {
    // Q6_K-compatible region (210 bytes) — DO NOT REORDER
    uint8_t  ql[QK_K/2];       // 128 bytes: quants, lower 4 bits
    uint8_t  qh[QK_K/4];       //  64 bytes: quants, upper 2 bits
    int8_t   scales[QK_K/16];  //  16 bytes: scales, 8-bit
    ggml_half d;               //   2 bytes: super-block scale
    // Outlier extension (12 bytes)
    uint8_t  outlier_idx[Q6_K_HIFI_OUTLIERS];    // 4 bytes
    ggml_half outlier_vals[Q6_K_HIFI_OUTLIERS];  // 8 bytes
} block_q6_k_hifi;
// 210 + 12 = 222 bytes
static_assert(sizeof(block_q6_k_hifi) == sizeof(block_q6_K)
              + Q6_K_HIFI_OUTLIERS + Q6_K_HIFI_OUTLIERS * sizeof(ggml_half),
              "wrong q6_k_hifi block size/padding");

// Q6_K_HIFI_RES8 — 232 bytes  (INT8 residual correction)

#define Q6_K_HIFI_RES8_MAX_OUTLIERS 8
#define Q6_K_HIFI_RES8_BLOCK_SIZE   232

typedef struct {
    // Q6_K-compatible region (210 bytes)
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    ggml_half d;
    // INT8 residual extension (22 bytes)
    uint8_t outlier_count;                              // 1: actual count (1–8)
    uint8_t outlier_idx[Q6_K_HIFI_RES8_MAX_OUTLIERS];  // 8: positions (0–255)
    int8_t  residual_vals[Q6_K_HIFI_RES8_MAX_OUTLIERS];// 8: INT8 corrections
    uint8_t _padding;                                   // 1: float alignment
    float   residual_scale;                             // 4: shared scale
} block_q6_k_hifi_res8;
// 210 + 22 = 232 bytes
static_assert(sizeof(block_q6_k_hifi_res8) == 232,
              "wrong q6_k_hifi_res8 block size/padding");
