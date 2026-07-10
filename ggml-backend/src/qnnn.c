#include "ggml-common.h"

#define Q3_K_HIFI_RES8_OUTLIERS 8

typedef struct {
    // Q3_K-compatible region (110 bytes) — DO NOT REORDER
    uint8_t  hmask[QK_K/8];  // 32 bytes
    uint8_t  qs[QK_K/4];     // 64 bytes
    uint8_t  scales[12];     // 12 bytes
    ggml_half d;             //  2 bytes
    // INT8 residual extension (22 bytes)
    uint8_t outlier_count;                           // 1
    uint8_t _pad1;                                   // 1 alignment
    uint8_t outlier_idx[Q3_K_HIFI_RES8_OUTLIERS];   // 8
    int8_t  residual_vals[Q3_K_HIFI_RES8_OUTLIERS]; // 8
    float   residual_scale;                          // 4
} block_q3_k_hifi_res8;
// 110 + 22 = 132 bytes
static_assert(sizeof(block_q3_k_hifi_res8)
              == sizeof(block_q3_K) + 2
              + Q3_K_HIFI_RES8_OUTLIERS + Q3_K_HIFI_RES8_OUTLIERS + sizeof(float),
              "wrong q3_k_hifi_res8 block size/padding");
