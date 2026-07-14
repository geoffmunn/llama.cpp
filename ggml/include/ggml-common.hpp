#pragma once

#include <cstdint>

// -----------------------------------------------------------------
// HIFI quantization types (C++ header)
// -----------------------------------------------------------------

typedef uint16_t ggml_half;

#ifndef Q3_K_HIFI_OUTLIERS
#define Q3_K_HIFI_OUTLIERS   8
#endif

#ifndef Q4_K_HIFI_OUTLIERS
#define Q4_K_HIFI_OUTLIERS   8
#endif

// Minimal forward definitions of standard K-block layouts for embedding.
// These match the full definitions in ggml-common.h and are sized identically.
#pragma pack(push, 1)
typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    ggml_half d;
} block_q3_K;
static_assert(sizeof(block_q3_K) == 110);

#pragma pack(pop)

#pragma pack(push, 1)
struct block_q3_k_hifi {
    block_q3_K   base;                              // standard Q3_K block (outlier positions zeroed)
    uint8_t  outlier_idx[Q3_K_HIFI_OUTLIERS];       // 8 indices (0–255), sorted ascending
    ggml_half outliers[Q3_K_HIFI_OUTLIERS];         // 8 FP16 replacement values
    uint8_t  outlier_count;                         // actual number of outliers stored (0–8)
    uint8_t  _pad;                                  // reserved; keep 136-byte total
};
#pragma pack(pop)

static_assert(sizeof(block_q3_k_hifi) == 110 + Q3_K_HIFI_OUTLIERS
              + Q3_K_HIFI_OUTLIERS * sizeof(ggml_half) + 2,
              "wrong q3_k_hifi block size/padding");
