#pragma once

#include <stdint.h>

// -----------------------------------------------------------------
// HIFI quantization types (C header)
// -----------------------------------------------------------------

typedef uint16_t ggml_half;

#ifndef Q3_K_HIFI_OUTLIERS
#define Q3_K_HIFI_OUTLIERS   8
#endif

#pragma pack(push, 1)
typedef struct {
    uint8_t  q3_k_data[110];                  // standard Q3_K block (outlier positions zeroed)
    uint8_t  outlier_idx[Q3_K_HIFI_OUTLIERS]; // 8 indices (0–255), sorted ascending
    ggml_half outliers[Q3_K_HIFI_OUTLIERS];   // 8 FP16 replacement values
    uint8_t  outlier_count;                   // actual number of outliers stored (0–8)
    uint8_t  _pad;                            // reserved; keep 136-byte total
} block_q3_k_hifi;
#pragma pack(pop)

/* 110 + 8 + 16 + 2 = 136 bytes */
