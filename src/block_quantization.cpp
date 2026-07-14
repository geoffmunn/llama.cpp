// Extended struct definition for block_q4_k_hifi
// Provides explicit field accessors alongside the opaque q4_k_data blob
// for use in vec_dot and dequantize implementations.
//
// Layout mirrors block_q4_K internals so that (block_q4_K *)bx->q4_k_data
// remains valid, while also exposing named arrays for outlier correction:
//   - qs[]          : 4-bit quantized values (QK_K/2 bytes)
//   - scales[]      : 6-bit quantized scales/mins (K_SCALE_SIZE bytes)
//   - d / dmin      : super-block scale and min
//   - outlier_idx[] : positions of FP16 outliers
//   - outliers[]    : FP16 replacement values (hifis_outliers / outsiders)
//
// The 'outsiders' alias refers to the FP16 outlier replacement values
// accessed as bx->outsiders[j] in vec_dot correction loops.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifndef Q4_K_HIFI_OUTLIERS
#define Q4_K_HIFI_OUTLIERS 8
#endif

#pragma pack(push, 1)
typedef struct {
    // --- Embedded Q4_K-compatible region (144 bytes) ---
    ggml_half  d;                                // super-block scale
    ggml_half  dmin;                             // super-block min scale
    uint8_t    scales[K_SCALE_SIZE];              // 6-bit quantized scales + mins
    uint8_t    qs[QK_K / 2];                      // 4-bit quantized weights

    // --- HIFI extension ---
    uint8_t    outlier_idx[Q4_K_HIFI_OUTLIERS];   // outlier positions (0-255)
    ggml_half  outliers[Q4_K_HIFI_OUTLIERS];      // FP16 replacement values
                                                    // aka hifis_outliers / outsiders
} block_q4_k_hifi_ext;
#pragma pack(pop)

static_assert(sizeof(block_q4_k_hifi_ext) == sizeof(ggml_half) * 2 + K_SCALE_SIZE + QK_K / 2
              + Q4_K_HIFI_OUTLIERS + Q4_K_HIFI_OUTLIERS * sizeof(ggml_half),
              "wrong q4_k_hifi_ext block size");
