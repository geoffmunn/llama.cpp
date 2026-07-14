#define Q3_K_LITE_BLOCK_SIZE    256
#define Q3_K_LITE_MAX_RESIDUALS 8

typedef struct {
    // Q2_K base (84 bytes)
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    GGML_EXTENSION union { ... } GGML_COMMON_AGGR_U;
    // INT8 extension (20 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q3_K_LITE_MAX_RESIDUALS];  // 8 bytes
    int8_t    residual_vals[Q3_K_LITE_MAX_RESIDUALS]; // 8 bytes
    uint8_t   _pad;
    ggml_half residual_scale;                          // 2 bytes
} block_q3_k_lite;
// 84 + 20 = 104 bytes
static_assert(sizeof(block_q3_k_lite) == 104, "wrong q3_k_lite block size/padding");
