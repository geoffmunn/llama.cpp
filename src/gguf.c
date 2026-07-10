#ifndef Q2_K_LITE_BLOCK_SIZE
#define Q2_K_LITE_BLOCK_SIZE 256
#endif

#ifndef Q3_K_HIFI_MAX_OUTLIERS
#define Q3_K_HIFI_MAX_OUTLIERS 8
#endif

// Tensor-specific outlier count for Q3_K_HIFI quantization.
// Returns the number of top-N outliers to preserve as FP16 values per block.
int ggml_q3_hifi_get_tensor_outliers(void) {
    return Q3_K_HIFI_MAX_OUTLIERS;
}
