#ifndef Q2_K_LITE_BLOCK_SIZE
#define Q2_K_LITE_BLOCK_SIZE 256
#endif

#ifndef Q3_K_HIFI_MAX_OUTLIERS
#define Q3_K_HIFI_MAX_OUTLIERS 8
#endif

#include "llama.h"
#include "ggml.h"

// Tensor-specific outlier count for Q3_K_HIFI quantization.
// Returns the number of top-N outliers to preserve as FP16 values per block.
int ggml_q3_hifi_get_tensor_outliers(void) {
    return Q3_K_HIFI_MAX_OUTLIERS;
}

// Convert a ggml_type to the corresponding llama_ftype for HIFI/LITE variants.
llama_ftype ggml_type_to_llama_ftype(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K_HIFI:          return LLAMA_FTYPE_MOSTLY_Q3_K_HIFI;
        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:  return LLAMA_FTYPE_MOSTLY_Q4_K_HIFI;
        case GGML_TYPE_Q6_K_HIFI_RES8:     return LLAMA_FTYPE_MOSTLY_Q4_K_HIFI;
        case GGML_TYPE_Q5_K_HIFI_RES8:     return LLAMA_FTYPE_MOSTLY_Q4_K_HIFI;
        case GGML_TYPE_Q4_K_HIFI:          return LLAMA_FTYPE_MOSTLY_Q4_K_HIFI;
        case GGML_TYPE_Q2_K_HIFI:          return LLAMA_FTYPE_MOSTLY_Q2_K_HIFI;
        case GGML_TYPE_Q2_K_LITE:          return LLAMA_FTYPE_MOSTLY_Q2_K_LITE;
        case GGML_TYPE_Q3_K_LITE:          return LLAMA_FTYPE_MOSTLY_Q3_K_LITE;
        case GGML_TYPE_Q4_K_LITE:          return LLAMA_FTYPE_MOSTLY_Q4_K_LITE;
        case GGML_TYPE_Q5_K_LITE:          return LLAMA_FTYPE_MOSTLY_Q5_K_LITE;
        case GGML_TYPE_Q6_K_LITE:          return LLAMA_FTYPE_MOSTLY_Q6_K_LITE;
        default:                           return (llama_ftype)0;
    }
}
