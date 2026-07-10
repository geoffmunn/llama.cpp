#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

///-------------------------------------------------------------------
/// Table-driven type selection for HIFI quantisation paths
///
/// Replaces nested if/else chains that compare tensor_category values
/// (OUTPUT, TOKEN_EMBD, ATTENTION_V, ...) with explicit lookup tables
/// keyed by tensor_category enum.
///
/// This header declares the types.  The concrete resolver functions
/// and table instances live in llama-quant.cpp so that they can
/// reference tensor_category and the HIFI helper functions.
///-------------------------------------------------------------------

// Forward references
enum class tensor_category;
struct quantize_state_impl;

/// Function-pointer signature for per-category type resolvers.
/// Receives the estimated model size (billions of params) and a
/// pointer to the quantisation state (for layer counters, etc.).
using hifi_category_type_fn = ggml_type (*)(float model_params_b, const quantize_state_impl * qs);

/// One row in a category→type lookup table.
struct hifi_category_entry {
    tensor_category cat;
    hifi_category_type_fn select_type;
};

/// Top-level descriptor for a single HIFI ftype's category table.
struct hifi_ftype_config {
    llama_ftype ftype;
    const hifi_category_entry * entries;
    size_t n_entries;
};

#endif // TENSOR_OPS_H
