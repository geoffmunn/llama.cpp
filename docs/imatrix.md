# Imatrix-Guided HiFi Quantization

## Pattern Invariant: Replace vs. Add

Two distinct outlier-correction patterns exist in the HIFI quantizer family:

| Pattern | Mechanism | Quantizer Types |
|---------|-----------|-----------------|
| **Replace** | FP16 values **replace** the base-dequant value entirely at outlier positions | Q3_K_HIFI, Q4_K_HIFI, Q6_K_HIFI, Q2_K_HIFI |
| **Add** | INT8 residuals **add** the correction to the base-dequant value | Q5_K_HIFI_RES8, Q6_K_HIFI_RES8, LITE variants |

### Prohibition

**Never mix the two patterns within the same quantizer implementation context.** A single quantizer type must use either replace or add — never both. Mixing violates the dequantization semantics and produces incorrect output.

This invariant is enforced by design: the struct layout, dequant kernel, and thread-local context all assume a single pattern per type. If you find yourself tempted to blend them, stop — the quantizer is already doing the right thing as-is.

---

## Imatrix Guidance Thresholds for Q3_K_HIFI

| Model size | Strategy |
|------------|----------|
| ≤2B | DISABLE imatrix-guided Q3_K_HIFI (hurts quality) |
| 2–5B | Top 30% of input projections |
| 5–10B | Top 20% |
| 10–20B | Top 15% |
| >20B | Top 10% |

---

## Critical Implementation Notes

1. **Outlier replacement vs residual addition.** FP16 variants (Q3_K_HIFI, Q4_K_HIFI,
   Q6_K_HIFI, Q2_K_HIFI) **replace** the base-dequant value entirely at outlier positions.
   INT8 residual variants (RES8, LITE) **add** the correction to the base-dequant value.
   Never mix the two patterns.

2. **Outlier positions zeroed before base quantization** (FP16 variants only).
   The base quantizer sees a zero at each outlier position, so its scale is not distorted
   by the extremes.

3. **Outlier indices sorted ascending.** Several backend kernels assume this. Always sort
   `outlier_idx[]` before storing.

4. **`block_q*_k_hifi` structs embed a complete standard block as their first field.**
   This means dequantizing the base portion can be done by casting the first N bytes to
   `block_q*_K *`. Preserve this layout invariant when adding new types.

5. **Thread-local context — two independent TLS variables.** The HIFI quantizer
   uses two separate thread-local storage slots:
   - `g_hifi_ctx` (set by `ggml_hifi_set_context`) — carries the full `ggml_hifi_quant_context`
     struct pointer. `llama_tensor_quantize_impl` sets this in *every* worker thread lambda.
   - `g_tensor_outliers` (set by `ggml_q3_hifi_set_tensor_outliers`) — a standalone int
     set only on the main thread before the worker pool is launched.
   `quantize_row_q3_k_hifi_ref` reads `g_tensor_outliers`, not `g_hifi_ctx->outlier_count`.
   Worker threads therefore see `g_tensor_outliers = 0` and fall through to the
   `Q3_K_HIFI_MAX_OUTLIERS` (8) default. This is the current shipping behaviour — do
   not "fix" it by switching to `g_hifi_ctx->outlier_count` without re-validating PPL,
   as empirical testing showed that properly applying fewer outliers (4) to all blocks
   produced *worse* perplexity than the accidental 8-outlier default for 0.8B models.

6. **Q5_K_HIFI_RES8 block size.** The `static_assert` in `ggml-common.h` is the authoritative
   value: 196 bytes. If your port defines `Q5_K_HIFI_RES8_BLOCK_SIZE` anywhere, ensure it
   matches 196. The `GGML_QUANT_SIZES` Python entry is also 196 (corrected during the
   2025 port). Any legacy source that shows 200 bytes is stale and wrong.
