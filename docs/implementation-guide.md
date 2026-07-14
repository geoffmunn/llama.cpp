# Implementation Guide

## Quantization Types

### ⚠️ Sentinel Pitfall — `uint8_t` index guard is always true

**Do NOT guard outlier loops with `if (idx < block_size)` when `idx` is `uint8_t`
and `block_size == 256`.**  `uint8_t` holds values 0–255 and `256` is never reached,
so the condition is always true.  Unused outlier slots filled with sentinel
`(idx=255, outlier_val=0)` will silently pass the guard and corrupt the result:
- In dequantize: position 255 gets overwritten with 0.0f.
- In vec_dot (FP16-replacement types): position 255's base contribution is subtracted,
  corrupting the dot product every block.

**Correct approaches by type:**

| Type | Has `outlier_count`? | Correct loop bound |
|------|----------------------|--------------------|
| Q3_K_HIFI | ✓ | `int nc = MIN(block->outlier_count, MAX_OUTLIERS); for (k < nc)` |
| Q4_K_HIFI | ✗ | `for (k < OUTLIERS) { if (outliers[k]==0) break; ... }` |
| Q2_K_HIFI | ✓ (masked) | `int nc = block->outlier_count & 0x7F; nc = MIN(nc, MAX)` |

The RES8 types (Q5_K_HIFI_RES8, Q6_K_HIFI_RES8) use `outlier_count` correctly and are
not affected by this pitfall.
