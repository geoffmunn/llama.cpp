# HIFI/LITE Quantization — Implementation Guide

This document is a complete technical specification for re-implementing the HIFI and LITE
quantization families on top of any future upstream `ggml-org/llama.cpp` version. It is
derived directly from the current source code and is intended as a **porting guide**, not
a user guide.

> **Note on discrepancies.** Where the Python helper in `gguf-py/gguf/quants.py` differs
> from the C `static_assert`, trust the `static_assert` — it is enforced at compile time.

---

## 1. Overview

Two custom quantization families are implemented:

| Family | Goal | Mechanism |
|--------|------|-----------|
| **HIFI** | Higher fidelity — replace or correct the worst weights | FP16 outlier replacement OR INT8 residual correction on top of a base type |
| **LITE** | Smaller+faster — one tier lighter than same-BPW standard type | INT8 residual correction on top of a *lower* base type |

Both share the same conceptual shape: a **base block** (standard Q2_K / Q3_K / Q4_K / Q5_K / Q6_K) followed by a compact **residual/outlier extension**.

---

## 2. Type Enumerations

### 2.1 `ggml_type` additions (`ggml/include/ggml.h`)

Add to the `ggml_type` enum, after the last existing upstream value:

```c
GGML_TYPE_Q3_K_HIFI        = 42,
GGML_TYPE_Q6_K_HIFI        = 43,
GGML_TYPE_Q6_K_HIFI_DYNAMIC = 44,
GGML_TYPE_Q6_K_HIFI_RES8   = 45,
GGML_TYPE_Q5_K_HIFI_RES8   = 46,
GGML_TYPE_Q3_K_HIFI_RES8   = 47,
GGML_TYPE_Q4_K_HIFI        = 48,
GGML_TYPE_Q2_K_HIFI        = 49,
GGML_TYPE_Q2_K_LITE        = 50,
GGML_TYPE_Q3_K_LITE        = 51,
GGML_TYPE_Q4_K_LITE        = 52,
GGML_TYPE_Q5_K_LITE        = 53,
GGML_TYPE_Q6_K_LITE        = 54,
```

### 2.2 `llama_ftype` additions (`include/llama.h`)

```c
// HIFI / LITE ftypes (44–52; 41–43 reserved — legacy HIFI ids removed)
LLAMA_FTYPE_MOSTLY_Q4_K_HIFI  = 44,
LLAMA_FTYPE_MOSTLY_Q3_K_HIFI  = 45,
LLAMA_FTYPE_MOSTLY_Q5_K_HIFI  = 46,
LLAMA_FTYPE_MOSTLY_Q2_K_HIFI  = 47,
LLAMA_FTYPE_MOSTLY_Q2_K_LITE  = 48,
LLAMA_FTYPE_MOSTLY_Q3_K_LITE  = 49,
LLAMA_FTYPE_MOSTLY_Q4_K_LITE  = 50,
LLAMA_FTYPE_MOSTLY_Q5_K_LITE  = 51,
LLAMA_FTYPE_MOSTLY_Q6_K_LITE  = 52,
```

### 2.3 Python constants (`gguf-py/gguf/constants.py`)

```python
class GGMLQuantizationType(IntEnum):
    # ... existing values ...
    Q3_K_HIFI        = 42
    Q6_K_HIFI        = 43
    Q6_K_HIFI_DYNAMIC = 44
    Q6_K_HIFI_RES8   = 45
    Q5_K_HIFI_RES8   = 46
    Q3_K_HIFI_RES8   = 47
    Q4_K_HIFI        = 48
    Q2_K_HIFI        = 49
    Q2_K_LITE        = 50
    Q3_K_LITE        = 51
    Q4_K_LITE        = 52
    Q5_K_LITE        = 53
    Q6_K_LITE        = 54

class LlamaFileType(IntEnum):
    # ... existing values ...
    MOSTLY_Q4_K_HIFI = 44
    MOSTLY_Q3_K_HIFI = 45
    MOSTLY_Q5_K_HIFI = 46
    MOSTLY_Q2_K_HIFI = 47
    MOSTLY_Q2_K_LITE = 48
    MOSTLY_Q3_K_LITE = 49
    MOSTLY_Q4_K_LITE = 50
    MOSTLY_Q5_K_LITE = 51
    MOSTLY_Q6_K_LITE = 52
```

Block size map additions for `gguf-py/gguf/constants.py` — added to the `GGML_QUANT_SIZES` dict
(use C `static_assert` sizes as truth; `quants.py` reads from this dict and needs no separate changes):

```python
GGMLQuantizationType.Q3_K_HIFI:        (256, 136),
GGMLQuantizationType.Q6_K_HIFI:        (256, 222),
GGMLQuantizationType.Q6_K_HIFI_DYNAMIC:(256, 236),
GGMLQuantizationType.Q6_K_HIFI_RES8:   (256, 232),
GGMLQuantizationType.Q5_K_HIFI_RES8:   (256, 196),
GGMLQuantizationType.Q3_K_HIFI_RES8:   (256, 132),
GGMLQuantizationType.Q4_K_HIFI:        (256, 168),
GGMLQuantizationType.Q2_K_HIFI:        (256, 96),
GGMLQuantizationType.Q2_K_LITE:        (256, 96),
GGMLQuantizationType.Q3_K_LITE:        (256, 104),
GGMLQuantizationType.Q4_K_LITE:        (256, 128),
GGMLQuantizationType.Q5_K_LITE:        (256, 164),
GGMLQuantizationType.Q6_K_LITE:        (256, 196),
```

---

## 3. Data Structures (`ggml/src/ggml-common.h`)

All structs go in `ggml-common.h` after the existing `block_q6_K` definition.
GPU-facing backends (CUDA/HIP/Metal) need the structs too, so they must live in
`ggml-common.h` which is included by all backends.

Use `#pragma pack(push, 1)` / `#pragma pack(pop)` guards as shown for the non-GPU
host path (the existing file already has the `GGML_COMMON_DECL_METAL`, `_CUDA`, `_HIP`
guards — mirror that pattern for the HIFI/LITE structs that contain fields needing
alignment control).

### 3.1 HIFI: FP16 Outlier Replacement Variants

These store the true FP16 values of the worst weights; dequantization **replaces**
(not adds) the base-type decoded value at those positions.

#### Q3_K_HIFI — 136 bytes

```c
#define Q3_K_HIFI_BLOCK_SIZE 256
#define Q3_K_HIFI_OUTLIERS   8
#define Q3_K_HIFI_INLIERS    248
#ifndef Q3_K_HIFI_MAX_OUTLIERS
#define Q3_K_HIFI_MAX_OUTLIERS 8
#endif

#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
#pragma pack(push, 1)
#endif
typedef struct {
    uint8_t  q3_k_data[110];                  // standard Q3_K block (outlier positions zeroed)
    uint8_t  outlier_idx[Q3_K_HIFI_OUTLIERS]; // 8 indices (0–255), sorted ascending
    ggml_half outliers[Q3_K_HIFI_OUTLIERS];   // 8 FP16 replacement values
    uint8_t  outlier_count;                   // actual number of outliers stored (0–8)
    uint8_t  _pad;                            // reserved; keep 136-byte total
} block_q3_k_hifi;
#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
#pragma pack(pop)
#endif
// 110 + 8 + 16 + 2 = 136 bytes
static_assert(sizeof(block_q3_k_hifi) == 110 + Q3_K_HIFI_OUTLIERS
              + Q3_K_HIFI_OUTLIERS * sizeof(ggml_half) + 2,
              "wrong q3_k_hifi block size/padding");
```

#### Q4_K_HIFI — 168 bytes

```c
#define Q4_K_HIFI_BLOCK_SIZE 256
#define Q4_K_HIFI_OUTLIERS   8
#define Q4_K_HIFI_INLIERS    248
#ifndef Q4_K_HIFI_MAX_OUTLIERS
#define Q4_K_HIFI_MAX_OUTLIERS 8
#endif

#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
#pragma pack(push, 1)
#endif
typedef struct {
    uint8_t  q4_k_data[144];                  // standard Q4_K block (outlier positions zeroed)
    uint8_t  outlier_idx[Q4_K_HIFI_OUTLIERS]; // 8 indices (0–255), sorted ascending
    ggml_half outliers[Q4_K_HIFI_OUTLIERS];   // 8 FP16 replacement values
} block_q4_k_hifi;
#if !defined(GGML_COMMON_DECL_METAL) && !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP)
#pragma pack(pop)
#endif
// 144 + 8 + 16 = 168 bytes → 5.25 BPW
static_assert(sizeof(block_q4_k_hifi) == 144 + Q4_K_HIFI_OUTLIERS
              + Q4_K_HIFI_OUTLIERS * sizeof(ggml_half),
              "wrong q4_k_hifi block size/padding");
```

#### Q6_K_HIFI — 222 bytes  (4 outliers; used for critical tensors in Q4_K_HIFI ftype)

```c
#define Q6_K_HIFI_OUTLIERS 4

typedef struct {
    // Q6_K-compatible region (210 bytes) — DO NOT REORDER
    uint8_t  ql[QK_K/2];       // 128 bytes: quants, lower 4 bits
    uint8_t  qh[QK_K/4];       //  64 bytes: quants, upper 2 bits
    int8_t   scales[QK_K/16];  //  16 bytes: scales, 8-bit
    ggml_half d;               //   2 bytes: super-block scale
    // Outlier extension (12 bytes)
    uint8_t  outlier_idx[Q6_K_HIFI_OUTLIERS];    // 4 bytes
    ggml_half outlier_vals[Q6_K_HIFI_OUTLIERS];  // 8 bytes
} block_q6_k_hifi;
// 210 + 12 = 222 bytes
static_assert(sizeof(block_q6_k_hifi) == sizeof(block_q6_K)
              + Q6_K_HIFI_OUTLIERS + Q6_K_HIFI_OUTLIERS * sizeof(ggml_half),
              "wrong q6_k_hifi block size/padding");
```

#### Q6_K_HIFI_DYNAMIC — 236 bytes  (2–8 dynamic outliers)

```c
#define Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS     8
#define Q6_K_HIFI_DYNAMIC_MIN_OUTLIERS     2
#define Q6_K_HIFI_DYNAMIC_DEFAULT_OUTLIERS 6
#define Q6_K_HIFI_EARLY_EXIT_THRESHOLD     4

typedef struct {
    // Q6_K-compatible region (210 bytes)
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    ggml_half d;
    // Dynamic outlier extension (26 bytes)
    uint8_t  outlier_count;                                     // 1 byte: actual count (2–8)
    uint8_t  outlier_idx[Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS];      // 8 bytes
    uint8_t  _padding;                                         // 1 byte: align for ggml_half
    ggml_half outlier_vals[Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS];   // 16 bytes
} block_q6_k_hifi_dynamic;
// 210 + 2 + 8 + 16 = 236 bytes
static_assert(sizeof(block_q6_k_hifi_dynamic) == sizeof(block_q6_K) + 2
              + Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS
              + Q6_K_HIFI_DYNAMIC_MAX_OUTLIERS * sizeof(ggml_half),
              "wrong q6_k_hifi_dynamic block size/padding");
```

### 3.2 HIFI: INT8 Residual Variants (RES8)

These add a correction after dequantization:
`decoded[i] = base_dequant[i] + residual_scale * (residual_vals[j] / 127.0f)`

#### Q6_K_HIFI_RES8 — 232 bytes

```c
#define Q6_K_HIFI_RES8_MAX_OUTLIERS 8
#define Q6_K_HIFI_RES8_BLOCK_SIZE   232

typedef struct {
    // Q6_K-compatible region (210 bytes)
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    ggml_half d;
    // INT8 residual extension (22 bytes)
    uint8_t outlier_count;                              // 1: actual count (1–8)
    uint8_t outlier_idx[Q6_K_HIFI_RES8_MAX_OUTLIERS];  // 8: positions (0–255)
    int8_t  residual_vals[Q6_K_HIFI_RES8_MAX_OUTLIERS];// 8: INT8 corrections
    uint8_t _padding;                                   // 1: float alignment
    float   residual_scale;                             // 4: shared scale
} block_q6_k_hifi_res8;
// 210 + 22 = 232 bytes
static_assert(sizeof(block_q6_k_hifi_res8) == 232,
              "wrong q6_k_hifi_res8 block size/padding");
```

#### Q5_K_HIFI_RES8 — 196 bytes

Uses a 1-byte E4M3 FP8 scale instead of 4-byte FP32 to save space.

```c
#define Q5_K_HIFI_RES8_MAX_OUTLIERS 8
#define Q5_K_HIFI_RES8_BLOCK_SIZE   196  // ← authoritative (static_assert)

typedef struct {
    // Q5_K-compatible region (176 bytes)
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // 12 bytes
    uint8_t qh[QK_K/8];           // 32 bytes: high bit
    uint8_t qs[QK_K/2];           // 128 bytes: low 4 bits
    // Compact INT8 residual extension (20 bytes)
    uint8_t outlier_count;                               // 1: actual count (0–8; 0 = no enhancement)
    uint8_t outlier_idx[Q5_K_HIFI_RES8_MAX_OUTLIERS];    // 8: positions
    int8_t  residual_vals[Q5_K_HIFI_RES8_MAX_OUTLIERS];  // 8: INT8 corrections
    uint8_t residual_scale_e4m3;                         // 1: E4M3 FP8 scale
    uint8_t _reserved[2];                                // 2: explicit padding so pack(1) still = 196
} block_q5_k_hifi_res8;
// 176 + 20 = 196 bytes (1+8+8+1+2 = 20 in extension)
static_assert(sizeof(block_q5_k_hifi_res8) == 196,
              "wrong q5_k_hifi_res8 block size/padding");
```

#### Q3_K_HIFI_RES8 — 132 bytes

```c
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
```

#### Q2_K_HIFI — 96 bytes  (FP16 outlier replacement, not residual)

```c
#define Q2_K_HIFI_BLOCK_SIZE     256
#define Q2_K_HIFI_MAX_OUTLIERS   3
#define Q2_K_HIFI_RESIDUAL_MODE_FLAG 0x80

typedef struct {
    // Q2_K-compatible region (84 bytes) — DO NOT REORDER
    uint8_t scales[QK_K/16];  // 16 bytes
    uint8_t qs[QK_K/4];       // 64 bytes
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    // FP16 outlier extension (12 bytes)
    uint8_t   outlier_count;                       // 1: actual count (0–3)
    uint8_t   outlier_idx[Q2_K_HIFI_MAX_OUTLIERS]; // 3: positions (0–255)
    ggml_half outlier_vals[Q2_K_HIFI_MAX_OUTLIERS]; // 6: FP16 replacement values
    uint8_t   _pad[2];                             // 2: align to 96 bytes
} block_q2_k_hifi;
// 84 + 12 = 96 bytes → 3.0 BPW
static_assert(sizeof(block_q2_k_hifi) == 96,
              "wrong q2_k_hifi block size/padding");
```

### 3.3 LITE Family: INT8 Residual on Lower Base

All LITE structs use the same extension pattern. The base type is **one tier lower** than
the name implies, so the block is smaller and faster:

```
Q2_K_LITE → Q2_K base  + 4 INT8 residuals →  96 bytes (3.0 BPW)
Q3_K_LITE → Q2_K base  + 8 INT8 residuals → 104 bytes (3.25 BPW)
Q4_K_LITE → Q3_K base  + 7 INT8 residuals → 128 bytes (4.0 BPW)
Q5_K_LITE → Q4_K base  + 8 INT8 residuals → 164 bytes (5.125 BPW)
Q6_K_LITE → Q5_K base  + 8 INT8 residuals → 196 bytes (6.125 BPW)
```

Residual reconstruction: `decoded[i] += ggml_half_to_float(residual_scale) * (residual_vals[j] / 127.0f)`

All LITE structs must be wrapped in `#pragma pack(push/pop, 1)` guards:

```c
#define Q2_K_LITE_BLOCK_SIZE    256
#define Q2_K_LITE_MAX_RESIDUALS 4

typedef struct {
    // Q2_K base (84 bytes)
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    // INT8 extension (12 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q2_K_LITE_MAX_RESIDUALS];  // 4 bytes
    int8_t    residual_vals[Q2_K_LITE_MAX_RESIDUALS]; // 4 bytes
    uint8_t   _pad;
    ggml_half residual_scale;                          // 2 bytes
} block_q2_k_lite;
// 84 + 12 = 96 bytes
static_assert(sizeof(block_q2_k_lite) == 96, "wrong q2_k_lite block size/padding");

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

#define Q4_K_LITE_BLOCK_SIZE    256
#define Q4_K_LITE_MAX_RESIDUALS 7

typedef struct {
    // Q3_K base (110 bytes): hmask[32] + qs[64] + scales[12] + d[2]
    uint8_t   hmask[QK_K/8];
    uint8_t   qs[QK_K/4];
    uint8_t   scales[K_SCALE_SIZE];
    ggml_half d;
    // INT8 extension (18 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q4_K_LITE_MAX_RESIDUALS];  // 7 bytes
    int8_t    residual_vals[Q4_K_LITE_MAX_RESIDUALS]; // 7 bytes
    uint8_t   _pad;
    ggml_half residual_scale;                          // 2 bytes
} block_q4_k_lite;
// 110 + 18 = 128 bytes
static_assert(sizeof(block_q4_k_lite) == 128, "wrong q4_k_lite block size/padding");

#define Q5_K_LITE_BLOCK_SIZE    256
#define Q5_K_LITE_MAX_RESIDUALS 8

typedef struct {
    // Q4_K base (144 bytes): dm[4] + scales[12] + qs[128]
    GGML_EXTENSION union { ... } GGML_COMMON_AGGR_U;
    uint8_t scales[3*QK_K/64];
    uint8_t qs[QK_K/2];
    // INT8 extension (20 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q5_K_LITE_MAX_RESIDUALS];  // 8 bytes
    int8_t    residual_vals[Q5_K_LITE_MAX_RESIDUALS]; // 8 bytes
    uint8_t   _pad;
    ggml_half residual_scale;
} block_q5_k_lite;
// 144 + 20 = 164 bytes
static_assert(sizeof(block_q5_k_lite) == 164, "wrong q5_k_lite block size/padding");

#define Q6_K_LITE_BLOCK_SIZE    256
#define Q6_K_LITE_MAX_RESIDUALS 8

typedef struct {
    // Q5_K base (176 bytes): dm[4] + scales[12] + qh[32] + qs[128]
    GGML_EXTENSION union { ... } GGML_COMMON_AGGR_U;
    uint8_t scales[3*QK_K/64];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
    // INT8 extension (20 bytes)
    uint8_t   residual_count;
    uint8_t   residual_idx[Q6_K_LITE_MAX_RESIDUALS];  // 8 bytes
    int8_t    residual_vals[Q6_K_LITE_MAX_RESIDUALS]; // 8 bytes
    uint8_t   _pad;
    ggml_half residual_scale;
} block_q6_k_lite;
// 176 + 20 = 196 bytes
static_assert(sizeof(block_q6_k_lite) == 196, "wrong q6_k_lite block size/padding");
```

---

## 4. New Files

### 4.1 `ggml/src/ggml-quants-hifi.h` and `ggml-quants-hifi.c`

These files implement the layer-adaptive quantization context API. Add them to the
GGML source list in `ggml/src/CMakeLists.txt` alongside the existing `ggml-quants.c`.

**Key API (`ggml-quants-hifi.h`):**

> **Include path pitfall**: `ggml-quants-hifi.h` must use `#include "ggml.h"` (not
> `#include "ggml/ggml.h"`). Files in `ggml/src/` are built with `ggml/include/` as the
> root, so `ggml.h` resolves correctly but `ggml/ggml.h` does not exist on that path.

```c
// Thread-local context passed into quantization functions
typedef struct {
    int   outlier_count;     // 1–8; number of outliers to preserve
    float layer_importance;  // 0.0–1.0; from imatrix aggregation
    int   layer_idx;         // current layer (debugging)
    int   total_layers;      // total model layers (debugging)
    int   is_active;         // 1 = adaptive mode on
    float model_params_b;    // model size in billions
} ggml_hifi_quant_context;

GGML_API const ggml_hifi_quant_context * ggml_hifi_get_context(void);
GGML_API void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx);

GGML_API int   ggml_hifi_compute_outlier_count(int layer_idx, int total_layers,
                                                float layer_importance, float model_params_b);
GGML_API float ggml_hifi_compute_tensor_importance(const float * imatrix_data, int64_t n_elements);
GGML_API float ggml_hifi_compute_block_importance(const float * imatrix_block, int block_size);
GGML_API int   ggml_hifi_compute_block_outlier_count(float block_importance,
                                                      int base_outlier_count, float model_params_b);

// Q3_K_HIFI model-size classification
typedef enum {
    Q3_HIFI_SIZE_TINY   = 0,  // ≤1.7B
    Q3_HIFI_SIZE_MEDIUM = 1,  // 2B–8B (sweet spot)
    Q3_HIFI_SIZE_LARGE  = 2,  // 14B+
} ggml_q3_hifi_size_category;

GGML_API ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b);
GGML_API int   ggml_q3_hifi_get_max_outliers(float model_params_b);
GGML_API float ggml_q3_hifi_get_outlier_threshold(float model_params_b);
GGML_API float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n);
GGML_API int   ggml_q3_hifi_should_enhance_tensor(const char * tensor_name,
                                                    const float * weights, int64_t n_elements,
                                                    float model_params_b,
                                                    int * enhanced_count, int max_enhanced);
GGML_API int   ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding);
GGML_API float ggml_q3_hifi_get_attn_v_threshold(float model_params_b);

// TLS per-tensor outlier control (reused by Q4_K_HIFI)
GGML_API void  ggml_q3_hifi_set_tensor_outliers(int outliers);
GGML_API int   ggml_q3_hifi_get_tensor_outliers(void);
GGML_API void  ggml_q3_hifi_set_tensor_importance(float importance);
GGML_API float ggml_q3_hifi_get_tensor_importance(void);
GGML_API void  ggml_q3_hifi_reset_tensor_state(void);
GGML_API int   ggml_q3_hifi_compute_block_outliers(float block_outlier_ratio,
                                                    int base_outlier_count, float model_params_b);

// Q4_K_HIFI
GGML_API int ggml_q4_hifi_get_max_outliers(float model_params_b);

// K_LITE tier-based residual budget
GGML_API int ggml_lite_get_residual_budget(float tensor_importance, float model_params_b,
                                            int max_residuals);
```

The implementation context is stored as a `thread_local` pointer in `ggml-quants-hifi.c`.

> **Porting note — required include guard in `ggml-quants-hifi.c`.**
> The constants `Q3_K_HIFI_MAX_OUTLIERS`, `Q6_K_HIFI_RES8_MAX_OUTLIERS`, etc. are defined
> inside the `#if defined(GGML_COMMON_DECL)` block in `ggml-common.h`. That guard is only
> activated when one of the `GGML_COMMON_DECL_*` variants is defined before the include.
> Add this at the top of `ggml-quants-hifi.c` (before any other includes):
>
> ```c
> #define GGML_COMMON_DECL_C
> #include "ggml-common.h"
> ```

---

## 5. Core Quantization Functions (`ggml/src/ggml-quants.c` and `ggml-quants-hifi.c`)

### 5.1 Function signatures to declare in `ggml/src/ggml-quants.h`

```c
// HIFI FP16-outlier types
void quantize_row_q3_k_hifi_ref(const float * x, block_q3_k_hifi * y, int64_t k);
void dequantize_row_q3_k_hifi(const block_q3_k_hifi * x, float * y, int64_t k);
size_t quantize_q3_k_hifi(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);

void quantize_row_q4_k_hifi_ref(const float * x, block_q4_k_hifi * y, int64_t k);
void dequantize_row_q4_k_hifi(const block_q4_k_hifi * x, float * y, int64_t k);
size_t quantize_q4_k_hifi(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);

void quantize_row_q6_k_hifi_ref(const float * x, block_q6_k_hifi * y, int64_t k);
void dequantize_row_q6_k_hifi(const block_q6_k_hifi * x, float * y, int64_t k);
size_t quantize_q6_k_hifi(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);

void quantize_row_q6_k_hifi_dynamic_ref(const float * x, block_q6_k_hifi_dynamic * y, int64_t k);
void dequantize_row_q6_k_hifi_dynamic(const block_q6_k_hifi_dynamic * x, float * y, int64_t k);
size_t quantize_q6_k_hifi_dynamic(const float * src, void * dst, int64_t nrows,
                                   int64_t n_per_row, const float * imatrix);

void quantize_row_q2_k_hifi_ref(const float * x, block_q2_k_hifi * y, int64_t k);
void dequantize_row_q2_k_hifi(const block_q2_k_hifi * x, float * y, int64_t k);
size_t quantize_q2_k_hifi(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);

// HIFI INT8-residual types
void quantize_row_q6_k_hifi_res8_ref(const float * x, block_q6_k_hifi_res8 * y, int64_t k);
void dequantize_row_q6_k_hifi_res8(const block_q6_k_hifi_res8 * x, float * y, int64_t k);
size_t quantize_q6_k_hifi_res8(const float * src, void * dst, int64_t nrows,
                                int64_t n_per_row, const float * imatrix);

void quantize_row_q5_k_hifi_res8_ref(const float * x, block_q5_k_hifi_res8 * y, int64_t k);
void dequantize_row_q5_k_hifi_res8(const block_q5_k_hifi_res8 * x, float * y, int64_t k);
size_t quantize_q5_k_hifi_res8(const float * src, void * dst, int64_t nrows,
                                int64_t n_per_row, const float * imatrix);

void quantize_row_q3_k_hifi_res8_ref(const float * x, block_q3_k_hifi_res8 * y, int64_t k);
void dequantize_row_q3_k_hifi_res8(const block_q3_k_hifi_res8 * x, float * y, int64_t k);
size_t quantize_q3_k_hifi_res8(const float * src, void * dst, int64_t nrows,
                                int64_t n_per_row, const float * imatrix);

// LITE types
void quantize_row_q2_k_lite_ref(const float * x, block_q2_k_lite * y, int64_t k);
void dequantize_row_q2_k_lite(const block_q2_k_lite * x, float * y, int64_t k);
size_t quantize_q2_k_lite(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);
// ... same pattern for q3_k_lite through q6_k_lite
```

### 5.2 Quantization Algorithm Design

#### FP16 Outlier Replacement (e.g. Q4_K_HIFI)

```
1. Identify top-N outliers by |weight| × imatrix_importance (or just |weight|)
   – N = ggml_q3_hifi_get_tensor_outliers() or model-size default
   – Selection uses 3σ rule + magnitude ranking

2. Zero the N outlier positions in a temporary copy

3. Quantize the zeroed copy with the standard base function
   (e.g. quantize_row_q4_k_ref for Q4_K_HIFI)

4. Store:
   – base block bytes (144 bytes for Q4_K)
   – outlier_idx[N]: positions sorted ascending
   – outlier FP16 values: ggml_half(x[outlier_idx[i]])
   – outlier_count = N  (use the `outlier_count` field; Q3_K_HIFI has this explicitly;
     other FP16 types that have a fixed count may omit the field)
```

#### INT8 Residual Correction (e.g. Q6_K_HIFI_RES8)

```
1. Quantize with base function normally (no zeroing)

2. Dequantize the base result immediately

3. Compute residuals: err[i] = x[i] - base_decoded[i]

4. Select top-N positions by |err[i]| × imatrix_importance[i]

5. Compute shared scale: residual_scale = max(|err[selected]|) / 127.0f

6. Quantize residuals: residual_vals[i] = round(err[selected[i]] / residual_scale)

7. Store residual_count, outlier_idx, residual_vals, residual_scale
```

#### Dequantization

FP16 replacement:
```c
// Base dequant into tmp[]
base_dequant(block, tmp, QK_K);
// Replace outlier positions
for (int i = 0; i < block->outlier_count; i++) {
    tmp[block->outlier_idx[i]] = ggml_half_to_float(block->outliers[i]);
}
memcpy(y, tmp, QK_K * sizeof(float));
```

INT8 residual:
```c
// Base dequant into y[]
base_dequant(block, y, QK_K);
// Add residuals
float rscale = block->residual_scale;
for (int i = 0; i < block->outlier_count; i++) {
    y[block->outlier_idx[i]] += rscale * ((float)block->residual_vals[i] / 127.0f);
}
```

#### LITE / RES8 base-block pointer idiom

LITE structs and `block_q5_k_hifi_res8` embed their base-block data at **offset 0**. The
correct portable way to pass the base block to an existing quantizer or dequantizer is to
cast the address of the LITE block directly:

```c
// Q5_K_LITE: Q4_K data starts at offset 0
quantize_row_q4_K_ref(src, (block_q4_K *)&y[i], QK_K);
dequantize_row_q4_K((const block_q4_K *)&x[i], out, QK_K);

// Q6_K_LITE / Q5_K_HIFI_RES8: Q5_K data starts at offset 0
quantize_row_q5_K_ref(src, (block_q5_K *)&y[i], QK_K);
dequantize_row_q5_K((const block_q5_K *)&x[i], out, QK_K);
```

> **Do NOT use the `GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d` pattern here.**
> In C mode (`GGML_COMMON_DECL_C`), both macros expand to the empty string, producing
> `y[i]...d` — a syntax error. The `&y[i]` cast is the correct and portable form.

---

## 6. Type Registration (`ggml/src/ggml.c`)

In the `ggml_type_traits` table, add entries for each new type:

```c
[GGML_TYPE_Q3_K_HIFI] = {
    .type_name      = "Q3_K_HIFI",
    .blck_size      = Q3_K_HIFI_BLOCK_SIZE,
    .type_size      = sizeof(block_q3_k_hifi),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q3_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q3_k_hifi_ref,
},
[GGML_TYPE_Q6_K_HIFI] = {
    .type_name      = "Q6_K_HIFI",
    .blck_size      = QK_K,
    .type_size      = sizeof(block_q6_k_hifi),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q6_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_hifi_ref,
},
[GGML_TYPE_Q6_K_HIFI_DYNAMIC] = {
    .type_name      = "Q6_K_HIFI_DYN",
    .blck_size      = QK_K,
    .type_size      = sizeof(block_q6_k_hifi_dynamic),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q6_k_hifi_dynamic,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_hifi_dynamic_ref,
},
[GGML_TYPE_Q6_K_HIFI_RES8] = {
    .type_name      = "Q6_K_HIFI_RES8",
    .blck_size      = QK_K,
    .type_size      = sizeof(block_q6_k_hifi_res8),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q6_k_hifi_res8,
    .from_float_ref = (ggml_from_float_t) quantize_row_q6_k_hifi_res8_ref,
},
[GGML_TYPE_Q5_K_HIFI_RES8] = {
    .type_name      = "Q5_K_HIFI_RES8",
    .blck_size      = QK_K,
    .type_size      = sizeof(block_q5_k_hifi_res8),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q5_k_hifi_res8,
    .from_float_ref = (ggml_from_float_t) quantize_row_q5_k_hifi_res8_ref,
},
[GGML_TYPE_Q3_K_HIFI_RES8] = {
    .type_name      = "Q3_K_HIFI_RES8",
    .blck_size      = Q3_K_HIFI_BLOCK_SIZE,
    .type_size      = sizeof(block_q3_k_hifi_res8),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q3_k_hifi_res8,
    .from_float_ref = (ggml_from_float_t) quantize_row_q3_k_hifi_res8_ref,
},
[GGML_TYPE_Q4_K_HIFI] = {
    .type_name      = "Q4_K_HIFI",
    .blck_size      = Q4_K_HIFI_BLOCK_SIZE,
    .type_size      = sizeof(block_q4_k_hifi),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q4_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q4_k_hifi_ref,
},
[GGML_TYPE_Q2_K_HIFI] = {
    .type_name      = "Q2_K_HIFI",
    .blck_size      = Q2_K_HIFI_BLOCK_SIZE,
    .type_size      = sizeof(block_q2_k_hifi),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_q2_k_hifi,
    .from_float_ref = (ggml_from_float_t) quantize_row_q2_k_hifi_ref,
},
// LITE types — same pattern
[GGML_TYPE_Q2_K_LITE] = { .type_name = "Q2_K_LITE", .blck_size = Q2_K_LITE_BLOCK_SIZE, ... },
[GGML_TYPE_Q3_K_LITE] = { .type_name = "Q3_K_LITE", .blck_size = Q3_K_LITE_BLOCK_SIZE, ... },
[GGML_TYPE_Q4_K_LITE] = { .type_name = "Q4_K_LITE", .blck_size = Q4_K_LITE_BLOCK_SIZE, ... },
[GGML_TYPE_Q5_K_LITE] = { .type_name = "Q5_K_LITE", .blck_size = Q5_K_LITE_BLOCK_SIZE, ... },
[GGML_TYPE_Q6_K_LITE] = { .type_name = "Q6_K_LITE", .blck_size = Q6_K_LITE_BLOCK_SIZE, ... },
```

Also add the `ggml_quantize_chunk` switch-case entries in `ggml.c`:

```c
case GGML_TYPE_Q3_K_HIFI:        result = quantize_q3_k_hifi(...);        break;
case GGML_TYPE_Q6_K_HIFI:        result = quantize_q6_k_hifi(...);        break;
case GGML_TYPE_Q6_K_HIFI_DYNAMIC:result = quantize_q6_k_hifi_dynamic(...);break;
case GGML_TYPE_Q6_K_HIFI_RES8:   result = quantize_q6_k_hifi_res8(...);   break;
case GGML_TYPE_Q5_K_HIFI_RES8:   result = quantize_q5_k_hifi_res8(...);   break;
case GGML_TYPE_Q3_K_HIFI_RES8:   result = quantize_q3_k_hifi_res8(...);   break;
case GGML_TYPE_Q4_K_HIFI:        result = quantize_q4_k_hifi(...);        break;
case GGML_TYPE_Q2_K_HIFI:        result = quantize_q2_k_hifi(...);        break;
case GGML_TYPE_Q2_K_LITE:        result = quantize_q2_k_lite(...);        break;
case GGML_TYPE_Q3_K_LITE:        result = quantize_q3_k_lite(...);        break;
case GGML_TYPE_Q4_K_LITE:        result = quantize_q4_k_lite(...);        break;
case GGML_TYPE_Q5_K_LITE:        result = quantize_q5_k_lite(...);        break;
case GGML_TYPE_Q6_K_LITE:        result = quantize_q6_k_lite(...);        break;
```

---

## 7. GGUF Type Name Mapping (`ggml/src/gguf.cpp`)

> **No changes required to `gguf.cpp` in current upstream versions.**
>
> Type name resolution is handled entirely by `ggml_type_name()`, which reads the
> `.type_name` field from the `ggml_type_traits` table registered in `ggml.c` (Section 6).
> There is no separate string↔type map in `gguf.cpp`; the table entries added in
> Section 6 are sufficient for models to be recognised on load.
>
> If a future upstream refactor reintroduces a standalone name map in `gguf.cpp`, add
> entries following this pattern:
>
> ```cpp
> { "Q3_K_HIFI",        GGML_TYPE_Q3_K_HIFI        },
> { "Q6_K_HIFI",        GGML_TYPE_Q6_K_HIFI        },
> { "Q6_K_HIFI_DYN",    GGML_TYPE_Q6_K_HIFI_DYNAMIC },
> { "Q6_K_HIFI_RES8",   GGML_TYPE_Q6_K_HIFI_RES8   },
> { "Q5_K_HIFI_RES8",   GGML_TYPE_Q5_K_HIFI_RES8   },
> { "Q3_K_HIFI_RES8",   GGML_TYPE_Q3_K_HIFI_RES8   },
> { "Q4_K_HIFI",        GGML_TYPE_Q4_K_HIFI        },
> { "Q2_K_HIFI",        GGML_TYPE_Q2_K_HIFI        },
> { "Q2_K_LITE",        GGML_TYPE_Q2_K_LITE        },
> { "Q3_K_LITE",        GGML_TYPE_Q3_K_LITE        },
> { "Q4_K_LITE",        GGML_TYPE_Q4_K_LITE        },
> { "Q5_K_LITE",        GGML_TYPE_Q5_K_LITE        },
> { "Q6_K_LITE",        GGML_TYPE_Q6_K_LITE        },
> ```

---

## 8. Quantization Selection Logic (`src/llama-quant.cpp`)

This is the most complex file. It controls **which tensor gets which type** during quantization.

### 8.1 Includes and Global State

```cpp
// At top of llama-quant.cpp
// Required: activates GGML_COMMON_DECL guard so that constants such as
// Q3_K_HIFI_MAX_OUTLIERS, Q5_K_HIFI_RES8_MAX_OUTLIERS, Q6_K_HIFI_RES8_MAX_OUTLIERS
// are visible.  Must appear before any other ggml-common.h include in this TU.
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

extern "C" {
#include "../ggml/src/ggml-quants-hifi.h"
}

// Imatrix guidance for Q3_K_HIFI
struct tensor_importance_entry {
    std::string name;
    float importance;
    bool is_candidate;  // input projection (not o_proj / down_proj)
};
static std::map<std::string, float> g_tensor_importance_map;
static float g_importance_threshold = 0.0f;
static bool  g_imatrix_guided_enabled = false;
```

### 8.2 Tensor Category Enum

```cpp
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};
```

### 8.3 Model Size Helper

```cpp
static float compute_model_params_b(const llama_hparams & hparams, int64_t n_vocab) {
    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_ff    = hparams.n_ff();
    const int64_t n_layer = hparams.n_layer;

    const int64_t attn_params = 4 * n_embd * n_embd * n_layer;
    const int64_t ffn_params  = 3 * n_embd * n_ff   * n_layer;
    const int64_t emb_params  = 2 * n_vocab * n_embd;

    return (float)(attn_params + ffn_params + emb_params) / 1e9f;
}
```

> **Porting note — `llama_vocab` vocab size accessor.**
> `llama_vocab` does **not** have an `n_vocab()` method. The correct call is `n_tokens()`:
> ```cpp
> const int64_t n_vcb = qs.model.vocab.n_tokens();
> ```
> Using `n_vocab()` produces a compile error: *no member named 'n_vocab' in 'llama_vocab'*.
> Pass the result as the `n_vocab` parameter to `compute_model_params_b` above.

### 8.4 HIFI Enhanced Type Selection

```cpp
// For Q4_K_HIFI enhancement of critical tensors
static ggml_type get_hifi_enhanced_type(float model_params_b) {
    return (model_params_b <= 5.0f) ? GGML_TYPE_Q5_K_HIFI_RES8
                                    : GGML_TYPE_Q6_K_HIFI_RES8;
}

// For Q5_K_HIFI enhancement of critical tensors
static ggml_type get_q5_hifi_enhanced_type(float model_params_b) {
    if (model_params_b <= 2.0f) return GGML_TYPE_Q6_K;          // no HIFI overhead
    if (model_params_b <= 5.0f) return GGML_TYPE_Q5_K_HIFI_RES8;
    return GGML_TYPE_Q6_K_HIFI_RES8;
}
```

### 8.5 Enhancement Thresholds (fraction of attn_v layers to upgrade)

```cpp
// Q4_K_HIFI: percentage of early attn_v layers to enhance
static float get_hifi_enhancement_threshold(float model_params_b) {
    if (model_params_b <= 1.0f)  return 0.32f;
    if (model_params_b <= 2.0f)  return 0.25f;
    if (model_params_b <= 5.0f)  return 0.20f;
    if (model_params_b <= 15.0f) return 0.20f;
    return 0.0f;   // no attn_v enhancement for very large models
}

// Q3_K_HIFI: more conservative thresholds
static float get_q3_hifi_attn_v_threshold(float model_params_b) {
    if (model_params_b <= 1.0f)  return 0.0f;  // skip for tiny
    if (model_params_b <= 1.7f)  return 0.0f;  // 1.7B: disabled
    if (model_params_b <= 5.0f)  return 0.25f;
    if (model_params_b <= 10.0f) return 0.15f;
    if (model_params_b <= 20.0f) return 0.08f;
    return 0.05f;
}

// Q5_K_HIFI attn_v threshold (focused on proven wins)
static float get_q5_hifi_attn_v_threshold(float model_params_b) {
    // See current source for exact values
}
```

### 8.6 Type Selection in `llama_tensor_get_type_impl`

The main function contains large `if/else` blocks switching on `ftype`. Here is the
pattern for each HIFI ftype:

**`LLAMA_FTYPE_MOSTLY_Q4_K_HIFI`:**

```cpp
if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI) {
    if (category == tensor_category::OUTPUT ||
        (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        new_type = get_hifi_enhanced_type(model_params_b);

    } else if (category == tensor_category::TOKEN_EMBD) {
        new_type = get_hifi_enhanced_type(model_params_b);

    } else if (category == tensor_category::ATTENTION_V || ...) {
        float threshold = get_hifi_enhancement_threshold(model_params_b);
        if (qs.i_attention_wv <= qs.n_attention_wv * threshold) {
            new_type = get_hifi_enhanced_type(model_params_b);
        } else if (use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) {
            new_type = GGML_TYPE_Q6_K;
        } else {
            new_type = GGML_TYPE_Q4_K_HIFI;
        }
    }
    // ... FFN gate handling for small models
}
```

**`LLAMA_FTYPE_MOSTLY_Q5_K_HIFI`:** — same structure, but threshold comes from
`get_q5_hifi_attn_v_threshold()` and the base type is `GGML_TYPE_Q5_K`.

**`LLAMA_FTYPE_MOSTLY_Q3_K_HIFI`:**

```cpp
if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) {
    if (category == tensor_category::OUTPUT || category == tensor_category::TOKEN_EMBD) {
        // Large models (>1.7B): use Q6_K; tiny: match Q3_K_M
        new_type = (model_params_b > 1.7f) ? GGML_TYPE_Q6_K
                                            : GGML_TYPE_Q3_K;
    } else if (category == tensor_category::ATTENTION_V || ...) {
        float threshold = get_q3_hifi_attn_v_threshold(model_params_b);
        ggml_type hifi_type = get_q3_hifi_attn_v_type(model_params_b);
        if (qs.i_attention_wv <= qs.n_attention_wv * threshold) {
            new_type = hifi_type;
        } else if (i_attention_wv <= 2) {
            new_type = GGML_TYPE_Q5_K;
        } else {
            new_type = GGML_TYPE_Q4_K;
        }
    }
    // Imatrix-guided: if g_imatrix_guided_enabled and tensor is a candidate,
    // check g_tensor_importance_map[name] > g_importance_threshold → use Q3_K_HIFI
}
```

**`LLAMA_FTYPE_MOSTLY_Q2_K_HIFI`:**

```cpp
if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_HIFI) {
    if (category == tensor_category::OUTPUT) {
        new_type = GGML_TYPE_Q6_K;   // always critical
    }
    // Other tensors remain Q2_K_HIFI (the default type returned by llama_ftype_to_ggml_type)
}
```

### 8.7 Default Type Mapping (`llama_ftype_to_ggml_type`)

```cpp
case LLAMA_FTYPE_MOSTLY_Q4_K_HIFI: return GGML_TYPE_Q4_K;
case LLAMA_FTYPE_MOSTLY_Q5_K_HIFI: return GGML_TYPE_Q5_K;
case LLAMA_FTYPE_MOSTLY_Q3_K_HIFI: return GGML_TYPE_Q3_K;
case LLAMA_FTYPE_MOSTLY_Q2_K_HIFI: return GGML_TYPE_Q2_K_HIFI;
case LLAMA_FTYPE_MOSTLY_Q2_K_LITE: return GGML_TYPE_Q2_K_LITE;
case LLAMA_FTYPE_MOSTLY_Q3_K_LITE: return GGML_TYPE_Q3_K_LITE;
case LLAMA_FTYPE_MOSTLY_Q4_K_LITE: return GGML_TYPE_Q4_K_LITE;
case LLAMA_FTYPE_MOSTLY_Q5_K_LITE: return GGML_TYPE_Q5_K_LITE;
case LLAMA_FTYPE_MOSTLY_Q6_K_LITE: return GGML_TYPE_Q6_K_LITE;
```

### 8.8 HIFI Context Setup During Quantization

In `llama_model_quantize_impl` (the per-tensor quantization loop):

```cpp
ggml_hifi_quant_context hifi_ctx = {};
const ggml_hifi_quant_context * hifi_ctx_ptr = nullptr;

// Q3_K_HIFI: model-size-aware + optional imatrix-guided outlier count
if (new_type == GGML_TYPE_Q3_K_HIFI && ftype == LLAMA_FTYPE_MOSTLY_Q3_K_HIFI) {
    int base_outliers = ggml_q3_hifi_get_max_outliers(model_params_b);
    float tensor_importance = 0.0f;
    if (imatrix && g_imatrix_guided_enabled) {
        tensor_importance = ggml_hifi_compute_tensor_importance(imatrix, n_per_row);
        if (tensor_importance > g_importance_threshold) {
            base_outliers = std::min(base_outliers + 2, Q3_K_HIFI_MAX_OUTLIERS);
        }
    }
    ggml_q3_hifi_set_tensor_outliers(base_outliers);
    ggml_q3_hifi_set_tensor_importance(tensor_importance);
    hifi_ctx = ggml_hifi_quant_context{ base_outliers, tensor_importance, -1, (int)n_layer, 1, model_params_b };
    hifi_ctx_ptr = &hifi_ctx;
}

// Q6_K_HIFI_RES8 / Q5_K_HIFI_RES8: layer-adaptive outlier count
if ((new_type == GGML_TYPE_Q6_K_HIFI_RES8 || new_type == GGML_TYPE_Q5_K_HIFI_RES8) &&
    (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_HIFI || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_HIFI)) {
    // Parse layer index from tensor name
    int layer_idx = parse_layer_idx(tensor->name);
    float layer_importance = 0.0f;
    if (imatrix)
        layer_importance = ggml_hifi_compute_tensor_importance(imatrix, n_per_row);
    int max_outliers = (new_type == GGML_TYPE_Q5_K_HIFI_RES8)
                       ? Q5_K_HIFI_RES8_MAX_OUTLIERS : Q6_K_HIFI_RES8_MAX_OUTLIERS;
    int outlier_count = ggml_hifi_compute_outlier_count(layer_idx, n_layers,
                                                         layer_importance, model_params_b);
    outlier_count = std::min(outlier_count, max_outliers);
    hifi_ctx = ggml_hifi_quant_context{ outlier_count, layer_importance, layer_idx, n_layers, 1, model_params_b };
    hifi_ctx_ptr = &hifi_ctx;
}

// Q4_K_HIFI: per-tensor outlier count via TLS
if (new_type == GGML_TYPE_Q4_K_HIFI) {
    int q4_outliers = ggml_q4_hifi_get_max_outliers(model_params_b);
    if (imatrix) {
        float importance = ggml_hifi_compute_tensor_importance(imatrix, n_per_row);
        if (importance > high_importance_threshold)
            q4_outliers = Q4_K_HIFI_MAX_OUTLIERS;
    }
    ggml_q3_hifi_set_tensor_outliers(q4_outliers);  // reuses Q3 TLS infrastructure
}

// Pass hifi_ctx_ptr to quantize function
llama_tensor_quantize_impl(new_type, f32_data, new_data, chunk_size,
                           nrows, n_per_row, imatrix, workers, nthread, hifi_ctx_ptr);
```

The `llama_tensor_quantize_impl` function sets/clears the thread-local context around
each quantization call:

```cpp
static size_t llama_tensor_quantize_impl(..., const ggml_hifi_quant_context * hifi_ctx) {
    // single-threaded path
    if (hifi_ctx) ggml_hifi_set_context(hifi_ctx);
    result = ggml_quantize_chunk(...);
    if (hifi_ctx) ggml_hifi_set_context(nullptr);
    return result;
}
```

---

## 9. Model Loader (`src/llama-model-loader.cpp`)

### 9.1 `llama_model_ftype_name` additions

```cpp
case LLAMA_FTYPE_MOSTLY_Q4_K_HIFI: return "Q4_K_HIFI - ~4.95 bpw";
case LLAMA_FTYPE_MOSTLY_Q3_K_HIFI: return "Q3_K_HIFI - ~3.7 bpw";
case LLAMA_FTYPE_MOSTLY_Q5_K_HIFI: return "Q5_K_HIFI - ~5.4 bpw";
case LLAMA_FTYPE_MOSTLY_Q2_K_HIFI: return "Q2_K_HIFI - ~3.0 bpw";
case LLAMA_FTYPE_MOSTLY_Q2_K_LITE: return "Q2_K_LITE - 3.0 bpw";
case LLAMA_FTYPE_MOSTLY_Q3_K_LITE: return "Q3_K_LITE - 3.25 bpw";
case LLAMA_FTYPE_MOSTLY_Q4_K_LITE: return "Q4_K_LITE - 4.0 bpw";
case LLAMA_FTYPE_MOSTLY_Q5_K_LITE: return "Q5_K_LITE - 5.13 bpw";
case LLAMA_FTYPE_MOSTLY_Q6_K_LITE: return "Q6_K_LITE - 6.13 bpw";
```

### 9.2 `type_max → ftype` mapping

```cpp
case GGML_TYPE_Q3_K_HIFI:          ftype = LLAMA_FTYPE_MOSTLY_Q3_K_HIFI; break;
case GGML_TYPE_Q6_K_HIFI_DYNAMIC:  ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
case GGML_TYPE_Q6_K_HIFI_RES8:     ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
case GGML_TYPE_Q5_K_HIFI_RES8:     ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
case GGML_TYPE_Q4_K_HIFI:          ftype = LLAMA_FTYPE_MOSTLY_Q4_K_HIFI; break;
case GGML_TYPE_Q2_K_HIFI:          ftype = LLAMA_FTYPE_MOSTLY_Q2_K_HIFI; break;
case GGML_TYPE_Q2_K_LITE:          ftype = LLAMA_FTYPE_MOSTLY_Q2_K_LITE; break;
case GGML_TYPE_Q3_K_LITE:          ftype = LLAMA_FTYPE_MOSTLY_Q3_K_LITE; break;
case GGML_TYPE_Q4_K_LITE:          ftype = LLAMA_FTYPE_MOSTLY_Q4_K_LITE; break;
case GGML_TYPE_Q5_K_LITE:          ftype = LLAMA_FTYPE_MOSTLY_Q5_K_LITE; break;
case GGML_TYPE_Q6_K_LITE:          ftype = LLAMA_FTYPE_MOSTLY_Q6_K_LITE; break;
```

---

## 10. Quantize Tool (`tools/quantize/quantize.cpp`)

Add HIFI/LITE type names to the `quantize_types` table:

```cpp
{ "Q2_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q2_K_HIFI,  "~3.0 bpw Q2_K + INT8 residuals" },
{ "Q3_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q3_K_HIFI,  "~3.7G Q3_K_M + scale-aware FP16 outliers" },
{ "Q4_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q4_K_HIFI,  "~4.95 bpw Q4_K + FP16 outliers, tiered" },
{ "Q5_K_HIFI",  LLAMA_FTYPE_MOSTLY_Q5_K_HIFI,  "~5.4 bpw Q5_K_M + Q6_K_HIFI_RES8 critical" },
{ "Q2_K_LITE",  LLAMA_FTYPE_MOSTLY_Q2_K_LITE,  "3.0 bpw Q2_K + INT8 residuals" },
{ "Q3_K_LITE",  LLAMA_FTYPE_MOSTLY_Q3_K_LITE,  "3.25 bpw Q2_K + INT8 residuals" },
{ "Q4_K_LITE",  LLAMA_FTYPE_MOSTLY_Q4_K_LITE,  "4.0 bpw Q3_K + INT8 residuals" },
{ "Q5_K_LITE",  LLAMA_FTYPE_MOSTLY_Q5_K_LITE,  "5.13 bpw Q4_K + INT8 residuals" },
{ "Q6_K_LITE",  LLAMA_FTYPE_MOSTLY_Q6_K_LITE,  "6.13 bpw Q5_K + INT8 residuals" },
```

Also add HIFI/LITE cases to the ARM `repack` guard that down-selects certain types
(see lines ~1413–1421 in the current source; HIFI/LITE map to their fallback standard types).

---

## 11. Backend Additions

All backends follow the same pattern: add dequantization kernels for each HIFI/LITE type,
and optionally add matrix-multiply (MMQ/MMVQ) kernels for GPU acceleration.

### 11.1 CPU Backend (`ggml/src/ggml-cpu/`)

**Required files** (reference / CPU-only inference):

- **`quants.h`**: Declare `quantize_row_q*_hifi/lite` wrappers (with `void *` output) and
  `ggml_vec_dot_q*_hifi_q8_K` / `ggml_vec_dot_q*_lite_q8_K` functions for all 13 types.
- **`quants.c`**: Implement the wrappers (call the `*_ref` functions) and the vec_dot
  functions. The reference vec_dot pattern is: dequantize one HIFI/LITE block to float,
  dequantize the corresponding Q8_K block to float, accumulate the float dot product.
  All HIFI/LITE types have `blck_size = 256`, so a fixed `float[256]` stack buffer works.
  Use a function-generating macro to avoid repetition:
  ```c
  #define HIFI_VEC_DOT_Q8K(name, block_type, deq_fn) \
  void name(int n, float * s, ...) { \
      float tmp_x[QK_K], tmp_y[QK_K]; \
      for (int i = 0; i < n/QK_K; i++) { \
          deq_fn((const block_type *)vx + i, tmp_x, QK_K); \
          dequantize_row_q8_K((const block_q8_K *)vy + i, tmp_y); \
          for (int j = 0; j < QK_K; j++) sumf += tmp_x[j] * tmp_y[j]; \
      } *s = sumf; }
  ```

  **Q3_K_HIFI optimized vec_dot** — do NOT use the generic dequant macro for this type.
  Instead, call the existing `ggml_vec_dot_q3_K_q8_K` SIMD function on the embedded
  `q3_k_data` field, then add outlier FMA corrections. This reuses all platform SIMD
  paths (AVX2, AVX-512, ARM NEON) automatically:
  ```c
  void ggml_vec_dot_q3_k_hifi_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                                     const void * GGML_RESTRICT vx, size_t bx,
                                     const void * GGML_RESTRICT vy, size_t by, int nrc) {
      assert(n % QK_K == 0);
      const int nb = n / QK_K;
      const block_q3_k_hifi * bx_hifi = (const block_q3_k_hifi *)vx;
      const block_q8_K       * by_q8  = (const block_q8_K *)vy;
      float result = 0.0f;
      for (int i = 0; i < nb; i++) {
          float block_dot = 0.0f;
          ggml_vec_dot_q3_K_q8_K(QK_K, &block_dot, sizeof(float),
                                  bx_hifi[i].q3_k_data, sizeof(block_q3_K),
                                  &by_q8[i], sizeof(block_q8_K), 1);
          result += block_dot;
          int nc = (int)bx_hifi[i].outlier_count;
          if ((unsigned)nc > (unsigned)Q3_K_HIFI_MAX_OUTLIERS) nc = Q3_K_HIFI_MAX_OUTLIERS;
          const float q8d = by_q8[i].d;
          for (int j = 0; j < nc; j++) {
              const int pos = (int)bx_hifi[i].outlier_idx[j];
              result += GGML_FP16_TO_FP32(bx_hifi[i].outliers[j])
                      * (q8d * (float)by_q8[i].qs[pos]);
          }
      }
      *s = result;
  }
  ```
  The correction is **additive** (not a replacement) in vec_dot because the base Q3_K
  block was quantized with zeros at the outlier positions, so those positions contribute
  exactly 0 to the SIMD sum — verified by the Q3_K zero-level decode identity
  `(level_4 - 4) * d = 0`.

  **Q4_K_HIFI vec_dot** — Q4_K uses an unsigned-level + dmin encoding so zeroed positions
  do **not** decode to exactly 0. The correction must subtract the Q4K decoded value at
  each outlier position and add the true FP16 value. There is no `outlier_count` field in
  `block_q4_k_hifi`; use a FP16-zero sentinel break instead (see Sentinel Pitfall below).
  The reference implementation dequantizes the embedded `q4_k_data` block to a float
  buffer, dots it with Q8_K, then applies per-outlier corrections:
  ```c
  void ggml_vec_dot_q4_k_hifi_q8_K(int n, float * GGML_RESTRICT s, ...) {
      const int nb = n / QK_K;
      const block_q4_k_hifi * x = vx;
      const block_q8_K       * y = vy;
      float total = 0.0f;
      for (int i = 0; i < nb; i++) {
          float w[QK_K];
          dequantize_row_q4_K((const block_q4_K *)x[i].q4_k_data, w, QK_K);
          const float dy = y[i].d;
          const int8_t * q8 = y[i].qs;
          float sum = 0.0f;
          for (int j = 0; j < QK_K; j++) sum += w[j] * (float)q8[j] * dy;
          // Outlier correction: replace Q4K contribution with true FP16 value.
          // Break on FP16-zero sentinel (unused slots store outliers=0x0000).
          for (int k = 0; k < Q4_K_HIFI_OUTLIERS; k++) {
              ggml_half oval = x[i].outliers[k];
              if (oval == 0) break;
              int pos = (int)x[i].outlier_idx[k];
              sum += (GGML_FP16_TO_FP32(oval) - w[pos]) * (float)q8[pos] * dy;
          }
          total += sum;
      }
      *s = total;
  }
  ```
  An optimized variant can call `ggml_vec_dot_q4_K_q8_K` as the SIMD base and then
  correct each outlier position by decoding that position's Q4K value (subblock scale +
  dmin unpacking); however this is more code and the dequant-then-dot reference above is
  correct and sufficient for most use cases.

> ### ⚠️ Sentinel Pitfall — `uint8_t` index guard is always true
>
> **Do NOT guard outlier loops with `if (idx < block_size)` when `idx` is `uint8_t`
> and `block_size == 256`.**  `uint8_t` holds values 0–255 and `256` is never reached,
> so the condition is always true.  Unused outlier slots filled with sentinel
> `(idx=255, outlier_val=0)` will silently pass the guard and corrupt the result:
> - In dequantize: position 255 gets overwritten with 0.0f.
> - In vec_dot (FP16-replacement types): position 255's base contribution is subtracted,
>   corrupting the dot product every block.
>
> **Correct approaches by type:**
>
> | Type | Has `outlier_count`? | Correct loop bound |
> |------|----------------------|--------------------|
> | Q3_K_HIFI | ✓ | `int nc = MIN(block->outlier_count, MAX_OUTLIERS); for (k < nc)` |
> | Q4_K_HIFI | ✗ | `for (k < OUTLIERS) { if (outliers[k]==0) break; ... }` |
> | Q2_K_HIFI | ✓ (masked) | `int nc = block->outlier_count & 0x7F; nc = MIN(nc, MAX)` |
>
> The RES8 types (Q5_K_HIFI_RES8, Q6_K_HIFI_RES8) use `outlier_count` correctly and are
> not affected by this pitfall.

- **`ggml-cpu.c`**: Add 13 entries to `type_traits_cpu[]`, each with
  `.vec_dot_type = GGML_TYPE_Q8_K`, `.nrows = 1`, and the wrappers above.

**Also required** — two switches in `ops.cpp` need HIFI/LITE cases:

**1. `ggml_compute_forward_clamp`** (around line 5553) enumerates every `ggml_type` with
no `default:`, emitting `-Wswitch` for unhandled values. Add all 13 HIFI/LITE cases to
the `GGML_ABORT` arm (same arm as `GGML_TYPE_Q8_K`, `GGML_TYPE_I8`, etc.):

```cpp
case GGML_TYPE_Q3_K_HIFI:
case GGML_TYPE_Q6_K_HIFI:
// ... all 13 types ...
case GGML_TYPE_Q6_K_LITE:
    GGML_ABORT("fatal error");
```

**2. `ggml_compute_forward_get_rows`** — the token embedding lookup op. Without HIFI/LITE
cases, this aborts at inference time whenever a HIFI type is used for `token_embd.weight`
(Q4_K_HIFI sets the embedding to `Q5_K_HIFI_RES8` via `get_hifi_enhanced_type`).
The fix: add all 13 HIFI/LITE types to the `ggml_compute_forward_get_rows_q` arm
(just before the `GGML_TYPE_F16` case). This works because each HIFI type registers
its `to_float` dequant function in `ggml.c`'s type traits table, which is exactly what
`ggml_compute_forward_get_rows_q` calls internally:

```cpp
case GGML_TYPE_Q3_K_HIFI:
case GGML_TYPE_Q6_K_HIFI:
case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
case GGML_TYPE_Q6_K_HIFI_RES8:
case GGML_TYPE_Q5_K_HIFI_RES8:
case GGML_TYPE_Q3_K_HIFI_RES8:
case GGML_TYPE_Q4_K_HIFI:
case GGML_TYPE_Q2_K_HIFI:
case GGML_TYPE_Q2_K_LITE:
case GGML_TYPE_Q3_K_LITE:
case GGML_TYPE_Q4_K_LITE:
case GGML_TYPE_Q5_K_LITE:
case GGML_TYPE_Q6_K_LITE:
    {
        ggml_compute_forward_get_rows_q(params, dst);
    } break;
```

**Not required** for a reference CPU implementation:

- **`repack.cpp`**: No repack handlers are required unless you add runtime repacking.
- **`arch/arm/quants.c`** and **`arch/x86/quants.c`**: SIMD-optimized paths are required
  for **Q3_K_HIFI** (see pattern above — delegate to `ggml_vec_dot_q3_K_q8_K` as base).
  **Q4_K_HIFI** can delegate to the fixed generic (which uses the FP16-zero sentinel break).
  Do **not** delegate Q3_K_HIFI to the generic — the generic's dequant loop is slower and
  the arch-specific call is the correct implementation path for production builds.

### 11.2 CUDA/ROCm Backend (`ggml/src/ggml-cuda/`)

Q3_K_HIFI uses the **dequantize-then-cuBLAS** path on GPU (no native vec_dot kernel).
This is slower per-FLOP than a native MMVQ kernel but correct and GPU-resident.
The implementation covers three areas:

#### 11.2.1 GPU Dequantization Kernel (`convert.cu`)

`block_q3_k_hifi` cannot be declared inside CUDA translation units because
`ggml-common.h` wraps it in `#if !defined(GGML_COMMON_DECL_CUDA)`. Access the
extension fields via raw byte-pointer arithmetic using these fixed offsets:

```c
// Byte layout of block_q3_k_hifi (136 bytes total):
//  [0  ..109] = q3_k_data  (110-byte embedded block_q3_K)
//  [110..117] = outlier_idx[8]       (uint8_t × 8)
//  [118..133] = outliers[8]          (__half × 8 = 16 bytes)
//  [134]      = outlier_count        (uint8_t)
//  [135]      = _pad                 (uint8_t)
static constexpr int Q3_K_HIFI_GPU_STRIDE    = 136;
static constexpr int Q3_K_HIFI_GPU_IDX_OFF   = 110;
static constexpr int Q3_K_HIFI_GPU_VALS_OFF  = 118;
static constexpr int Q3_K_HIFI_GPU_COUNT_OFF = 134;

template<typename dst_t>
static __global__ void dequantize_block_q3_k_hifi(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const uint8_t * blk = (const uint8_t *)vx + i * Q3_K_HIFI_GPU_STRIDE;
    const block_q3_K * x = (const block_q3_K *) blk;   // first 110 bytes cast to base type

    // 64-thread Q3_K dequant (identical to dequantize_block_q3_K):
    const int64_t r   = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0  = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n   = tid / 4;
    const int64_t j   = tid - 4*n;
    uint8_t m  = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift  = 2*j;
    int8_t us = is <  4 ? (x->scales[is-0] & 0xF) | (((x->scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x->scales[is-0] & 0xF) | (((x->scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x->scales[is-8] >>  4) | (((x->scales[is+0] >> 4) & 3) << 4) :
                          (x->scales[is-8] >>  4) | (((x->scales[is-4] >> 6) & 3) << 4);
    const float d_all = x->d;
    const float dl    = d_all * (us - 32);
    dst_t * y  = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q  = x->qs + 32*n;
    const uint8_t * hm = x->hmask;
    for (int l = l0; l < l0+4; ++l) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }

    // Thread 0 applies outlier corrections after all other threads finish
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint8_t * idx_ptr = blk + Q3_K_HIFI_GPU_IDX_OFF;
        const __half * val_ptr  = (const __half *)(blk + Q3_K_HIFI_GPU_VALS_OFF);
        int nc = (int)(blk[Q3_K_HIFI_GPU_COUNT_OFF]);
        if (nc < 0 || nc > Q3_K_HIFI_MAX_OUTLIERS) nc = Q3_K_HIFI_MAX_OUTLIERS;
        dst_t * out = yy + i*QK_K;
        for (int k = 0; k < nc; k++) {
            out[(int)idx_ptr[k]] = (dst_t)__half2float(val_ptr[k]);
        }
    }
}

template<typename dst_t>
static void dequantize_row_q3_k_hifi_cuda(const void * vx, dst_t * y,
                                           const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_k_hifi<<<nb, 64, 0, stream>>>(vx, y);
}
```

Then add `case GGML_TYPE_Q3_K_HIFI: return dequantize_row_q3_k_hifi_cuda;` to **both**
`ggml_get_to_fp16_cuda` and `ggml_get_to_fp32_cuda` dispatch switches in `convert.cu`.

**Q4_K_HIFI CUDA dequant** — uses the same byte-pointer approach. Byte layout of
`block_q4_k_hifi` (168 bytes): `[0..143]` = q4_k_data, `[144..151]` = outlier_idx[8],
`[152..167]` = outliers[8] (FP16 × 8). There is no `outlier_count` byte; use FP16-zero
sentinel: loop over outlier slots and break when `__half2float(val) == 0.0f` (or bit
pattern `val_u16 == 0`). Example kernel outline:
```c
static constexpr int Q4_K_HIFI_GPU_STRIDE   = 168;
static constexpr int Q4_K_HIFI_GPU_IDX_OFF  = 144;
static constexpr int Q4_K_HIFI_GPU_VALS_OFF = 152;

// In thread 0 of each block, after the base Q4_K dequant:
if (threadIdx.x == 0) {
    const uint8_t * idx_ptr = blk + Q4_K_HIFI_GPU_IDX_OFF;
    const __half  * val_ptr = (const __half *)(blk + Q4_K_HIFI_GPU_VALS_OFF);
    dst_t * out = yy + i * QK_K;
    for (int k = 0; k < Q4_K_HIFI_OUTLIERS; k++) {
        __half v = val_ptr[k];
        if (__hisinf(v) == 0 && __half2float(v) == 0.0f) break; // FP16-zero sentinel
        out[(int)idx_ptr[k]] = (dst_t)__half2float(v);
    }
}
```

#### 11.2.2 MMVQ Guard (`mmvq.cuh` / `mmvq.cu`)

Q3_K_HIFI does **not** have a native `vec_dot_q_cuda_t` function registered in
`get_vec_dot_q_cuda`. Without a guard, calling `get_vec_dot_q_cuda(GGML_TYPE_Q3_K_HIFI)`
returns `nullptr` and subsequent kernel launch crashes.

Add to `mmvq.cuh`:
```c
// Returns true only for types that have a native vec_dot GPU kernel.
bool ggml_cuda_has_mmvq_kernel(ggml_type type);
```

Implement in `mmvq.cu`:
```c
bool ggml_cuda_has_mmvq_kernel(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4: case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}
```

In `ggml-cuda.cu`, guard **all three** `use_mul_mat_vec_q` check sites:
```c
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
    && ggml_cuda_has_mmvq_kernel(src0->type)   // ← required for HIFI types
    && !bad_padding_clear
    && src1->type == GGML_TYPE_F32
    && dst->type  == GGML_TYPE_F32
    && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

Similarly in `ggml_cuda_mul_mat_id`:
```c
if (ggml_is_quantized(src0->type) && ggml_cuda_has_mmvq_kernel(src0->type)) {
    // MMVQ path
}
```

#### 11.2.3 MUL_MAT GPU Registration (`ggml-cuda.cu`) — CRITICAL

Add `case GGML_TYPE_Q3_K_HIFI:` to the supported-types list inside
`ggml_cuda_mul_mat` (around the block that lists `GGML_TYPE_Q3_K`, `GGML_TYPE_Q4_K`, etc.).

**Without this entry, Q3_K_HIFI is not recognised as GPU-capable.** The tensor
scheduler then places *all* tensors — including activation tensors and SSM/GDN state
tensors — on CPU, even when `--ngl 0` is used. This causes a ~26× inference slowdown
(measured: 105 s/pass → 4.7 s/pass on a Qwen3.5 9B hybrid model) because the GPU
handles none of the compute, not even the non-weight operations.

The fix is a single `case` label. Once registered, the scheduler correctly assigns
activation tensors to the GPU device even when the weight tensors stay on CPU.

Other backends (Metal, Vulkan, SYCL) follow the same three-part pattern: dequant
kernel, MMVQ/vec_dot guard, MUL_MAT registration.

### 11.3 Metal Backend (`ggml/src/ggml-metal/`)

#### Required immediately (CPU fallback)

`ggml_metal_device_supports_op` in `ggml-metal-device.m` decides whether the Metal backend
handles an op or defers to CPU. The `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID` case currently
accepts every type except `GGML_TYPE_NVFP4`. Without an explicit exclusion, Metal will attempt
to look up a pipeline for HIFI/LITE types, find none, and `GGML_ABORT("not implemented")`.

Add a type guard at the top of the `MUL_MAT` arm:

```objc
case GGML_OP_MUL_MAT:
case GGML_OP_MUL_MAT_ID:
    switch (op->src[0]->type) {
        case GGML_TYPE_Q3_K_HIFI:
        case GGML_TYPE_Q6_K_HIFI:
        case GGML_TYPE_Q6_K_HIFI_DYNAMIC:
        case GGML_TYPE_Q6_K_HIFI_RES8:
        case GGML_TYPE_Q5_K_HIFI_RES8:
        case GGML_TYPE_Q3_K_HIFI_RES8:
        case GGML_TYPE_Q4_K_HIFI:
        case GGML_TYPE_Q2_K_HIFI:
        case GGML_TYPE_Q2_K_LITE:
        case GGML_TYPE_Q3_K_LITE:
        case GGML_TYPE_Q4_K_LITE:
        case GGML_TYPE_Q5_K_LITE:
        case GGML_TYPE_Q6_K_LITE:
            return false; // no Metal kernels — fall back to CPU
        default:
            break;
    }
    return has_simdgroup_reduction && op->src[0]->type != GGML_TYPE_NVFP4;
```

This makes the model load and run correctly on Metal-enabled builds using CPU for HIFI/LITE
weight tensors while still using the GPU for all other operations.

#### Full Metal acceleration (optional, future work)

Files to modify once Metal kernels are written:
- **`ggml-metal.metal`**: Add Metal shader functions following the `kernel_dequantize_q*`
  pattern. HIFI structs must be declared identically to the C side (use
  `GGML_COMMON_DECL_METAL` guard path in `ggml-common.h`).
- **`ggml-metal-device.m`**: Register new kernels, add pipeline objects, and remove the
  CPU-fallback exclusions above once the pipelines exist.
- **`ggml-metal-impl.h`**: Add helper declarations if needed.

### 11.4 Vulkan Backend (`ggml/src/ggml-vulkan/`)

New shader files (each follows the pattern of `dequant_q3_k.comp` / `mul_mat_vec_q3_k.comp`):

```
vulkan-shaders/dequant_q2_k_hifi.comp
vulkan-shaders/dequant_q3_k_hifi.comp
vulkan-shaders/mul_mat_vec_q2_k_hifi.comp
vulkan-shaders/mul_mat_vec_q3_k_hifi.comp
```

Register these in **`vulkan-shaders/vulkan-shaders-gen.cpp`** and in
**`ggml-vulkan.cpp`** where shader pipelines are created.

The Vulkan structs are declared via the `types.glsl` include; add the HIFI struct
layouts there (matching the C layout exactly, including padding). Dequantize functions
go in `dequant_funcs.glsl`.

### 11.5 SYCL Backend (`ggml/src/ggml-sycl/`)

Files to modify:
- **`dequantize.hpp`**: Template specializations for HIFI types.
- **`vecdotq.hpp`**: Dot-product specializations.
- **`mmvq.cpp`**: Dispatch additions.
- **`convert.cpp`**: Conversion kernels.

---

## 12. Imatrix Fix (`tools/imatrix/imatrix.cpp`)

The `log_softmax` function has a fix for FP16 models that produce `+Inf` logits
(e.g. large-vocabulary heads in F16 mode). Without this fix, PPL computation silently
produces NaN.

```cpp
static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) {
            // +Inf - +Inf = NaN; return sentinel to avoid poisoning accumulated NLL
            return {0.0, logits[tok], 1.0f};
        }
        max_logit = std::max(max_logit, logits[i]);
    }
    // ... rest unchanged
}
```

---

## 13. Build System (`ggml/src/CMakeLists.txt`)

Add `ggml-quants-hifi.c` and `ggml-quants-hifi.h` to the GGML source list:

```cmake
target_sources(ggml PRIVATE
    # ... existing files ...
    ggml-quants-hifi.c
    ggml-quants-hifi.h
)
```

For Vulkan, register the new `.comp` shader files in the appropriate `add_shader` calls
in `ggml-vulkan/CMakeLists.txt` or equivalent.

---

## 14. Performance & Quality Reference

### 14.1 Bits-per-Weight

| Type | BPW | Block bytes |
|------|-----|-------------|
| Q2_K_HIFI | 3.0 | 96 |
| Q2_K_LITE | 3.0 | 96 |
| Q3_K_LITE | 3.25 | 104 |
| Q3_K_HIFI_RES8 | 4.13 | 132 |
| Q3_K_HIFI | 4.25 | 136 |
| Q4_K_LITE | 4.0 | 128 |
| Q5_K_LITE | 5.125 | 164 |
| Q4_K_HIFI | 5.25 | 168 |
| Q5_K_HIFI_RES8 | ~5.6–6.1 | 196 |
| Q6_K_LITE | 6.125 | 196 |
| Q6_K_HIFI_RES8 | 7.25 | 232 |
| Q6_K_HIFI | 6.94 | 222 |
| Q6_K_HIFI_DYNAMIC | ~6.9 | 236 |

### 14.2 Measured PPL Results (wikitext-2, 512-token context, Qwen3.5 hybrid)

Tested on a Radeon 8060S (98 GiB VRAM), `-ngl 0` (all weights on CPU, GPU used for
compute). All HIFI files were quantized without imatrix.

| Model | Format | BPW | PPL | Notes |
|-------|--------|-----|-----|-------|
| Qwen3.5-9B | Q3_K_M | 4.84 | 9.3218 | baseline |
| Qwen3.5-9B | Q3_K_HIFI | ~4.25 | **9.0148** | −0.31 PPL; 3.8 s/pass |
| Qwen3.5-0.8B | Q3_K_M | 4.84 | 23.03 | baseline |
| Qwen3.5-0.8B | Q3_K_HIFI | 3.99 | 25.09 | higher BPW gap explains most gap |

**Key takeaway:** For 9B+, Q3_K_HIFI achieves lower PPL than Q3_K_M at lower BPW.
For sub-2B models the PPL gap is largely explained by the BPW difference (HIFI stores
the outlier slots regardless of whether they help, adding ~0.5 BPW overhead that cannot
be recovered for models with flat weight distributions).

### 14.3 Inference Speed

Without the GPU registration fix (§11.2.3), Q3_K_HIFI ran 26× slower than Q3_K_M even
at `-ngl 0`:

| Format | Condition | Speed |
|--------|-----------|-------|
| Q3_K_M | -ngl 0 | 4.77 s/pass |
| Q3_K_HIFI (unregistered) | -ngl 0 | ~105 s/pass |
| Q3_K_HIFI (registered) | -ngl 0 | **4.71 s/pass** |

The root cause: without `GGML_TYPE_Q3_K_HIFI` in the GPU-capable type list, the tensor
scheduler placed *all* tensors on CPU — including the SSM/GDN activation and state
tensors that Q3_K_M sends to GPU. Registering the type restores the correct device
assignment.

### 14.4 Small Model Weight Distribution (diagnostic findings)

For the Qwen3.5-0.8B model, measuring the top-8 weight magnitudes per block shows:

```
rank:  1      2      3      4    | 5      6      7      8
avg: 0.067  0.060  0.056  0.054 | 0.052  0.050  0.048  0.047
```

Rank-5 is 77% of rank-1. There is no "elbow" — these are not genuine outliers, just the
highest of a uniformly-distributed set. The outlier correction scheme is designed for
models where rank-1 is 3–10× larger than the median (typical in 7B+ models). For
sub-2B models with flat distributions:
- Zeroing positions before Q3_K quantization barely reduces the block scale.
- The FP16 corrections store near-average values, contributing minimal accuracy gain.
- The BPW overhead (~0.5 BPW for the 8-slot outlier extension) is the dominant effect.

**Conclusion:** Q3_K_HIFI is best suited for models ≥3B where genuine weight outliers
exist. For sub-2B models, the closest fair comparison is Q3_K_S (not Q3_K_M) at
similar BPW.

### 14.5 Model Size Behaviour (tensor selection)

| Model size | Q3_K_HIFI attn_v % | Q4_K_HIFI attn_v % | Critical tensor type |
|------------|---------------------|---------------------|----------------------|
| ≤1B | 0% | 32% | Q5_K_HIFI_RES8 |
| 1.7B | 0% (disabled) | 25% | Q5_K_HIFI_RES8 |
| 4B | 25% | 20% | Q5_K_HIFI_RES8 |
| 8–14B | 15% | 20% | Q6_K_HIFI_RES8 |
| 32B+ | 5% | 0% | Q6_K_HIFI_RES8 |

### 14.6 Imatrix Guidance Thresholds for Q3_K_HIFI

| Model size | Strategy |
|------------|----------|
| ≤2B | DISABLE imatrix-guided Q3_K_HIFI (hurts quality) |
| 2–5B | Top 30% of input projections |
| 5–10B | Top 20% |
| 10–20B | Top 15% |
| >20B | Top 10% |

---

## 15. Critical Implementation Notes

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

7. **Q3_K_HIFI outlier count by model size.** `ggml_q3_hifi_get_max_outliers` returns:
   - **TINY (≤1.7B): 2** — small models have near-uniform weight distributions (rank-8
     magnitude is ≈70% of rank-1; there are no genuine outliers). The TINY value of 2 is
     used only for the few blocks that run on the main thread; worker threads default to 8
     (see note 5 above). Empirically, forcing all blocks to 4 outliers via a context fix
     gave PPL 27.00 vs 25.09 for the accidental-8 path — *more* outliers can hurt when
     the weight distribution is flat because removing positions before Q3_K quantization
     changes the block scale without a commensurate accuracy gain from the corrections.
   - **MEDIUM (2B–8B): 8** — the sweet spot where genuine outliers exist.
   - **LARGE (14B+): 6** — outliers are rarer per-block at very large scale.
   The attn_v enhancement (`get_q3_hifi_attn_v_threshold`) is additionally disabled for
   ≤1.7B models, so those layers fall back to Q3_K_M behavior entirely.

8. **`block_q3_k_hifi` has an explicit `outlier_count` field** (stored in what was formerly
   `padding[0]`). The dequant loop must use this count — do NOT use a zero-index heuristic
   to detect the end of valid outliers. The correct pattern is:
   ```c
   int nc = (int)x[i].outlier_count;
   if (nc < 0 || nc > Q3_K_HIFI_MAX_OUTLIERS) nc = Q3_K_HIFI_MAX_OUTLIERS;
   for (int j = 0; j < nc; j++)
       tmp[x[i].outlier_idx[j]] = GGML_FP16_TO_FP32(x[i].outliers[j]);
   ```
   Other FP16 types (Q4_K_HIFI, Q2_K_HIFI) have a fixed max count and store all slots,
   so they do not need a separate count field.

10. **LITE base types.** Q3_K_LITE uses a **Q2_K** base block, Q4_K_LITE uses Q3_K, etc.
    (one tier lower). The LITE-type quantizer must call the N-1 tier base quantizer.

11. **GGUF metadata.** `llama-quant.cpp` writes `general.quantization_type` as a string
    ("Q4_K_HIFI", "Q5_K_HIFI") for HIFI ftypes.

12. **Environment variable debug hook.** Set `Q3_K_HIFI_DEBUG=1` to log Q3_K_HIFI tensor
    counts at model load time (in `llama-model-loader.cpp`).

13. **Double-promotion in C helper files.** When writing accumulation loops in C with
    `-Wdouble-promotion` enabled, a bare `float` operand in mixed arithmetic is silently
    promoted to `double`. Always cast explicitly:
    ```c
    // Wrong — float promoted to double implicitly:
    sum   += x[i];
    sumsq += (double)x[i] * x[i];
    if (fabs(weights[i]) > threshold) ...
    // Correct — both operands cast before arithmetic:
    sum   += (double)x[i];
    sumsq += (double)x[i] * (double)x[i];
    if (fabs((double)weights[i]) > threshold) ...
    ```
    This applies to `ggml-quants-hifi.c` and any accumulation helpers in `ggml-quants.c`.

12. **Unused parameter suppression.** If a function parameter is intentionally unused
    (e.g. `model_params_b` in `ggml_hifi_compute_block_outlier_count`), add `(void)param;`
    at the top of the function body to silence `-Wunused-parameter`.

13. **`ggml_validate_row_data` must handle all 13 types.** `ggml_validate_row_data` in
    `ggml-quants.c` has a `switch` over every `ggml_type`; the `default:` arm prints
    "invalid type" and returns `false`, which causes quantization to abort after every
    block is written. Add all 13 HIFI/LITE cases to the "nothing to validate" arm (the
    same arm as `GGML_TYPE_I8` / `I16` / `I32` / `I64`). Size-divisibility is already
    verified before the switch, so no further check is needed:
    ```c
    case GGML_TYPE_Q3_K_HIFI:
    case GGML_TYPE_Q6_K_HIFI:
    // ... all 13 types ...
    case GGML_TYPE_Q6_K_LITE:
        // nothing to validate beyond size check above
        break;
    ```

14. **GPU MUL_MAT registration affects tensor placement, not just GPU math.**
    Even with `--ngl 0` (no GPU layers), the tensor scheduler consults the GPU-capable
    type list to decide where to place *activation* and *state* tensors. A HIFI type
    missing from that list causes the scheduler to place the entire compute graph on CPU,
    including SSM/GDN state tensors that would otherwise run on GPU — a 26× slowdown.
    Always register every HIFI type in the GPU `MUL_MAT` supported-types block even if
    the type's GPU path is purely dequantize-then-cuBLAS (see §11.2.3).

15. **`block_q3_k_hifi` is not accessible inside CUDA/HIP translation units.**
    `ggml-common.h` wraps the struct in `#if !defined(GGML_COMMON_DECL_CUDA)`, so GPU
    kernels cannot use `sizeof(block_q3_k_hifi)` or field names. Access all extension
    fields via the hardcoded byte offsets documented in §11.2.1. Verify these offsets
    match `offsetof(block_q3_k_hifi, outlier_idx)` etc. in a CPU unit if the struct
    layout ever changes.

---

## 16. File Change Summary

| File | Change type | Notes |
|------|-------------|-------|
| `ggml/include/ggml.h` | Add enums | 13 new GGML_TYPE_* values (IDs 42–54) |
| `ggml/src/ggml-common.h` | Add structs | All HIFI + LITE block structs |
| `ggml/src/ggml.c` | Add registrations | Type traits table + quantize_chunk cases |
| `ggml/src/gguf.cpp` | Add name map | 13 type name strings |
| `ggml/src/ggml-quants.h` | Add declarations | ~30 new function prototypes |
| `ggml/src/ggml-quants.c` | Add implementations | Quant/dequant reference functions + `ggml_validate_row_data` cases |
| `ggml/src/ggml-quants-hifi.h` | NEW | Context API header |
| `ggml/src/ggml-quants-hifi.c` | NEW | Context API + HIFI helpers |
| `ggml/src/CMakeLists.txt` | Add source | ggml-quants-hifi.c/h |
| `ggml/src/ggml-impl.h` | Add helpers | Any HIFI-specific inline helpers |
| `ggml/src/ggml-cpu/quants.c` | Add CPU paths | Register function pointers |
| `ggml/src/ggml-cpu/quants.h` | Add declarations | CPU function prototypes |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Type dispatch | Handle new types in op switch |
| `ggml/src/ggml-cpu/ops.cpp` | Op dispatch | HIFI matmul handling |
| `ggml/src/ggml-cpu/repack.cpp` | Repack | Handle HIFI in repack paths |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | SIMD | Optional x86 optimizations |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | SIMD | Optional ARM NEON optimizations |
| `ggml/src/ggml-cuda/dequantize.cuh` | CUDA kernels | Dequant device functions |
| `ggml/src/ggml-cuda/convert.cu` | CUDA | Conversion kernels |
| `ggml/src/ggml-cuda/vecdotq.cuh` | CUDA | Vec-dot functions |
| `ggml/src/ggml-cuda/mmvq.cu` | CUDA | MMVQ dispatch |
| `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` | CUDA | MMQ paths |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA | Top-level dispatch |
| `ggml/src/ggml-cuda/common.cuh` | CUDA | Constants/helpers |
| `ggml/src/ggml-metal/ggml-metal-device.m` | **Required** | CPU fallback: exclude HIFI/LITE from `MUL_MAT` in `ggml_metal_device_supports_op` |
| `ggml/src/ggml-metal/ggml-metal.metal` | Optional (future) | Metal shader functions |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | Optional (future) | Pipeline registration |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | Optional (future) | Helper declarations |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Vulkan | Type registration, pipelines |
| `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` | Vulkan | Struct declarations |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` | Vulkan | Dequant functions |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_k_hifi.comp` | NEW Vulkan | |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q3_k_hifi.comp` | NEW Vulkan | |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k_hifi.comp` | NEW Vulkan | |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k_hifi.comp` | NEW Vulkan | |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | Vulkan | Register shaders |
| `ggml/src/ggml-sycl/dequantize.hpp` | SYCL | Template specializations |
| `ggml/src/ggml-sycl/vecdotq.hpp` | SYCL | Dot product specializations |
| `ggml/src/ggml-sycl/mmvq.cpp` | SYCL | MMVQ dispatch |
| `ggml/src/ggml-sycl/convert.cpp` | SYCL | Conversion kernels |
| `include/llama.h` | Add enums | 9 new LLAMA_FTYPE_* values (44–52) |
| `src/llama-quant.cpp` | Core logic | Tensor selection + HIFI context setup |
| `src/llama-model.h` | Ftype additions | (if ftype declared there) |
| `src/llama-model.cpp` | Integration | Any arch-specific handling |
| `src/llama-model-loader.cpp` | Loader | Ftype name strings + type_max map |
| `tools/quantize/quantize.cpp` | Quantize tool | Type name table entries |
| `tools/imatrix/imatrix.cpp` | Imatrix fix | log_softmax FP16 inf guard |
| `gguf-py/gguf/constants.py` | Python | GGMLQuantizationType + LlamaFileType + GGML_QUANT_SIZES block size entries |
| `gguf-py/gguf/quants.py` | Python | **No changes required** — reads GGML_QUANT_SIZES from constants.py |

