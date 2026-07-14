#pragma once

/// Outlier correction strategy for HIFI quantization types.
///
/// FP16 variants (Q3_K_HIFI, Q4_K_HIFI, Q6_K_HIFI, Q2_K_HIFI) use REPLACE:
///   the base-dequant value is entirely replaced by the stored FP16 outlier
///   value at each outlier position.
///
/// INT8 residual variants (RES8, LITE) use ADD:
///   the stored INT8 correction is added to the base-dequant value at each
///   outlier position.
///
/// Never mix the two patterns within a single type.
#define GGML_HIOUTLIER_REPLACE 0  ///< Replace base-dequant value entirely at outlier positions
#define GGML_HIOUTLIER_ADD     1  ///< Add correction to base-dequant value at outlier positions
