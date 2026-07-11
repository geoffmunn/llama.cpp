#pragma once

/// @file
/// Top-K percentage thresholds for imatrix-guided HIFI quantization (Q3_K_HIFI).
///
/// The percentage determines what fraction of input projections receive
/// high-fidelity outlier treatment, mapped against model parameter count.
///
/// Reference: §14.6 Imatrix Guidance Thresholds for Q3_K_HIFI

#include <cstdint>
#include <array>

namespace gguf {

/// Model size ranges and their associated top-K percentage for HIFI quantization.
struct hifi_topk_range {
    /// Upper bound on parameter count (in billions), inclusive.
    /// Use UINT64_MAX for the open-ended ">20B" bucket.
    uint64_t params_billions;

    /// Top-K percentage applied in this range.
    /// 0 means HIFI imatrix guidance is disabled entirely.
    uint8_t topk_percent;
};

/// Thresholds from smallest to largest model.
///
/// | Range   | topk_percent | Meaning                              |
/// |---------|-------------|--------------------------------------|
/// | ≤ 2 B   | 0           | DISABLED (imatrix hurts quality)     |
/// | 2–5 B   | 30          | Keep top 30 % of input projections   |
/// | 5–10 B  | 20          | Keep top 20 %                        |
/// | 10–20 B | 15          | Keep top 15 %                        |
/// | > 20 B  | 10          | Keep top 10 %                        |
inline constexpr std::array<hifi_topk_range, 5> hifi_topk_thresholds = {{
    {2, 0},   // ≤2B  : disabled
    {5, 30},  // 2–5B : top 30 %
    {10, 20}, // 5–10B: top 20 %
    {20, 15}, // 10–20B: top 15 %
    {UINT64_MAX, 10}, // >20B: top 10 %
}};

} // namespace gguf
