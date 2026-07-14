// threading-cpu.cc — Worker-thread-aware outlier defaults for quantization
//
// Guide §7: TINY models (≤1.7B) use 2 outliers on the main thread, but worker
// threads default to 8 because the few blocks that land on workers still benefit
// from the higher budget.  Non-TINY model sizes are unaffected.

#include <thread>
#include <atomic>

// ------------------------------------------------------------------
// Thread-role tracking
// ------------------------------------------------------------------

static std::atomic<bool> g_worker_thread_active{false};

/// Call before spawning worker threads (main thread side).
void ggml_quantization_enter_workers() {
    g_worker_thread_active.store(true);
}

/// Call after all worker threads have joined (main thread side).
void ggml_quantization_exit_workers() {
    g_worker_thread_active.store(false);
}

/// Returns true if the current thread is a quantization worker.
static bool is_quantization_worker_thread() {
    return g_worker_thread_active.load();
}

// ------------------------------------------------------------------
// Outlier count helpers
// ------------------------------------------------------------------

/// Default outlier count for TINY models when running on a worker thread.
static constexpr int kTinyWorkerOutliers = 8;

/// Return the effective maximum outlier count for the given model size,
/// taking thread role into account.
///
/// - TINY (<2B):   main thread → \p small_limit (usually 2),
///                 worker      → 8
/// - MEDIUM (2-8B): always 8
/// - LARGE (≥14B):  always 6
///
/// \param model_params_b  Model size in billions of parameters.
/// \param small_limit     Configured small-limit value for TINY models
///                        (typically 2; see guide §7).
/// \return                Effective outlier count.
int ggml_q3_hifi_get_max_outliers(float model_params_b, int small_limit) {
    if (model_params_b < 2.0f) {
        // TINY model
        if (is_quantization_worker_thread()) {
            return kTinyWorkerOutliers;
        }
        return small_limit;
    } else if (model_params_b <= 8.0f) {
        return 8;
    } else {
        return 6;
    }
}
