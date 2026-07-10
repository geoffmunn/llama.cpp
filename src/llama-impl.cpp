#include "llama-impl.h"

#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <limits>
#include <vector>
#include <sstream>

struct llama_logger_state {
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_logger_state g_logger_state;

time_meas::time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

void llama_log_get(ggml_log_callback * log_callback, void ** user_data) {
    ggml_log_get(log_callback, user_data);
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    ggml_log_set(log_callback, user_data);
    g_logger_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, ne.at(i));
    }
    return buf;
}

std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

//
// outlier detection
//

std::vector<int> llama_find_top_outliers(const float * row,
                                         int64_t n_per_row,
                                         int n_outliers,
                                         const float * imatrix) {
    if (row == nullptr || n_per_row <= 0 || n_outliers <= 0) {
        return {};
    }

    // Clamp requested outlier count
    if (static_cast<int64_t>(n_outliers) > n_per_row) {
        n_outliers = static_cast<int>(n_per_row);
    }

    // --- Stage 1: compute per-element scores ---
    // score[i] = fabs(weight[i]) * importance[i]
    std::vector<float> scores(n_per_row);
    for (int64_t i = 0; i < n_per_row; ++i) {
        float importance = (imatrix != nullptr) ? fabsf(imatrix[i]) : 1.0f;
        scores[i] = fabsf(row[i]) * importance;
    }

    // --- Stage 2: 3-sigma threshold ---
    float sum = 0.0f;
    for (float s : scores) {
        sum += s;
    }
    float mean = sum / static_cast<float>(n_per_row);

    float var_sum = 0.0f;
    for (float s : scores) {
        float d = s - mean;
        var_sum += d * d;
    }
    float stddev = sqrtf(var_sum / static_cast<float>(n_per_row));

    // Threshold: elements with score > mean + 3*stddev are candidates
    float threshold = mean + 3.0f * stddev;

    // --- Stage 3: rank candidates by score descending, take top-N ---
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(n_per_row);
    for (int64_t i = 0; i < n_per_row; ++i) {
        if (scores[i] > threshold) {
            candidates.emplace_back(scores[i], static_cast<int>(i));
        }
    }

    // Partial sort to get the top-n_outliers by score (descending)
    int take = std::min(n_outliers, static_cast<int>(candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(),
                     [](const auto & a, const auto & b) { return a.first > b.first; });
    candidates.resize(take);

    // Collect indices and sort ascending
    std::vector<int> result;
    result.reserve(candidates.size());
    for (auto & p : candidates) {
        result.push_back(p.second);
    }
    std::sort(result.begin(), result.end());

    return result;
}

