#include "llama.h"
#include "ggml.h"
#include "ggml/src/ggml-quants-hifi.h"
#include <cstdio>
#include <cstring>

// GGUF key for imatrix threshold configuration data
#define GGUF_KEY_IMATRIX_THRESHOLD_CONFIG "general.imatrix.threshold_config"

// GGUF value types (matches gguf_type enum in gguf.h)
#define GGUF_TYPE_UINT8     0
#define GGUF_TYPE_INT8      1
#define GGUF_TYPE_UINT16    2
#define GGUF_TYPE_INT16     3
#define GGUF_TYPE_UINT32    4
#define GGUF_TYPE_INT32     5
#define GFUF_TYPE_ARRAY    22

/// Read ggml_imatrix_threshold_config_t from a GGUF file header.
///
/// Scans the GGUF key-value metadata for the imatrix threshold config entry.
/// Returns 0 on success (populates *config), non-zero on error:
///   1 — file could not be opened or header parse failed
///   2 — no imatrix threshold config key found in the GGUF metadata
///   3 — config data present but too small to hold the expected struct
int gguf_read_imatrix_threshold_config(const char * fname, ggml_imatrix_threshold_config_t * config) {
    if (!fname || !config) {
        return -1;
    }

    FILE * f = std::fopen(fname, "rb");
    if (!f) {
        return 1;
    }

    // --- Read and validate header ---
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4 ||
        std::memcmp(magic, "GGUF", 4) != 0) {
        std::fclose(f);
        return 1;
    }

    uint32_t version = 0;
    if (std::fread(&version, sizeof(version), 1, f) != 1) {
        std::fclose(f);
        return 1;
    }

    int64_t n_tensors = 0;
    int64_t n_kv      = 0;
    if (std::fread(&n_tensors, sizeof(n_tensors), 1, f) != 1 ||
        std::fread(&n_kv,      sizeof(n_kv),      1, f) != 1) {
        std::fclose(f);
        return 1;
    }

    // --- Walk KV pairs looking for the imatrix threshold config ---
    const size_t expected_size = sizeof(ggml_imatrix_threshold_config_t);
    bool found = false;

    for (int64_t i = 0; i < n_kv; ++i) {
        // Read key string length
        uint32_t key_len = 0;
        if (std::fread(&key_len, sizeof(key_len), 1, f) != 1) {
            std::fclose(f);
            return 1;
        }

        // Read key string bytes
        std::string key_str(key_len, '\0');
        if (key_len > 0 && std::fread(&key_str[0], 1, key_len, f) != key_len) {
            std::fclose(f);
            return 1;
        }

        // Read value type
        uint32_t vtype = 0;
        if (std::fread(&vtype, sizeof(vtype), 1, f) != 1) {
            std::fclose(f);
            return 1;
        }

        bool is_array = false;
        uint64_t arr_count = 1;
        if (vtype == GFUF_TYPE_ARRAY) {
            is_array = true;
            uint32_t inner_type = 0;
            if (std::fread(&inner_type, sizeof(inner_type), 1, f) != 1) {
                std::fclose(f);
                return 1;
            }
            if (std::fread(&arr_count, sizeof(arr_count), 1, f) != 1) {
                std::fclose(f);
                return 1;
            }
            vtype = inner_type;
        }

        // If this is our target key, read the value into config
        if (key_str == GGUF_KEY_IMATRIX_THRESHOLD_CONFIG) {
            if (!is_array || arr_count * sizeof(float) < expected_size) {
                std::fclose(f);
                return 3;
            }
            // Read raw float array into the struct
            if (std::fread(config, 1, expected_size, f) != expected_size) {
                std::fclose(f);
                return 1;
            }
            found = true;
            break;
        }

        // Not our key — skip the value bytes
        size_t val_size = 0;
        switch (vtype) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
                val_size = (is_array ? arr_count : 1) * 1;
                break;
            case 8:  // FLOAT32
                val_size = (is_array ? arr_count : 1) * 4;
                break;
            case 9:  // UINT64
            case 12: // INT64
            case 13: // FLOAT64
                val_size = (is_array ? arr_count : 1) * 8;
                break;
            case 14: // STRING — each element has u32 length prefix
                for (uint64_t s = 0; s < arr_count; ++s) {
                    uint32_t slen = 0;
                    if (std::fread(&slen, sizeof(slen), 1, f) != 1) {
                        std::fclose(f);
                        return 1;
                    }
                    if (slen > 0 && std::fseek(f, (long)slen, SEEK_CUR) != 0) {
                        std::fclose(f);
                        return 1;
                    }
                }
                val_size = 0; // already consumed
                break;
            default:
                std::fclose(f);
                return 1;
        }

        if (val_size > 0 && std::fseek(f, (long)val_size, SEEK_CUR) != 0) {
            std::fclose(f);
            return 1;
        }
    }

    std::fclose(f);

    if (!found) {
        return 2;
    }
    return 0;
}

/// Convenience wrapper: load config from GGUF file path, returning error if absent or unreadable.
int gguf_load_imatrix_threshold_config(const char * fname, ggml_imatrix_threshold_config_t * out_config) {
    return gguf_read_imatrix_threshold_config(fname, out_config);
}
