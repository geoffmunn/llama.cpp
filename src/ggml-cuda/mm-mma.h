#pragma once

/// Maximum batch size threshold for switching between MMVQ and cuBLAS paths.
/// Batch sizes <= this value use the MMVQ path; larger batches fall through to cuBLAS.
#define MMVQ_MAX_BATCH_SIZE 128
