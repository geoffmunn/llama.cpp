#ifndef QUANTIZE_H
#define QUANTIZE_H

/**
 * Return the enhancement threshold (0.0–1.0) used by the MOSTLY_Q4_K_HIFI
 * ftype to decide whether an attention-weight tensor should be promoted
 * to a higher-precision type.
 *
 * Smaller models get a higher threshold (more tensors promoted); larger
 * models get a lower threshold (fewer tensors promoted).
 *
 * \param model_params_b  Model size in billions of parameters.
 * \return                Fraction of attention tensors that qualify for
 *                        the enhanced type.
 */
float get_hifi_enhancement_threshold(float model_params_b);

#endif // QUANTIZE_H
