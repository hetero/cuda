#ifndef CUDA_IDCT_H
#define CUDA_IDCT_H

extern void cuda_dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
			 uint8_t *out_data, uint8_t id_quant);

#endif
