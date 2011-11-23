#ifndef CUDA_IDCT_H
#define CUDA_IDCT_H

void cuda_dequantize_idct(uint32_t width, uint32_t height, 
        uint8_t id_quant, int16_t *d_in_data, uint8_t *d_prediction,
        uint8_t *d_out_data);

#endif
