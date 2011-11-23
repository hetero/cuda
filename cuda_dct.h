#ifndef CUDA_DCT_H
#define CUDA_DCT_H

void cuda_dct_quantize(uint32_t width, uint32_t height,
        uint8_t id_quant, uint8_t *d_in_data, uint8_t *d_prediction,
        int16_t *d_out_data);

#endif
