#ifndef CUDA_DCT_H
#define CUDA_DCT_H

extern void cuda_dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t id_quant);

#endif
