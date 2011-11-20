#ifndef CUDA_DCT_H
#define CUDA_DCT_H

extern void cuda_dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t id_quant,
        uint8_t *d_in_data, uint8_t *d_prediction, int16_t *d_out_data);

extern void cuda_dct_malloc(size_t size, uint8_t **d_in_data, uint8_t **d_prediction, int16_t **d_out_data);

extern void cuda_dct_free(uint8_t *d_in_data, uint8_t *d_prediction, int16_t *d_out_data);

#endif
