#ifndef CUDA_DCT_H
#define CUDA_DCT_H

void cuda_dct_quantize(uint32_t width, uint32_t height,
        uint8_t id_quant, uint8_t *d_in_data, uint8_t *d_prediction,
        int16_t *d_out_data);

void printf_init();
void printf_end();

//extern void cuda_dct_malloc(size_t size, uint8_t **d_in_data, uint8_t **d_prediction, int16_t **d_out_data);

//extern void cuda_dct_free(uint8_t *d_in_data, uint8_t *d_prediction, int16_t *d_out_data);

#endif
