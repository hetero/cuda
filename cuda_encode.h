#ifndef CUDA_ENCODE_H
#define CUDA_ENCODE_H

#include "c63.h"

void cuda_init_c63_encode(int width, int height,
        uint8_t **origY, uint8_t **origU, uint8_t **origV,
        uint8_t **reconsY, uint8_t **reconsU, uint8_t **reconsV,
        uint8_t **predY, uint8_t **predU, uint8_t **predV,
        int16_t **residY, int16_t **residU, int16_t **residV,
        struct macroblock **mbsY, struct macroblock **mbsU,
        struct macroblock **mbsV
    );

void cuda_free_c63_encode(
        uint8_t **origY, uint8_t **origU, uint8_t **origV,
        uint8_t **reconsY, uint8_t **reconsU, uint8_t **reconsV,
        uint8_t **predY, uint8_t **predU, uint8_t **predV,
        int16_t **residY, int16_t **residU, int16_t **residV,
        struct macroblock **mbsY, struct macroblock **mbsU,
        struct macroblock **mbsV
    );

void cuda_copy_image(int width, int height, yuv_t *image, 
        uint8_t *origY, uint8_t *origU, uint8_t *origV
    );


void cuda_c63_encode_image(struct c63_common *cm, int width, int height,
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbsY, struct macroblock *mbsU,
        struct macroblock *mbsV
    );

#endif
