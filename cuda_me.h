#ifndef CUDA_ME_H
#define CUDA_ME_H

void cuda_c63_motion_estimate(int width, int height,
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        struct macroblock *mbs[3]);

void cuda_c63_motion_compensate(int width, int height,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        struct macroblock *mbs[3]);

#endif
