#ifndef CUDA_ME_H
#define CUDA_ME_H

typedef struct {
    int sad, mv_x, mv_y;
} mv_out_t;

void cuda_c63_motion_estimate(struct c63_common *cm);

#endif
