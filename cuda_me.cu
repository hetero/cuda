#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <sm_11_atomic_functions.h>
#include <sm_12_atomic_functions.h>
#include <sm_13_double_functions.h>

#include "c63.h"
#include "cuda_me.h"

#define REF_SIZE 39
#define ORIG_SIZE 8

__device__ void cuda_sad_block_8x8(uint8_t *block1, uint8_t *block2,
        int mv_xy, int *result)
{
    int res = 0;
    
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 31;

    // sadxy = sad*1024 + (mv_x+16)*32 + (mv_y+16)
    *result = (res << 10) + mv_xy;
}

__global__ void k_me_block_8x8(uint8_t *orig, uint8_t *ref, mv_out_t *mv_out, int w, int h)
{
    __shared__ int best_sadxy;
    __shared__ uint8_t shared_orig[ORIG_SIZE * ORIG_SIZE];
    __shared__ uint8_t shared_ref[REF_SIZE * REF_SIZE];
    best_sadxy = INT_MAX;

    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    int block_nr = mb_y * w / 8 + mb_x;
    
    int mx = mb_x * 8;
    int my = mb_y * 8;
    
    // copying ORIG global->shared
    if (threadIdx.x < ORIG_SIZE && threadIdx.y < ORIG_SIZE)
        shared_orig[threadIdx.y * ORIG_SIZE + threadIdx.x]
            = orig[(my+threadIdx.y) * w + (mx+threadIdx.x)];


    int range = 16; //TODO

    int left = mb_x*8 - range;
    int top = mb_y*8 - range;
    int right = mb_x*8 + range;
    int bottom = mb_y*8 + range;

    // Make sure we are within bounds of reference frame
    // TODO: Support partial frame bounds
    if (left < 0)
        left = 0;
    if (top < 0)
        top = 0;
    if (right > (w - 8))
        right = w - 8;
    if (bottom > (h - 8))
        bottom = h - 8;

    int rightEnd = right+7;
    int bottomEnd = bottom+7;

    //copying REF

    //1st whole block
    if (left + threadIdx.x < rightEnd && 
            top + threadIdx.y < bottomEnd) {
        shared_ref[threadIdx.y * REF_SIZE + threadIdx.x] =
            ref[(top + threadIdx.y) * w + (left + threadIdx.x)];
    }
    //2nd whole block
    if (left + 16 + threadIdx.x < rightEnd 
            && top + threadIdx.y < bottomEnd) {
        shared_ref[threadIdx.y * REF_SIZE + 16 + threadIdx.x] =
            ref[(top + threadIdx.y) * w + (left + 16 + threadIdx.x)];
    }

    //right border
    if (threadIdx.x < 7) {
        if (left + 32 + threadIdx.x < rightEnd
                && top + threadIdx.y < bottomEnd) {
            shared_ref[threadIdx.y * REF_SIZE + 32 + threadIdx.x] =
                ref[(top + threadIdx.y) * w + 
                    (left + 32 + threadIdx.x)];
        }
    }
    //bottom border
    else if (threadIdx.x < 14) {
        if (top + 32 + (threadIdx.x - 7) < bottomEnd 
                && left + threadIdx.y < rightEnd) {
            shared_ref[(32 + (threadIdx.x - 7)) * REF_SIZE +
                threadIdx.y] =
                ref[(top + 32 + (threadIdx.x - 7)) * w 
                    + (left + threadIdx.y)];
        }
    }
    //right-bottom corner
    else if ((threadIdx.y & 7) != 7) {
        int x = (threadIdx.y >> 3) + 4 * (threadIdx.x - 14);
        int y = threadIdx.y & 7;
        if (top + 32 + y < bottomEnd && left + 32 + x < rightEnd) {
            shared_ref[(32 + y) * REF_SIZE + (32 + x)] =
                ref[(top + 32 + y) * w + (left + 32 + x)];
        }
    }

    __syncthreads();

    int x = 2 * threadIdx.x;
    int y = threadIdx.y;

    if (top+y<bottom && left+x<right)
    {
        int sad1, sad2;
        int mv_xy = ((left+x-mx + 16) << 5) + (top+y-my + 16);
        cuda_sad_block_8x8(shared_orig, shared_ref + y*REF_SIZE+x, 
                mv_xy, &sad1);
        x++;
        cuda_sad_block_8x8(shared_orig, shared_ref + y*REF_SIZE+x, 
                mv_xy + 32, &sad2);
        atomicMin(&best_sadxy, min(sad1, sad2));
    }

    __syncthreads();

    mv_out[block_nr].sadxy = best_sadxy;
}

void cuda_me_cc(struct c63_common *cm, int cc)
{
    /* Compare this frame with previous reconstructed frame */
    int mb_x, mb_y;

    uint8_t *orig, *ref;
    mv_out_t *mv_out_dev, *mv_out_host;
    int frame_size = cm->padw[cc] * cm->padh[cc];
    int mb_cols = cm->padw[cc] / 8;
    int mb_rows = cm->padh[cc] / 8;
    int blocks = mb_cols * mb_rows;
    cudaMalloc(&orig, frame_size * sizeof(uint8_t));
    cudaMalloc(&ref, frame_size * sizeof(uint8_t));
    cudaMalloc(&mv_out_dev, blocks * sizeof(mv_out_t));

    mv_out_host = (mv_out_t *) malloc(blocks * sizeof(mv_out_t));

    // Copy vectors from host memory to device memory
    uint8_t *cur, *recons;
    switch (cc) {
        case 0:
            cur = cm->curframe->orig->Y;
            recons = cm->refframe->recons->Y;
            break;
        case 1:
            cur = cm->curframe->orig->U;
            recons = cm->refframe->recons->U;
            break;
        case 2:
            cur = cm->curframe->orig->V;
            recons = cm->refframe->recons->V;
    }
    
    cudaMemcpy(orig, cur, frame_size, 
            cudaMemcpyHostToDevice);
    cudaMemcpy(ref, recons, frame_size, 
            cudaMemcpyHostToDevice);
    
    // Invoke kernel

    dim3 threadsPerBlock(16, 32);
    dim3 numBlocks(mb_cols, mb_rows);
    
    k_me_block_8x8<<<numBlocks, threadsPerBlock>>>
        (orig, ref, mv_out_dev, cm->padw[cc], cm->padh[cc]); 
    
    // Copy result from device memory to host memory
    cudaMemcpy(mv_out_host, mv_out_dev, blocks * sizeof(mv_out_t), 
            cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(orig);
    cudaFree(ref);
    cudaFree(mv_out_dev);

    for (mb_y = 0; mb_y < mb_rows; ++mb_y) {
        for (mb_x = 0; mb_x < mb_cols; ++mb_x) {
            int block_nr = mb_y * mb_cols + mb_x;
            // sadxy = sad*1024 + (mv_x+16)*32 + (mv_y+16)
            int sadxy = mv_out_host[block_nr].sadxy;
            int sad = sadxy >> 10;
            int mv_x = ((sadxy >> 5) & 31) - 16;
            int mv_y = (sadxy & 31) - 16;
            struct macroblock *mb = &cm->curframe->mbs[cc][block_nr];
            //printf("(%d,%d): MV = (%d,%d), sad=%d\n",mb_x,mb_y,mv_x,
            //        mv_y,sad);
            if (sad < 512) {
                mb->use_mv = 1;
                mb->mv_x = mv_x;
                mb->mv_y = mv_y;
            }
            else {
                mb->use_mv = 0;
            }
        }
    }
}

void cuda_c63_motion_estimate(struct c63_common *cm) {
    for (int cc = 0; cc <= 2; cc++) {
        cuda_me_cc(cm, cc);
    }
}
