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
//#include "cuPrintf.cu"

#define REF_WIDTH 48
#define REF_HEIGHT 48
#define ORIG_SIZE 8

__device__ void cuda_sad_block_8x8(uint8_t *block1, uint8_t *block2,
        int *result)
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
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    block2 += 40;

    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res); ++block1; ++block2;
    res = __sad(*block2, *block1, res);

    // sadxy = sad*1024 + (mv_x+16)*32 + (mv_y+16)
    *result = (res << 10);
}

__global__ void k_me_block_8x8(uint8_t *orig, uint8_t *ref, int *mv_out, int w, int h)
{
    __shared__ int best_sadxy;
    __shared__ uint8_t shared_orig[ORIG_SIZE * ORIG_SIZE];
    __shared__ uint8_t shared_ref[REF_HEIGHT * REF_WIDTH];
    best_sadxy = INT_MAX;

    // copying ORIG global->shared
    if (threadIdx.y < 8) {
        int x = 4 * (threadIdx.x % 2) + (threadIdx.y / 2);
        int y = (threadIdx.x / 2) + 4 * (threadIdx.y % 2);
        shared_orig[y * ORIG_SIZE + x]
            = orig[(blockIdx.y * 8 + y) * w 
            + (blockIdx.x * 8 + x)];
    }

    int left = (blockIdx.x*8 - 16);
    int top = (blockIdx.y*8 - 16);
    int right = (blockIdx.x*8 + 16);
    int bottom = (blockIdx.y*8 + 16);

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

#define rightEnd (right+8)
#define bottomEnd (bottom+8)


    //copying REF

    // 32 x 32
    int x1 = 4 * threadIdx.x;
    int y1 = 2 * threadIdx.y;
    for (int i = 0; i < 8; i++) {
        if (y1 < bottom - top && x1 < right - left) {
            int i_x = x1 + (i & 3);
            int i_y = y1 + (i >> 2);
            shared_ref[i_y * REF_WIDTH + i_x] = 
                ref[(top + i_y) * w + left + i_x];
        }
    }

    // bottom
    int x2 = 4 * threadIdx.x + 2 * (threadIdx.y / 8);
    int y2 = bottom - top + 4 * ((threadIdx.y % 8) / 4) 
        + (threadIdx.y % 4) / 2 + 2 * (threadIdx.y % 2);
    if (x2 < right - left) {
        shared_ref[y2 * REF_WIDTH + x2] = 
            ref[(top + y2) * w + left + x2];
        shared_ref[y2 * REF_WIDTH + x2 + 1] = 
            ref[(top + y2) * w + left + x2 + 1];
    }

    // right
    if (threadIdx.y % 2 == 0) {
        int x = 4 * (threadIdx.x % 2);
        int y = 4 * (threadIdx.y / 2) + (threadIdx.x / 2);
        if (y < bottom - top) {
            for (int i = 0; i < 4; ++i) {
                int i_x = right - left + x + (i & 3);
                int i_y = y;
                shared_ref[i_y * REF_WIDTH + i_x]
                    = ref[(top + i_y) * w + left + i_x];
            }
        }
    }

    // corner
    if (threadIdx.y % 2 == 0) {
        int x = right - left + 
            4 * (threadIdx.x % 2) + ((threadIdx.y / 2) / 2);
        int y = bottom - top + 
            4 * ((threadIdx.y / 2) % 2) + (threadIdx.x / 2);
        shared_ref[y * REF_WIDTH + x]
            = ref[(top + y) * w + left + x];
    }
    
    __syncthreads();

    int x = 4 * threadIdx.x;
    int y = 2 * threadIdx.y;
    // SAD
    if (top + y < bottom && 
            left + x < right)
    {
        int mv_xy = ((left + x - blockIdx.x * 8 + 16) << 5) 
            + (top + y - blockIdx.y * 8 + 16);
        int sad, minsad = INT_MAX;
        for (int i = 0; i < 8; ++i) {
            cuda_sad_block_8x8(shared_orig, shared_ref 
                    + (y + (i >> 2)) * REF_WIDTH
                    + x + (i & 3), 
                &sad);
            minsad = min(sad + ((i & 3) << 5) + (i >> 2), minsad);
        }
        atomicMin(&best_sadxy, minsad + mv_xy);
    }

    __syncthreads();

    // write out
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mv_out[blockIdx.y * w / 8 + blockIdx.x] = best_sadxy;
    }
}

void cuda_me_cc(struct c63_common *cm, int cc)
{
    /* Compare this frame with previous reconstructed frame */
    int mb_x, mb_y;

    uint8_t *orig, *ref;
    int *mv_out_dev, *mv_out_host;
    int frame_size = cm->padw[cc] * cm->padh[cc];
    int mb_cols = cm->padw[cc] / 8;
    int mb_rows = cm->padh[cc] / 8;
    int blocks = mb_cols * mb_rows;
    cudaMalloc(&orig, frame_size * sizeof(uint8_t));
    cudaMalloc(&ref, frame_size * sizeof(uint8_t));
    cudaMalloc(&mv_out_dev, blocks * sizeof(int));

    mv_out_host = (int *) malloc(blocks * sizeof(int));

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

    dim3 threadsPerBlock(8, 16);
    dim3 numBlocks(mb_cols, mb_rows);
    
    k_me_block_8x8<<<numBlocks, threadsPerBlock>>>
        (orig, ref, mv_out_dev, cm->padw[cc], cm->padh[cc]); 
     
    // Copy result from device memory to host memory
    cudaMemcpy(mv_out_host, mv_out_dev, blocks * sizeof(int), 
            cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(orig);
    cudaFree(ref);
    cudaFree(mv_out_dev);

    for (mb_y = 0; mb_y < mb_rows; ++mb_y) {
        for (mb_x = 0; mb_x < mb_cols; ++mb_x) {
            int block_nr = mb_y * mb_cols + mb_x;
            // sadxy = sad*1024 + (mv_x+16)*32 + (mv_y+16)
            int sadxy = mv_out_host[block_nr];
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
//    cudaPrintfInit();
    for (int cc = 0; cc <= 2; cc++) {
        cuda_me_cc(cm, cc);
    }
//    cudaPrintfDisplay();
//    cudaPrintfEnd();
}
