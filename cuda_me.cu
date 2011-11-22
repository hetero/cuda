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

#define rightEnd (right+8)
#define bottomEnd (bottom+8)

__device__ void cuda_sad_block_8x8(uint8_t *block1, uint8_t *block2,
        int *result)
{
    int sum, minsad = INT_MAX;
    uint8_t *b1, *b2;

    for (int k = 0; k < 4; ++k) {
        sum = 0;
        b1 = block1;
        b2 = block2 + k;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                int l1 = *b1;
                int l2 = *b2;
                sum = __sad(l2, l1, sum); ++b1; ++b2;
            }
            b2 += 40;
        }
        minsad = min((sum << 10) + k, minsad);
    }
    for (int k = 0; k < 4; ++k) {
        sum = 0;
        b1 = block1;
        b2 = block2 + REF_WIDTH + k;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                int l1 = *b1;
                int l2 = *b2;
                sum = __sad(l2, l1, sum); ++b1; ++b2;
            }
            b2 += 40;
        }
        minsad = min((sum << 10) + 32 + k, minsad);
    }
    // sadxy = sad*1024 + (mv_y+16)*32 + (mv_x+16)
    *result = minsad;
}

__global__ void k_me_block_8x8(uint8_t *orig, uint8_t *ref, int *mv_out, int w, int h)
{
    __shared__ int best_sadxy;
    __shared__ uint8_t shared_orig[ORIG_SIZE * ORIG_SIZE];
    __shared__ uint8_t shared_ref[REF_HEIGHT * REF_WIDTH];
    best_sadxy = INT_MAX;

    // copying ORIG global->shared
    int x = 4 * (threadIdx.x % 2) + (threadIdx.y / 2);
    int y = (threadIdx.x / 2) + 4 * (threadIdx.y % 2);
    if (threadIdx.y < 8) {
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

    //copying REF

    // 32 x 32
    x = 4 * threadIdx.x;
    y = 2 * threadIdx.y;
    for (int i = 0; i < 8; i++) {
        if (y < bottom - top && x < right - left) {
            int i_x = x + (i & 3);
            int i_y = y + (i >> 2);
            shared_ref[i_y * REF_WIDTH + i_x] = 
                ref[(top + i_y) * w + left + i_x];
        }
    }
    // bottom
    x = 4 * threadIdx.x + 2 * (threadIdx.y / 8);
    y = bottom - top + 1 * ((threadIdx.y % 8) / 4) 
        + 4 * ((threadIdx.y % 4) / 2) + 2 * (threadIdx.y % 2);
    
    if (x < right - left) {
        shared_ref[y * REF_WIDTH + x] = 
            ref[(top + y) * w + left + x];
        shared_ref[y * REF_WIDTH + x + 1] = 
            ref[(top + y) * w + left + x + 1];
    }
    
    // right
    x = 4 * (threadIdx.x % 2);
    y = 4 * (threadIdx.y / 2) + (threadIdx.x / 2);
    if (threadIdx.y % 2 == 0) {
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
    x = right - left + 
        4 * (threadIdx.x % 2) + ((threadIdx.y / 2) / 2);
    y = bottom - top + 
        4 * ((threadIdx.y / 2) % 2) + (threadIdx.x / 2);
    if (threadIdx.y % 2 == 0) {
        shared_ref[y * REF_WIDTH + x]
            = ref[(top + y) * w + left + x];
    }
    
    __syncthreads();

    x = 4 * threadIdx.x;
    y = 2 * threadIdx.y;
    // SAD
    if (top + y < bottom && 
            left + x < right)
    {
        int mv_xy = (left + x - blockIdx.x * 8 + 16) 
            + ((top + y - blockIdx.y * 8 + 16) << 5);
        int sad;

        cuda_sad_block_8x8(shared_orig, shared_ref + y * REF_WIDTH + x, &sad);
        atomicMin(&best_sadxy, sad + mv_xy);
    }

    __syncthreads();

    // write out
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mv_out[blockIdx.y * w / 8 + blockIdx.x] = best_sadxy;
    }
}

__global__ void k_mb(int *mv_out, struct macroblock *mbs,
        int padw, int padh)
{
    int mb_cols = padw / 8;
    int mb_rows = padh / 8;
    for (int mb_y = 0; mb_y < mb_rows; ++mb_y) {
        for (int mb_x = 0; mb_x < mb_cols; ++mb_x) {
            int block_nr = mb_y * mb_cols + mb_x;
            // sadxy = sad*1024 + (mv_y+16)*32 + (mv_x+16)
            int sadxy = mv_out[block_nr];
            int sad = sadxy >> 10;
            int mv_y = ((sadxy >> 5) & 31) - 16;
            int mv_x = (sadxy & 31) - 16;
            struct macroblock *mb = &mbs[block_nr];
            
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

void cuda_me_cc(int padw, int padh, uint8_t *orig, uint8_t *ref,
        struct macroblock *mbs)
{
    /* Compare this frame with previous reconstructed frame */
    int *mv_out_dev;
    int mb_cols = padw / 8;
    int mb_rows = padh / 8;
    int blocks = mb_cols * mb_rows;
    cudaMalloc(&mv_out_dev, blocks * sizeof(int));

    // Invoke kernel

    dim3 threadsPerBlock(8, 16);
    dim3 numBlocks(mb_cols, mb_rows);
    
    k_me_block_8x8<<<numBlocks, threadsPerBlock>>>
        (orig, ref, mv_out_dev, padw, padh); 
     
    k_mb<<<1, 1>>> (mv_out_dev, mbs, padw, padh);

    cudaFree(mv_out_dev);
}

void cuda_c63_motion_estimate(int width, int height,
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        struct macroblock *mbs[3]) 
{
//    cudaPrintfInit();
    cuda_me_cc(width, height, origY, reconsY, mbs[0]);
    cuda_me_cc(width / 2, height / 2, origU, reconsU, mbs[1]);
    cuda_me_cc(width / 2, height / 2, origV, reconsV, mbs[2]);
//    cudaPrintfDisplay();
//    cudaPrintfEnd();
}


__global__ void cuda_mc_block_8x8(int w, int h, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, struct macroblock *mb) 
{
    if (!mb->use_mv)
        return;

    int left = mb_x*8;
    int top = mb_y*8;
    int right = left + 8;
    int bottom = top + 8;

    /* Copy block from ref mandated by MV */
    int x,y;
    for (y=top; y < bottom; ++y)
    {   
        for (x=left; x < right; ++x)
        {   
            predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
        }
    }       
}

void cuda_c63_motion_compensate(int width, int height,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        struct macroblock *mbs[3]) 
{
    int mb_cols = width / 8;
    int mb_rows = height / 8;
    int mb_x, mb_y;

    /* Luma */
    for (mb_y=0; mb_y < mb_rows; ++mb_y)
    {
        for (mb_x=0; mb_x < mb_cols; ++mb_x)
        {
            cuda_mc_block_8x8<<<1,1>>> (width, height, 
                    mb_x, mb_y, predY, reconsY, mbs[0]);
        }
    }

    /* Chroma */
    for (mb_y=0; mb_y < mb_rows/2; ++mb_y)
    {
        for (mb_x=0; mb_x < mb_cols/2; ++mb_x)
        {
            cuda_mc_block_8x8<<<1, 1>>> (width / 2, height / 2, 
                    mb_x, mb_y, predU, reconsU, mbs[1]);
            cuda_mc_block_8x8<<<1, 1>>> (width / 2, height / 2, 
                    mb_x, mb_y, predV, reconsV, mbs[2]);
        }
    }
}


