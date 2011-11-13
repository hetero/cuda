#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

#include "c63.h"
#include "cuda_me.h"

__device__ void cuda_sad_block_8x8(uint8_t *block1, uint8_t *block2,
        int stride, int *result)
{
    *result = 0;
 
    int u,v;
    for (v=0; v<8; ++v)
        for (u=0; u<8; ++u)
            *result += abs(block2[v*stride+u] - block1[v*stride+u]);
}

__global__ void k_me_block_8x8(uint8_t *orig, uint8_t *ref, mv_out_t *mv_out, int w, int h)
{
    #define SAD_SIZE 32
    __shared__ int sad[SAD_SIZE * SAD_SIZE];

    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    int block_nr = mb_y * w / 8 + mb_x;
    
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


    int x = left + 2 * threadIdx.x;
    int y = top + threadIdx.y;

    //cuPrintf("(x,y) = (%d, %d)\n", x, y);
    int mx = mb_x * 8;
    int my = mb_y * 8;

    if (y<bottom && x<right)
    {
        cuda_sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, 
                &sad[(y-top) * SAD_SIZE + (x-left)]);
        x++;
        cuda_sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, 
                &sad[(y-top) * SAD_SIZE + (x-left)]);
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int best_sad = INT_MAX;
        int best_x, best_y;

        for (x = left; x < right; ++x) {
            for (y = top; y < bottom; ++y) {
//            printf("(%4d,%4d) - %d\n", x, y, sad);
                int sad_temp = sad[(y-top) * SAD_SIZE + (x-left)];
                if (sad_temp < best_sad)
                {
                    best_x = x - mx;
                    best_y = y - my;
                    best_sad = sad_temp;
                }
            }
        }
        mv_out[block_nr].sad = best_sad;
        mv_out[block_nr].mv_x = best_x;
        mv_out[block_nr].mv_y = best_y;
    }
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
            int sad = mv_out_host[block_nr].sad;
            int mv_x = mv_out_host[block_nr].mv_x;
            int mv_y = mv_out_host[block_nr].mv_y;
            struct macroblock *mb = &cm->curframe->mbs[cc][block_nr];
            if (sad < 512) {
                mb->use_mv = 1;
                mb->mv_x = mv_x;
                mb->mv_y = mv_y;
            }
            else {
                mb->use_mv = 0;
            }
                //printf("(%d,%d): MV (%d, %d) with SAD %d\n", mb_x, mb_y, mb->mv_x, mb->mv_y, sad);
        }
    }
}

void cuda_c63_motion_estimate(struct c63_common *cm) {
    for (int cc = 0; cc <= 2; cc++) {
        cuda_me_cc(cm, cc);
    }
}
