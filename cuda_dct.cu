#include <stdint.h>
#include "cuda_common.h"

__constant__ static uint8_t quanttbl[2][64] =
{
    {
        6, 4, 4, 5, 4, 4, 6, 5,
        5, 5, 7, 6, 6, 7, 9, 16, 
        10, 9, 8, 8, 9, 19, 14, 14, 
        11, 16, 23, 20, 24, 12, 22, 20, 
        22, 22, 25, 28, 36, 31, 25, 27, 
        34, 27, 22, 22, 32, 43, 32, 34, 
        38, 39, 41, 41, 41, 24, 30, 45, 
        48, 44, 40, 48, 36, 40, 41, 39
    },
    {
        6, 7, 7, 9, 8, 9, 18, 10, 
        10, 18, 39, 26, 22, 26, 39, 39, 
        39, 39, 39, 39, 39, 39, 39, 39, 
        39, 39, 39, 39, 39, 39, 39, 39, 
        39, 39, 39, 39, 39, 39, 39, 39, 
        39, 39, 39, 39, 39, 39, 39, 39, 
        39, 39, 39, 39, 39, 39, 39, 39, 
        39, 39, 39, 39, 39, 39, 39, 39
    }
};

__constant__ static uint8_t zigzag_U[64] =
{
    0,
    1, 0,
    0, 1, 2,
    3, 2, 1, 0,
    0, 1, 2, 3, 4,
    5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6,
    7, 6, 5, 4, 3, 2, 1, 0,
    1, 2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3, 2,
    3, 4, 5, 6, 7,
    7, 6, 5, 4,
    5, 6, 7,
    7, 6,
    7,
};

__constant__ static uint8_t zigzag_V[64] =
{
    0,
    0, 1,
    2, 1, 0,
    0, 1, 2, 3,
    4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5,
    6, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3, 2, 1,
    2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3,
    4, 5, 6, 7,
    7, 6, 5,
    6, 7,
    7,
};

__constant__ static float dctlookup[8][8] = {
    {1.000000f, 0.980785f, 0.923880f, 0.831470f, 0.707107f, 0.555570f, 0.382683f, 0.195090f, },
    {1.000000f, 0.831470f, 0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
    {1.000000f, 0.555570f, -0.382683f, -0.980785f, -0.707107f, 0.195090f, 0.923880f, 0.831470f, },
    {1.000000f, 0.195090f, -0.923880f, -0.555570f, 0.707107f, 0.831470f, -0.382683f, -0.980785f, },
    {1.000000f, -0.195090f, -0.923880f, 0.555570f, 0.707107f, -0.831470f, -0.382683f, 0.980785f, },
    {1.000000f, -0.555570f, -0.382683f, 0.980785f, -0.707107f, -0.195090f, 0.923880f, -0.831470f, },
    {1.000000f, -0.831470f, 0.382683f, 0.195090f, -0.707107f, 0.980785f, -0.923880f, 0.555570f, },
    {1.000000f, -0.980785f, 0.923880f, -0.831470f, 0.707107f, -0.555570f, 0.382683f, -0.195090f, },
};
/*
__constant__ static float isqrt[64] = {
    0.5f, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};
*/
__device__ static void cuda_dct_1d(float *in_row, float *out_cell, const int &col_mb)
{
#define dct_col (threadIdx.y)
    
    /*
    float dct = in_row[col_mb] * dctlookup[col_mb][dct_col];
    dct += in_row[(col_mb + 1) & 7] * dctlookup[(col_mb + 1) & 7][dct_col];
    dct += in_row[(col_mb + 2) & 7] * dctlookup[(col_mb + 2) & 7][dct_col];
    dct += in_row[(col_mb + 3) & 7] * dctlookup[(col_mb + 3) & 7][dct_col];
    dct += in_row[(col_mb + 4) & 7] * dctlookup[(col_mb + 4) & 7][dct_col];
    dct += in_row[(col_mb + 5) & 7] * dctlookup[(col_mb + 5) & 7][dct_col];
    dct += in_row[(col_mb + 6) & 7] * dctlookup[(col_mb + 6) & 7][dct_col];
    dct += in_row[(col_mb + 7) & 7] * dctlookup[(col_mb + 7) & 7][dct_col];
    */
    /*
    float dct = in_row[act] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][dct_col];
    */
    
    // many shared conflicts
    
    float dct = in_row[7] * dctlookup[7][dct_col];
    dct += in_row[0] * dctlookup[0][dct_col];
    dct += in_row[1] * dctlookup[1][dct_col];
    dct += in_row[2] * dctlookup[2][dct_col];
    dct += in_row[3] * dctlookup[3][dct_col];
    dct += in_row[4] * dctlookup[4][dct_col];
    dct += in_row[5] * dctlookup[5][dct_col];
    dct += in_row[6] * dctlookup[6][dct_col];
    
    *out_cell = dct;
}

__device__ static void cuda_scale_block(float *in_data, float *out_data, const int &col_mb)
{
#define row_mb (threadIdx.y)
    float a1 = (!col_mb ? ISQRT2 : 1.0f);
    float a2 = (!row_mb ? ISQRT2 : 1.0f);
    int idx = DCT_TH_X * threadIdx.y + col_mb;
    out_data[idx] = in_data[idx] * a1 * a2;
}

__device__ static void cuda_quantize_block(float *in_data, float *out_data,
        const uint8_t &id_quant, const int &col_mb)
{
    // TODO better const memory accesing
    int zigzag = 8 * threadIdx.y + col_mb;
    out_data[DCT_TH_X * threadIdx.y + col_mb] =
        rintf(
                (in_data[DCT_TH_X * zigzag_V[zigzag]
                 + zigzag_U[zigzag]] / 4.0f)
                / quanttbl[id_quant][zigzag]);
}

__device__ static void cuda_dct_quant_block_8x8(float (&mb)[DCT_BL_SIZE],
        float (&mb2)[DCT_BL_SIZE], int16_t *out_data,
        const uint8_t &id_quant, const int &block_pos)
{
    int col_mb = (threadIdx.x & 7);
    int first_col = ((threadIdx.x >> 3) << 3);
    int first_col_row = DCT_TH_X * col_mb + first_col;
    // change 2 to 1
    cuda_dct_1d(mb + first_col_row, mb2 + block_pos, col_mb);
    __syncthreads();
    cuda_dct_1d(mb2 + first_col_row, mb + block_pos, col_mb);
    __syncthreads();
    cuda_scale_block(mb + first_col, mb2 + first_col, col_mb);
    __syncthreads();
    cuda_quantize_block(mb2 + first_col, mb + first_col, id_quant, col_mb);
    __syncthreads();
    out_data[8 * first_col + 8 * threadIdx.y + col_mb] = mb[block_pos];
}

__global__ static void k_dct_quant_block_8x8(uint8_t *in_data, uint8_t *prediction, uint32_t width, int16_t *out_data, uint8_t id_quant)
{
    __shared__ float mb[DCT_BL_SIZE], mb2[DCT_BL_SIZE];
    int first_col_block = (DCT_TH_X * blockIdx.x);
    if (first_col_block + threadIdx.x < width)
    {
        int block_pos = DCT_TH_X * threadIdx.y + threadIdx.x;
        int first_row_block = (8 * width * blockIdx.y);
        int idxIn = ((first_row_block + first_col_block) + (width * threadIdx.y + threadIdx.x));
        mb[block_pos] = (int16_t)in_data[idxIn] - prediction[idxIn];
        // we can assume that one row is done in the same time
        // because it is done by the same half-warp 
        __syncthreads();
        cuda_dct_quant_block_8x8(
            mb,
            mb2,
            out_data + (first_row_block + DCT_BL_SIZE * blockIdx.x),
            id_quant,
            block_pos);
    }
}

__host__ void cuda_dct_quantize(uint32_t width, uint32_t height, 
        uint8_t id_quant,uint8_t *d_in_data, uint8_t *d_prediction,
        int16_t *d_out_data)
{
    dim3 threadsPerBlock(DCT_TH_X, DCT_TH_Y);
    dim3 blocksPerGrid((width + DCT_TH_X - 1) / DCT_TH_X, height / 8);
    k_dct_quant_block_8x8<<<blocksPerGrid, threadsPerBlock>>>(
            d_in_data, d_prediction, width, d_out_data, id_quant);
}

__host__ void cuda_dct_free(uint8_t *d_in_data, uint8_t *d_prediction, int16_t *d_out_data)
{
    cudaFree(d_in_data);
    cudaFree(d_prediction);
    cudaFree(d_out_data);
}
