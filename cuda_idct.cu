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

__device__ static void cuda_scale_block(float *in_data, float *out_data, const int &col_mb)
{
#define row_mb (threadIdx.y)
    float a1 = !col_mb ? ISQRT2 : 1.0f;
    float a2 = !row_mb ? ISQRT2 : 1.0f;
    int idx = DCT_TH_X * threadIdx.y + col_mb;
    out_data[idx] = in_data[idx] * a1 * a2;
}

__device__ static void cuda_idct_1d(float *in_row, float *out_cell, const int &col_mb)
{
#define dct_col (threadIdx.y)
    float idct = in_row[0] * dctlookup[dct_col][0];
    idct += in_row[1] * dctlookup[dct_col][1];
    idct += in_row[2] * dctlookup[dct_col][2];
    idct += in_row[3] * dctlookup[dct_col][3];
    idct += in_row[4] * dctlookup[dct_col][4];
    idct += in_row[5] * dctlookup[dct_col][5];
    idct += in_row[6] * dctlookup[dct_col][6];
    idct += in_row[7] * dctlookup[dct_col][7];
    
    *out_cell = idct;
}

__device__ static void cuda_dequantize_block(float *in_data, float *out_data, uint8_t id_quant, int col_mb)
{
    int zigzag = 8 * threadIdx.y + col_mb;
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];
    float dct = in_data[DCT_TH_X * threadIdx.y + col_mb];
    out_data[DCT_TH_X * v + u] = rintf((dct * quanttbl[id_quant][zigzag]) / 4.0f);
}

__device__ static void cuda_dequant_idct_block_8x8(
        float *mb, float *mb2, uint8_t id_quant,
        const int &col_mb, const int &block_pos)
{
    int first_col = (threadIdx.x >> 3) << 3;
    int first_col_row = DCT_TH_X * col_mb + first_col;
    cuda_dequantize_block(mb + first_col, mb2 + first_col, id_quant, col_mb);
    __syncthreads();
    cuda_scale_block(mb2 + first_col, mb + first_col, col_mb);
    __syncthreads();
    cuda_idct_1d(mb + first_col_row, mb2 + block_pos, col_mb);
    __syncthreads();
    cuda_idct_1d(mb2 + first_col_row, mb + block_pos, col_mb);
    __syncthreads();
}

__global__ static void k_dequant_idct_block_8x8(
        int16_t *in_data, uint8_t *prediction, uint32_t width,
        uint8_t *out_data, uint8_t id_quant)
{
    __shared__ float mb[DCT_BL_SIZE], mb2[DCT_BL_SIZE];
    int col_mb = threadIdx.x & 7;
    int nr_mb = threadIdx.x >> 3;
    int block_pos = DCT_TH_X * threadIdx.y + threadIdx.x;
    int idxIn = 8 * width * blockIdx.y + DCT_BL_SIZE * blockIdx.x + 8 * threadIdx.y + col_mb + 64 * nr_mb;
    mb[block_pos] = in_data[idxIn];
    cuda_dequant_idct_block_8x8(mb, mb2, id_quant, col_mb, block_pos);
    int idxPredOut = 8 * width * blockIdx.y + DCT_TH_X * blockIdx.x + width * threadIdx.y + threadIdx.x;
    int tmp = (int)mb[block_pos] + (int)prediction[idxPredOut];
    if (tmp < 0)
        tmp = 0;
    else if (tmp > 255)
        tmp = 255;
    out_data[idxPredOut] = tmp;
}

__host__ void cuda_dequantize_idct(uint32_t width, uint32_t height,
        uint8_t id_quant, int16_t *d_in_data, uint8_t *d_prediction,
        uint8_t *d_out_data)
{
    dim3 threadsPerBlock(DCT_TH_X, DCT_TH_Y);
    dim3 blocksPerGrid((width + DCT_TH_X - 1) / DCT_TH_X, height / 8);
    k_dequant_idct_block_8x8<<<blocksPerGrid, threadsPerBlock>>>(
            d_in_data, d_prediction, width, d_out_data, id_quant);
}
