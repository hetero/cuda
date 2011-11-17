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

__device__ static void cuda_transpose_block(float *in_data, float *out_data, int col_mb)
{
    int row_mb = (threadIdx.y + col_mb) % 8;
    out_data[DCT_TH_X * col_mb + row_mb] = in_data[DCT_TH_X * row_mb + col_mb];
}

__device__ static void cuda_dct_1d(float *in_row, float *out_cell, int col_mb)
{
    // we can add constant-conflict-free
    int act = col_mb;
    float dct = in_row[act] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    dct += in_row[act = (act + 1) & 7] * dctlookup[act][col_mb];
    *out_cell = dct;
/*
    int act = col_mb;
    float dct = 0.0f;
    do
    {
        dct += in_row[act] * dctlookup[act][col_mb];
        act = (act + 1) % 8;
    }
    while (act != col_mb);
    *out_cell = dct;
*/
}

__device__ static void cuda_scale_block(float *in_data, float *out_data, int col_mb)
{
    float a1 = !(col_mb) ? ISQRT2 : 1.0f;
    float a2 = !(threadIdx.y) ? ISQRT2 : 1.0f;
    out_data[DCT_TH_X * threadIdx.y + col_mb] = in_data[DCT_TH_X * threadIdx.y + col_mb] * a1 * a2;
}

__device__ static void cuda_quantize_block(float *in_data, float *out_data, uint8_t id_quant, int col_mb)
{
    // better const memory accesing
    uint8_t u = zigzag_U[8 * threadIdx.y + col_mb];
    uint8_t v = zigzag_V[8 * threadIdx.y + col_mb];
    out_data[DCT_TH_X * threadIdx.y + col_mb] =
        rintf((in_data[DCT_TH_X * v + u] / 4.0f) / quanttbl[id_quant][8 * threadIdx.y + col_mb]);
}

__device__ static void cuda_dct_quant_block_8x8(float *mb, float *mb2, int16_t *out_data, uint8_t id_quant)
{
    int first_col = (threadIdx.x >> 3) << 3;
    int first_col_row = DCT_TH_X * threadIdx.y + first_col;
    int col_mb = threadIdx.x & 7;
    cuda_dct_1d(mb + first_col_row, mb2 + DCT_TH_X * threadIdx.y + threadIdx.x, col_mb);
    __syncthreads();
    cuda_transpose_block(mb2 + first_col, mb + first_col, col_mb);
    __syncthreads();
    cuda_dct_1d(mb + first_col_row, mb2 + DCT_TH_X * threadIdx.y + threadIdx.x, col_mb);
    __syncthreads();
    cuda_transpose_block(mb2 + first_col, mb + first_col, col_mb);
    __syncthreads();
    cuda_scale_block(mb + first_col, mb2 + first_col, col_mb);
    __syncthreads();
    cuda_quantize_block(mb2 + first_col, mb + first_col, id_quant, col_mb);
    __syncthreads();
    out_data[8 * first_col + 8 * threadIdx.y + col_mb] = mb[DCT_TH_X * threadIdx.y + threadIdx.x];
}

__global__ static void k_dct_quant_block_8x8(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height, int16_t *out_data, uint8_t id_quant)
{
    __shared__ float mb[DCT_BL_SIZE], mb2[DCT_BL_SIZE];
    int first_col_block = DCT_TH_X * blockIdx.x;
    if (first_col_block + threadIdx.x < width)
    {
        int first_row_block = 8 * width * blockIdx.y;
	int idxIn = (first_row_block + first_col_block) + (width * threadIdx.y + threadIdx.x);
        mb[DCT_TH_X * threadIdx.y + threadIdx.x] = (int16_t)in_data[idxIn] - prediction[idxIn];
        // we can assume that one row is done in the same time
        // because it is done by the same half-warp 
        //__syncthreads();
        cuda_dct_quant_block_8x8(
            mb,
            mb2,
            out_data + (first_row_block + DCT_BL_SIZE * blockIdx.x),
            id_quant);
    }
}

__host__ void cuda_dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t id_quant)
{
    size_t size = width * height;
    uint8_t *d_in_data, *d_prediction;
    int16_t *d_out_data;
    cudaMalloc(&d_in_data, size);
    cudaMalloc(&d_prediction, size);
    cudaMalloc(&d_out_data, size * sizeof(int16_t));
    cudaMemcpy(d_in_data, in_data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prediction, prediction, size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(DCT_TH_X, DCT_TH_Y);
    dim3 blocksPerGrid((width + DCT_TH_X - 1) / DCT_TH_X, height / 8);
    k_dct_quant_block_8x8<<<blocksPerGrid, threadsPerBlock>>>(d_in_data, d_prediction, width, height, d_out_data, id_quant);
    cudaMemcpy(out_data, d_out_data, size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_in_data);
    cudaFree(d_prediction);
    cudaFree(d_out_data);
}

