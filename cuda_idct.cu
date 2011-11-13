#include <stdint.h>

#define ISQRT2 0.70710678118654f

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

__device__ static void cuda_transpose_block(float *in_data, float *out_data)
{
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        {
            out_data[i * 8 + j] = in_data[j * 8 + i];
        }
}

__device__ static void cuda_scale_block(float *in_data, float *out_data)
{
    for (int v = 0; v < 8; ++v)
    {
        for (int u = 0; u < 8; ++u)
        {
            float a1 = !u ? ISQRT2 : 1.0f;
            float a2 = !v ? ISQRT2 : 1.0f;

            /* Scale according to normalizing function */
            out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
        }
    }
}

__device__ static void cuda_idct_1d(float *in_data, float *out_data)
{
    for (int j = 0; j < 8; ++j)
    {
        float idct = 0;

        for (int i = 0; i < 8; ++i)
        {
            idct += in_data[i] * dctlookup[j][i];
        }

        out_data[j] = idct;
    }
}

__device__ static void cuda_dequantize_block(float *in_data, float *out_data, uint8_t id_quant)
{
    for (int zigzag = 0; zigzag < 64; ++zigzag)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[zigzag];

        /* Zig-zag and de-quantize */
        out_data[v * 8 + u] = rintf((dct * quanttbl[id_quant][zigzag]) / 4.0f);
    }
}

__device__ static void cuda_dequant_idct_block_8x8(int16_t *in_data, float *mb, uint8_t id_quant)
{
    float mb2[8 * 8];

    for (int i = 0; i < 64; ++i)
        mb[i] = in_data[i];

    cuda_dequantize_block(mb, mb2, id_quant);

    cuda_scale_block(mb2, mb);

    for (int v = 0; v < 8; ++v)
    {
        cuda_idct_1d(mb + v * 8, mb2 + v * 8);
    }

    cuda_transpose_block(mb2, mb);

    for (int v = 0; v < 8; ++v)
    {
        cuda_idct_1d(mb + v * 8, mb2 + v * 8);
    }

    cuda_transpose_block(mb2, mb);
}

__global__ static void k_dequant_idct_block_8x8(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height, uint8_t *out_data, uint8_t id_quant)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    float block[8 * 8];

    int16_t *my_in_data = &in_data[8 * width * row + 64 * col];
    uint8_t *my_prediction = &prediction[8 * width * row + 8 * col];
    uint8_t *my_out_data = &out_data[8 * width * row + 8 * col];
    /* Perform the dequantization and iDCT */
    cuda_dequant_idct_block_8x8(my_in_data, block, id_quant);

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        {
            /* Add prediction block. Note: DCT is not precise - Clamp to legal values */
            int16_t tmp = (int16_t)block[i * 8 + j] + (int16_t)my_prediction[i * width + j];
            if (tmp < 0)
                tmp = 0;
            else if (tmp > 255)
                tmp = 255;

            my_out_data[i * width + j] = tmp;
        }
}

__host__ void cuda_dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height, uint8_t *out_data, uint8_t id_quant)
{
    size_t size = width * height;
    int16_t *d_in_data;
    uint8_t *d_out_data, *d_prediction;
    cudaMalloc(&d_in_data, size * sizeof(int16_t));
    cudaMalloc(&d_prediction, size);
    cudaMalloc(&d_out_data, size);
    cudaMemcpy(d_in_data, in_data, size * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prediction, prediction, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = width / 8;
    int blocksPerGrid = height / 8;
    k_dequant_idct_block_8x8<<<blocksPerGrid, threadsPerBlock>>>(d_in_data, d_prediction, width, height, d_out_data, id_quant);
    cudaMemcpy(out_data, d_out_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in_data);
    cudaFree(d_prediction);
    cudaFree(d_out_data);
}

