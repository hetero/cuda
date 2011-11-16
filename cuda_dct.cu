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

__device__ static void cuda_transpose_block(float *in_data, float *out_data, int col_mb, int row_mb)
{
    out_data[8 * col_mb + row_mb] = in_data[8 * row_mb + col_mb];
    /*
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        {
            out_data[i * 8 + j] = in_data[j * 8 + i];
        }
    */
}

__device__ static void cuda_dct_1d(float *in_data, float *out_data, int col_mb, int nr_mb)
{
    // tricks to better accesing const and shared memory
    /*
    int first = (col_mb + nr_mb) % 8;
    int act = first;
    float dct = 0.0f;
    do
    {
        dct += in_data[act] * dctlookup[act][col_mb];
        act = (act + 1) % 8;
    }
    while (act != (col_mb + nr_mb) % 8);
    out_data[col_mb] = dct;
    */
    
    float dct = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        dct += in_data[i] * dctlookup[i][col_mb];
    }
    out_data[col_mb] = dct;
     
    /*
    for (int j = 0; j < 8; ++j)
    {
        float dct = 0;

        for (int i = 0; i < 8; ++i)
        {
            dct += in_data[i] * dctlookup[i][j];
        }

        out_data[j] = dct;
    }
    */
}

__device__ static void cuda_scale_block(float *in_data, float *out_data, int col_mb, int row_mb)
{
    float a1 = !col_mb ? ISQRT2 : 1.0f;
    float a2 = !row_mb ? ISQRT2 : 1.0f;
    out_data[8 * col_mb + row_mb] = in_data[8 * col_mb + row_mb] * a1 * a2;
    /*
    for (int v = 0; v < 8; ++v)
    {
        for (int u = 0; u < 8; ++u)
        {
            float a1 = !u ? ISQRT2 : 1.0f;
            float a2 = !v ? ISQRT2 : 1.0f;

            out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
        }
    }
    */
}

__device__ static void cuda_quantize_block(float *in_data, float *out_data, uint8_t id_quant, int col_mb, int row_mb, int nr_mb)
{
    // better const memory accesing
    /*
       int zigzag = (8 * row_mb + col_mb + 8 * nr_mb) % 64;
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];
    float dct = in_data[8 * v + u];
    out_data[zigzag] = rintf((dct / 4.0f) / quanttbl[id_quant][zigzag]);
    */
    int zigzag = 8 * row_mb + col_mb;
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];
    out_data[zigzag] = rintf((in_data[8 * v + u] / 4.0f) / quanttbl[id_quant][zigzag]);
    
    /*
    for (int zigzag = 0; zigzag < 64; ++zigzag)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[v * 8 + u];

        out_data[zigzag] = rintf((dct / 4.0f) / quanttbl[id_quant][zigzag]);
    }
    */
}

__device__ static void cuda_dct_quant_block_8x8(float *mb, float *mb2, int16_t *out_data, uint8_t id_quant, int col_mb, int row_mb, int nr_mb)
{
    cuda_dct_1d(mb + 8 * row_mb, mb2 + 8 * row_mb, col_mb, nr_mb);
    //__syncthreads();
    // row <-> half-warp
    cuda_transpose_block(mb2, mb, col_mb, row_mb);
    __syncthreads();
    cuda_dct_1d(mb + 8 * row_mb, mb2 + 8 * row_mb, col_mb, nr_mb);
    //__syncthreads();
    // row <-> half-warp
    cuda_transpose_block(mb2, mb, col_mb, row_mb);
    //__syncthreads();
    // col <-> half-warp
    cuda_scale_block(mb, mb2, col_mb, row_mb);
    __syncthreads();
    cuda_quantize_block(mb2, mb, id_quant, col_mb, row_mb, nr_mb);
    __syncthreads();
    //out_data[8 * row_mb + col_mb] = mb[8 * row_mb + col_mb];
    /*
    float mb[8 * 8];

    // mb = mb2 * dctlookup
    for (int v = 0; v < 8; ++v)
    {
        cuda_dct_1d(mb2 + v * 8, mb + v * 8);
    }
    // mb2 = mb^T
    cuda_transpose_block(mb, mb2);

    // mb = mb2 * dctlookup
    for (int v = 0; v < 8; ++v)
    {
        cuda_dct_1d(mb2 + v * 8, mb + v * 8);
    }

    // mb2 = mb^T
    cuda_transpose_block(mb, mb2);
    // first row and col multiplied by ISQRT2
    cuda_scale_block(mb2, mb);

    cuda_quantize_block(mb, mb2, id_quant);

    for (int i = 0; i < 64; ++i)
        out_data[i] = mb2[i];
        */
}

__global__ static void k_dct_quant_block_8x8(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height, int16_t *out_data, uint8_t id_quant)
{

    // this way to avoid banks conflict
    // x - nr of pixel in mb, y - nr of mb
    __shared__ float mb[8][64], mb2[8][64];
    int col_mb = threadIdx.x % 8;
    //int row_mb = threadIdx.y;
    int nr_mb = threadIdx.x / 8;
    int col_frame = blockIdx.x * 8 + nr_mb;
    //int row_frame = blockIdx.y;
    if (col_frame * 8 < width)
    {
        uint8_t *my_in_data = &in_data[8 * width * blockIdx.y + 8 * col_frame];
        uint8_t *my_prediction = &prediction[8 * width * blockIdx.y + 8 * col_frame];

        int16_t *my_out_data = &out_data[8 * width * blockIdx.y + 64 * col_frame];
        mb[nr_mb][8 * threadIdx.y + col_mb] = (int16_t)my_in_data[width * threadIdx.y + col_mb] - my_prediction[width * threadIdx.y + col_mb];
        // we can assume that one row is done in the same time
        // because it is done by the same half-warp 
        //__syncthreads();
        
        cuda_dct_quant_block_8x8(mb[nr_mb], mb2[nr_mb], my_out_data, id_quant, col_mb, threadIdx.y, nr_mb);
    /*    
        // PROBUJEMY
        cuda_dct_1d(mb[nr_mb] + 8 * threadIdx.y, mb2[nr_mb] + 8 * threadIdx.y, col_mb, nr_mb);
        //__syncthreads();
        // row <-> half-warp
        cuda_transpose_block(mb2[nr_mb], mb[nr_mb], col_mb, threadIdx.y);
        __syncthreads();
        cuda_dct_1d(mb[nr_mb] + 8 * threadIdx.y, mb2[nr_mb] + 8 * threadIdx.y, col_mb, nr_mb);
        //__syncthreads();
        // row <-> half-warp
        cuda_transpose_block(mb2[nr_mb], mb[nr_mb], col_mb, threadIdx.y);
        //__syncthreads();
        // col <-> half-warp
        cuda_scale_block(mb[nr_mb], mb2[nr_mb], col_mb, threadIdx.y);
        __syncthreads();
        cuda_quantize_block(mb2[nr_mb], mb[nr_mb], id_quant, col_mb, threadIdx.y, nr_mb);
        __syncthreads();
        //out_data[8 * threadIdx.y + col_mb] = mb[8 * threadIdx.y + col_mb];
       */ 
        // ---------
        
        // out is in mb
        //out_data[8 * row_mb + col_mb] = mb[8 * row_mb + col_mb];
        my_out_data[8 * threadIdx.y + col_mb] = mb[nr_mb][8 * threadIdx.y + col_mb];
    
    }
    //if (8 * threadIdx.y < width - blockIdx.x)
    /*
    int16_t *my_out_data = &out_data[8 * width * blockIdx.y + 64 * 8 * blockIdx.x];
    if (64 * threadIdx.y < width * 8 - blockIdx.x * 8)
    {
        my_out_data[64 * threadIdx.y + threadIdx.x] = mb[threadIdx.y][threadIdx.x];
    }
    */

    /*
    int row = blockIdx.x;
    int col = threadIdx.x;
    float block[8 * 8];
    int16_t *my_out_data;
    uint8_t *my_in_data, *my_prediction;
    my_in_data = &in_data[8 * width * row + 8 * col];
    my_prediction = &prediction[8 * width * row + 8 * col];
    my_out_data = out_data + (8 * width * row + 64 * col);

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            block[8 * i + j] = ((int16_t)my_in_data[width * i + j] - my_prediction[width * i + j]);

    cuda_dct_quant_block_8x8(block, my_out_data, id_quant);
    */
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
    dim3 threadsPerBlock(64, 8);
    dim3 blocksPerGrid((width + 63) / 64, height / 8);
    k_dct_quant_block_8x8<<<blocksPerGrid, threadsPerBlock>>>(d_in_data, d_prediction, width, height, d_out_data, id_quant);
    cudaMemcpy(out_data, d_out_data, size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_in_data);
    cudaFree(d_prediction);
    cudaFree(d_out_data);
}

