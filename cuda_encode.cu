#include "c63.h"
#include "cuda_encode.h"
#include "cuda_me.h"
#include "cuda_dct.h"
    
void cuda_init_c63_encode(int width, int height,
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbs[3])
{
    int ypw = 16 * ((width + 15) / 16);
    int yph = 16 * ((height + 15) / 16);
    int uvpw = 8 * ((width/2 + 7) / 8);
    int uvph = 8 * ((height/2 + 7) / 8);
    cudaMalloc(&origY, ypw * yph);
    cudaMalloc(&origU, uvpw * uvph);
    cudaMalloc(&origV, uvpw * uvph);
    cudaMalloc(&reconsY, ypw * yph);
    cudaMalloc(&reconsU, uvpw * uvph);
    cudaMalloc(&reconsV, uvpw * uvph);
    cudaMalloc(&predY, ypw * yph);
    cudaMalloc(&predU, uvpw * uvph);
    cudaMalloc(&predV, uvpw * uvph);
    cudaMalloc(&residY, ypw * yph * sizeof(int16_t));
    cudaMalloc(&residU, uvpw * uvph * sizeof(int16_t));
    cudaMalloc(&residV, uvpw * uvph * sizeof(int16_t));

    cudaMalloc(&mbs[0], ypw * yph / 64 * sizeof(struct macroblock));
    cudaMalloc(&mbs[1], uvpw * uvph / 64 * sizeof(struct macroblock));
    cudaMalloc(&mbs[2], uvpw * uvph / 64 * sizeof(struct macroblock));
}


void cuda_free_c63_encode(
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbs[3])
{
    cudaFree(origY);
    cudaFree(origU);
    cudaFree(origV);
    cudaFree(reconsY);
    cudaFree(reconsU);
    cudaFree(reconsV);
    cudaFree(predY);
    cudaFree(predU);
    cudaFree(predV);
    cudaFree(residY);
    cudaFree(residU);
    cudaFree(residV);
    cudaFree(mbs[0]);
    cudaFree(mbs[1]);
    cudaFree(mbs[2]);
}

void cuda_copy_image(int width, int height, yuv_t *image,
        uint8_t *origY, uint8_t *origU, uint8_t *origV)
{
    cudaMemcpy(origY, image->Y, width * height, 
            cudaMemcpyHostToDevice);
    cudaMemcpy(origU, image->U, width * height / 4, 
            cudaMemcpyHostToDevice);
    cudaMemcpy(origV, image->V, width * height / 4, 
            cudaMemcpyHostToDevice);
}
void cuda_next_frame(int width, int height,
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbs[3]) 
{
    int ypw = 16 * ((width + 15) / 16);
    int yph = 16 * ((height + 15) / 16);
    int uvpw = 8 * ((width/2 + 7) / 8);
    int uvph = 8 * ((height/2 + 7) / 8);
    cudaMemset(predY, 0x80, ypw * yph);
    cudaMemset(predU, 0x80, uvpw * uvph);
    cudaMemset(predV, 0x80, uvpw * uvph);
    cudaMemset(residY, 0x80, ypw * yph * sizeof(int16_t));
    cudaMemset(residU, 0x80, uvpw * uvph * sizeof(int16_t));
    cudaMemset(residV, 0x80, uvpw * uvph * sizeof(int16_t));
    cudaMemset(mbs[0], 0, ypw * yph / 64
            * sizeof(struct macroblock));
    cudaMemset(mbs[1], 0, uvpw * uvph / 64 
            * sizeof(struct macroblock));
    cudaMemset(mbs[2], 0, uvpw * uvph / 64 
            * sizeof(struct macroblock));
}

void cuda_c63_encode_image(int keyframe, int width, int height, 
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbs[3])
{
    int ypw = 16 * ((width + 15) / 16);
    int yph = 16 * ((height + 15) / 16);
    int uvpw = 8 * ((width/2 + 7) / 8);
    int uvph = 8 * ((height/2 + 7) / 8);
    /* Advance to next frame */
    cuda_next_frame(width, height,
            origY, origU, origV, reconsY, reconsU, reconsV,
            predY, predU, predV, residY, residU, residV,
            mbs); 

    if (!keyframe)
    {   
        cuda_c63_motion_estimate(ypw, yph,
            origY, origU, origV, reconsY, reconsU, reconsV,
            mbs);
        cuda_c63_motion_compensate(ypw, yph,
                reconsY, reconsU, reconsV, predY, predU, predV,
                mbs);
    }  
    
/*
    cuda_dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[0], cm->padh[0], cm->curframe->residuals->Ydct, 0, dct_in_data_y, dct_prediction_y, dct_out_data_y);
    cuda_dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[1], cm->padh[1], cm->curframe->residuals->Udct, 1, dct_in_data_uv, dct_prediction_uv, dct_out_data_uv);
    cuda_dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[2], cm->padh[2], cm->curframe->residuals->Vdct, 1, dct_in_data_uv, dct_prediction_uv, dct_out_data_uv);

    cuda_dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, 0, idct_in_data_y, idct_prediction_y, idct_out_data_y);
    cuda_dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, 1, idct_in_data_uv, idct_prediction_uv, idct_out_data_uv);
    cuda_dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, 1, idct_in_data_uv, idct_prediction_uv, idct_out_data_uv);
*/
    
    //write_frame(cm);

    //++cm->framenum;
    //++cm->frames_since_keyframe;
}

