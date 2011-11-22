#include "c63.h"
#include "cuda_encode.h"
#include "cuda_me.h"
#include "cuda_dct.h"
#include "cuda_idct.h"
//#include "cuPrintf.cu"
    
void cuda_init_c63_encode(int width, int height,
        uint8_t **origY, uint8_t **origU, uint8_t **origV,
        uint8_t **reconsY, uint8_t **reconsU, uint8_t **reconsV,
        uint8_t **predY, uint8_t **predU, uint8_t **predV,
        int16_t **residY, int16_t **residU, int16_t **residV,
        struct macroblock **mbsY, struct macroblock **mbsU,
        struct macroblock **mbsV)
{
//    cudaPrintfInit();

    int ypw = 16 * ((width + 15) / 16);
    int yph = 16 * ((height + 15) / 16);
    int uvpw = 8 * ((width/2 + 7) / 8);
    int uvph = 8 * ((height/2 + 7) / 8);
    cudaMalloc(origY, ypw * yph);
    cudaMalloc(origU, uvpw * uvph);
    cudaMalloc(origV, uvpw * uvph);
    cudaMalloc(reconsY, ypw * yph);
    cudaMalloc(reconsU, uvpw * uvph);
    cudaMalloc(reconsV, uvpw * uvph);
    cudaMalloc(predY, ypw * yph);
    cudaMalloc(predU, uvpw * uvph);
    cudaMalloc(predV, uvpw * uvph);
    cudaMalloc(residY, ypw * yph * sizeof(int16_t));
    cudaMalloc(residU, uvpw * uvph * sizeof(int16_t));
    cudaMalloc(residV, uvpw * uvph * sizeof(int16_t));

    cudaMalloc(mbsY, ypw * yph / 64 * sizeof(struct macroblock));
    cudaMalloc(mbsU, uvpw * uvph / 64 * sizeof(struct macroblock));
    cudaMalloc(mbsV, uvpw * uvph / 64 * sizeof(struct macroblock));
}


void cuda_free_c63_encode(
        uint8_t **origY, uint8_t **origU, uint8_t **origV,
        uint8_t **reconsY, uint8_t **reconsU, uint8_t **reconsV,
        uint8_t **predY, uint8_t **predU, uint8_t **predV,
        int16_t **residY, int16_t **residU, int16_t **residV,
        struct macroblock **mbsY, struct macroblock **mbsU,
        struct macroblock **mbsV)
{
    cudaFree(*origY);
    cudaFree(*origU);
    cudaFree(*origV);
    cudaFree(*reconsY);
    cudaFree(*reconsU);
    cudaFree(*reconsV);
    cudaFree(*predY);
    cudaFree(*predU);
    cudaFree(*predV);
    cudaFree(*residY);
    cudaFree(*residU);
    cudaFree(*residV);
    cudaFree(*mbsY);
    cudaFree(*mbsU);
    cudaFree(*mbsV);

//    cudaPrintfDisplay();
//    cudaPrintfEnd();
}

void cuda_copy_image(int width, int height, yuv_t *image,
        uint8_t *origY, uint8_t *origU, uint8_t *origV)
{
    /*
    uint8_t t[7] = {0,1,2,3,4,5,6};
    uint8_t s[7];
    
    cudaMalloc(&origY, 7);
    cudaMemcpy(origY, t, 7, cudaMemcpyHostToDevice);
    cudaMemcpy(s, origY, 7, cudaMemcpyDeviceToHost);
    printf("7: (%d, %d, %d, %d, %d, %d %d)\n", 
            s[0], s[1], s[2], s[3], s[4], s[5], s[6]);

*/

    cudaMemcpy(origY, image->Y, width * height, 
            cudaMemcpyHostToDevice);
    cudaMemcpy(origU, image->U, width * height / 4, 
            cudaMemcpyHostToDevice);
    cudaMemcpy(origV, image->V, width * height / 4, 
            cudaMemcpyHostToDevice);
   /* 
    uint8_t tab[352*288];
    int pos = 139 * width + 171;
    printf("(copy in: 171, 139-141) = %d, %d, %d\n",image->Y[pos],
            image->Y[pos+1], image->Y[pos+2]);

    cudaMemcpy(tab, origY, width * height, 
            cudaMemcpyHostToDevice);
    printf("(copy out: 171, 139-141) = %d, %d, %d\n",tab[pos],
            tab[pos+1], tab[pos+2]);
            */
}

void cuda_next_frame(int width, int height,
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbsY, struct macroblock *mbsU,
        struct macroblock *mbsV)
{
    int ypw = 16 * ((width + 15) / 16);
    int yph = 16 * ((height + 15) / 16);
    int uvpw = 8 * ((width/2 + 7) / 8);
    int uvph = 8 * ((height/2 + 7) / 8);
    cudaMemset(predY, 0x80, ypw * yph);
    cudaMemset(predU, 0x80, uvpw * uvph);
    cudaMemset(predV, 0x80, uvpw * uvph);
/*    cudaMemset(residY, 0x80 ypw * yph * sizeof(int16_t));
    cudaMemset(residU, 0x80, uvpw * uvph * sizeof(int16_t));
    cudaMemset(residV, 0x80, uvpw * uvph * sizeof(int16_t));*/
    cudaMemset(mbsY, 0, ypw * yph / 64
            * sizeof(struct macroblock));
    cudaMemset(mbsU, 0, uvpw * uvph / 64 
            * sizeof(struct macroblock));
    cudaMemset(mbsV, 0, uvpw * uvph / 64 
            * sizeof(struct macroblock));
}

void cuda_c63_encode_image(struct c63_common *cm, int width, int height, 
        uint8_t *origY, uint8_t *origU, uint8_t *origV,
        uint8_t *reconsY, uint8_t *reconsU, uint8_t *reconsV,
        uint8_t *predY, uint8_t *predU, uint8_t *predV,
        int16_t *residY, int16_t *residU, int16_t *residV,
        struct macroblock *mbsY, struct macroblock *mbsU,
        struct macroblock *mbsV)
{
    int ypw = 16 * ((width + 15) / 16);
    int yph = 16 * ((height + 15) / 16);
    int uvpw = 8 * ((width/2 + 7) / 8);
    int uvph = 8 * ((height/2 + 7) / 8);
/*    
    //DEBUG
    uint8_t tab[352*288];
    cudaMemcpy(tab, reconsY, width * height, 
            cudaMemcpyDeviceToHost);
    int pos = 139 * width + 171;
    printf("przed next Y: (171, 139-141) = %d, %d, %d\n",tab[pos],
            tab[pos+1], tab[pos+2]);
*/

    /* Advance to next frame */
    cuda_next_frame(width, height,
            origY, origU, origV, reconsY, reconsU, reconsV,
            predY, predU, predV, residY, residU, residV,
            mbsY, mbsU, mbsV);
  /*  
    //DEBUG
    cudaMemcpy(tab, reconsY, width * height, 
            cudaMemcpyDeviceToHost);
    printf("po next (Y: 171, 139-141) = %d, %d, %d\n",tab[pos],
            tab[pos+1], tab[pos+2]);
*/
    
       if (!cm->curframe->keyframe)
    {   
        cuda_c63_motion_estimate(ypw, yph,
                origY, origU, origV, reconsY, reconsU, reconsV,
                mbsY, mbsU, mbsV);
        cuda_c63_motion_compensate(ypw, yph,
                reconsY, reconsU, reconsV, predY, predU, predV,
                mbsY, mbsU, mbsV);
    }  
    
    cuda_dct_quantize(ypw, yph, 0, origY, predY, residY);
    cuda_dct_quantize(uvpw, uvph, 1, origU, predU, residU);
    cuda_dct_quantize(uvpw, uvph, 1, origV, predV, residV);

    cuda_dequantize_idct(ypw, yph, 0, residY, predY, reconsY);
    cuda_dequantize_idct(uvpw, uvph, 1, residU, predU, reconsU);
    cuda_dequantize_idct(uvpw, uvph, 1, residV, predV, reconsV);
    
    cudaMemcpy(cm->curframe->residuals->Ydct, residY, 
            ypw * yph * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cm->curframe->residuals->Udct, residU, 
            uvpw * uvph * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cm->curframe->residuals->Vdct, residV, 
            uvpw * uvph * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(cm->curframe->mbs[0], mbsY, ypw * yph / 64 *
            sizeof(struct macroblock), cudaMemcpyDeviceToHost);
    cudaMemcpy(cm->curframe->mbs[1], mbsU, uvpw * uvph / 64 *
            sizeof(struct macroblock), cudaMemcpyDeviceToHost);
    cudaMemcpy(cm->curframe->mbs[2], mbsV, uvpw * uvph / 64 *
            sizeof(struct macroblock), cudaMemcpyDeviceToHost);
/*
    //DEBUG
    cudaMemcpy(tab, origY, width * height, 
            cudaMemcpyDeviceToHost);
    printf("(przed write Y: 171, 139-141) = %d, %d, %d\n",tab[pos],
            tab[pos+1], tab[pos+2]);
    //END DEBUG*/

    write_frame(cm);
    
}

