#include <list>
#include <pthread.h>

#include "c63.h"
#include "cuda_encode.h"
#include "cuda_me.h"
#include "cuda_dct.h"
#include "cuda_idct.h"
//#include "cuPrintf.cu"

using std::list;
extern list<pthread_t> th_id_list;
extern struct entropy_ctx write_entropy;
extern pthread_mutex_t mutex;
struct c63_common tmp_cm;

void *thread_write_frame(void *tmp_cm)
{
    struct c63_common *cm = (struct c63_common *)tmp_cm;
    cm->e_ctx = write_entropy;
    pthread_mutex_lock(&mutex);
    write_frame(cm);
    // small hack to remember entropy_ctx
    write_entropy = cm->e_ctx;
    destroy_cm_write(cm);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}


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

    /* Advance to next frame */
    cuda_next_frame(width, height,
            origY, origU, origV, reconsY, reconsU, reconsV,
            predY, predU, predV, residY, residU, residV,
            mbsY, mbsU, mbsV);
    
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
    
    pthread_t t;
    tmp_cm = *cm;
    pthread_create(&t, NULL, thread_write_frame, (void*)&tmp_cm);
    th_id_list.push_back(t);
    cuda_fake_cm_init(cm); 
    
    //write_frame(cm);
}


