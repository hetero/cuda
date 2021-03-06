#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>

#include "c63.h"
#include "tables.h"

void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h, int y,
			 uint8_t *out_data, uint8_t *quantization)
{
    int x;

    int16_t block[8*8];

    /* Perform the dequantization and iDCT */
    for(x = 0; x < w; x += 8)
    {
        int i,j;
        dequant_idct_block_8x8(in_data+(x*8), block, quantization);


        for (i=0; i<8; ++i)
            for (j=0; j<8; ++j)
            {
                /* Add prediction block. Note: DCT is not precise - Clamp to legal values */
                int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];
                if (tmp < 0)
                    tmp = 0;
                else if (tmp > 255)
                    tmp = 255;

                out_data[i*w+j+x] = tmp;
            }
    }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
			 uint8_t *out_data, uint8_t *quantization)
{
    uint32_t y;
    for (y=0; y<height; y+=8)
    {
        dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y, out_data+y*width, quantization);
    }
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
        int16_t *out_data, uint8_t *quantization)
{
    int x;

    int16_t block[8*8];

    /* Perform the DCT and quantization */
    for(x = 0; x < w; x += 8)
    {
        int i,j;
        for (i=0; i<8; ++i)
            for (j=0; j<8; ++j)
                block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);

        /* Store MBs linear in memory, i.e. the 64 coefficients are stored continous.
         * This allows us to ignore stride in DCT/iDCT and other functions. */
        dct_quant_block_8x8(block, out_data+(x*8), quantization);
    }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t *quantization)
{
    uint32_t y;
    for (y=0; y<height; y+=8)
    {
        dct_quantize_row(in_data+y*width, prediction+y*width, width, height, out_data+y*width, quantization);
    }
}

void destroy_frame(struct frame *f)
{
    if (!f) // First frame
        return;

    free(f->recons->Y);
    free(f->recons->U);
    free(f->recons->V);

    free(f->residuals->Ydct);
    free(f->residuals->Udct);
    free(f->residuals->Vdct);
    free(f->residuals);

    free(f->predicted->Y);
    free(f->predicted->U);
    free(f->predicted->V);
    free(f->predicted);

    free(f->mbs[0]);
    free(f->mbs[1]);
    free(f->mbs[2]);

    free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
    struct frame *f = (struct frame*)malloc(sizeof(struct frame));

    f->orig = image;

    f->recons = (yuv_t*)malloc(sizeof(yuv_t));
    f->recons->Y = (uint8_t*)malloc(cm->ypw * cm->yph);
    f->recons->U = (uint8_t*)malloc(cm->upw * cm->uph);
    f->recons->V = (uint8_t*)malloc(cm->vpw * cm->vph);

    f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
    f->predicted->Y = (uint8_t*)malloc(cm->ypw * cm->yph);
    f->predicted->U = (uint8_t*)malloc(cm->upw * cm->uph);
    f->predicted->V = (uint8_t*)malloc(cm->vpw * cm->vph);

    memset(f->predicted->Y, 0x80, cm->ypw * cm->yph);
    memset(f->predicted->U, 0x80, cm->upw * cm->uph);
    memset(f->predicted->V, 0x80, cm->vpw * cm->vph);

    f->residuals = (dct_t*)malloc(sizeof(dct_t));
    f->residuals->Ydct = (int16_t*)malloc(cm->ypw * cm->yph * sizeof(int16_t));
    f->residuals->Udct = (int16_t*)malloc(cm->upw * cm->uph * sizeof(int16_t));
    f->residuals->Vdct = (int16_t*)malloc(cm->vpw * cm->vph * sizeof(int16_t));

    memset(f->residuals->Ydct, 0x80, cm->ypw * cm->yph * sizeof(int16_t));
    memset(f->residuals->Udct, 0x80, cm->upw * cm->uph * sizeof(int16_t));
    memset(f->residuals->Vdct, 0x80, cm->vpw * cm->vph * sizeof(int16_t));

    f->mbs[0] = (struct macroblock*)calloc(cm->ypw * cm->yph, sizeof(struct macroblock));
    f->mbs[1] = (struct macroblock*)calloc(cm->upw * cm->uph, sizeof(struct macroblock));
    f->mbs[2] = (struct macroblock*)calloc(cm->vpw * cm->vph, sizeof(struct macroblock));

    return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
    fwrite(image->Y, 1, w*h, fp);
    fwrite(image->U, 1, w*h/4, fp);
    fwrite(image->V, 1, w*h/4, fp);
}

struct dct *dct_copy_write(struct c63_common *cm)
{
    struct dct *ret = (struct dct *)malloc(sizeof(struct dct));
    ret->Ydct = (int16_t*)malloc(cm->ypw * cm->yph * sizeof(int16_t));
    ret->Udct = (int16_t*)malloc(cm->upw * cm->uph * sizeof(int16_t));
    ret->Vdct = (int16_t*)malloc(cm->vpw * cm->vph * sizeof(int16_t));
    memcpy(ret->Ydct, cm->curframe->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
    memcpy(ret->Udct, cm->curframe->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
    memcpy(ret->Vdct, cm->curframe->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));
    return ret;
}

struct c63_common *cm_copy_write(struct c63_common *cm, struct entropy_ctx *entropy)
{
    struct c63_common *ret = (struct c63_common *)malloc(sizeof(struct c63_common));
    ret->ypw = cm->ypw;
    ret->yph = cm->yph;
    ret->upw = cm->upw;
    ret->uph = cm->uph;
    ret->vpw = cm->vpw;
    ret->vph = cm->vph;
    ret->padw[0] = cm->padw[0];
    ret->padw[1] = cm->padw[1];
    ret->padw[2] = cm->padw[2];
    ret->width = cm->width;
    ret->height = cm->height;
    memcpy(ret->quanttbl, cm->quanttbl, 3 * 64 * sizeof(uint8_t));
    ret->curframe = (struct frame *)malloc(sizeof(struct frame));
    ret->curframe->keyframe = cm->curframe->keyframe;
    ret->curframe->residuals = dct_copy_write(cm);
    ret->curframe->mbs[0]
        = (struct macroblock *)malloc(cm->ypw * cm->yph * sizeof(struct macroblock));
    ret->curframe->mbs[1]
        = (struct macroblock *)malloc(cm->upw * cm->uph * sizeof(struct macroblock));
    ret->curframe->mbs[2]
        = (struct macroblock *)malloc(cm->vpw * cm->vph * sizeof(struct macroblock));
    memcpy(ret->curframe->mbs[0], cm->curframe->mbs[0],
            cm->ypw * cm->yph * sizeof(struct macroblock));
    memcpy(ret->curframe->mbs[1], cm->curframe->mbs[1],
            cm->upw * cm->uph * sizeof(struct macroblock));
    memcpy(ret->curframe->mbs[2], cm->curframe->mbs[2],
            cm->vpw * cm->vph * sizeof(struct macroblock));
    ret->e_ctx = *entropy;
    return ret;
}

void destroy_frame_write(struct frame *f)
{
    free(f->residuals->Ydct);
    free(f->residuals->Udct);
    free(f->residuals->Vdct);
    free(f->residuals);
    free(f->mbs[0]);
    free(f->mbs[1]);
    free(f->mbs[2]);
    free(f);
}

void destroy_cm_write(struct c63_common *cm)
{
    destroy_frame_write(cm->curframe);
    free(cm);
}

void cuda_fake_cm_init(struct c63_common *cm) {
    cm->curframe = (struct frame *) malloc(sizeof(struct frame));

    cm->curframe->residuals = (dct_t *) malloc(sizeof(dct_t));
    cm->curframe->residuals->Ydct = 
        (int16_t *) malloc(cm->ypw * cm->yph * sizeof(int16_t));
    cm->curframe->residuals->Udct = 
        (int16_t *) malloc(cm->upw * cm->uph * sizeof(int16_t));
    cm->curframe->residuals->Vdct = 
        (int16_t *) malloc(cm->vpw * cm->vph * sizeof(int16_t));

    cm->curframe->mbs[0] = (struct macroblock *) malloc(cm->mb_cols
            * cm->mb_rows * sizeof(struct macroblock));
    cm->curframe->mbs[1] = (struct macroblock *) malloc(cm->mb_cols
            * cm->mb_rows / 4 * sizeof(struct macroblock));
    cm->curframe->mbs[2] = (struct macroblock *) malloc(cm->mb_cols
            * cm->mb_rows / 4 * sizeof(struct macroblock));
}
