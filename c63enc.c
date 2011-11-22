#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#include <pthread.h>
#include <list>

#include "c63.h"
#include "tables.h"
#include "cuda_me.h"
#include "cuda_dct.h"
#include "cuda_idct.h"

using std::list;

// list for write requests
static list<c63_common*> write_list;
static list<pthread_t> th_id_list;
static struct entropy_ctx write_entropy;
pthread_mutex_t mutex;

static char *output_file, *input_file;
FILE *outfile;
//FILE *predfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;
static uint32_t yph;
static uint32_t ypw;
static uint32_t uph;
static uint32_t upw;
static uint32_t vph;
static uint32_t vpw;

// DCT device pointers
static uint8_t *dct_in_data_y;
static uint8_t *dct_prediction_y;
static int16_t *dct_out_data_y;
static uint8_t *dct_in_data_uv;
static uint8_t *dct_prediction_uv;
static int16_t *dct_out_data_uv;

static int16_t *idct_in_data_y;
static uint8_t *idct_prediction_y;
static uint8_t *idct_out_data_y;
static int16_t *idct_in_data_uv;
static uint8_t *idct_prediction_uv;
static uint8_t *idct_out_data_uv;

/* getopt */
extern int optind;
extern char *optarg;

// time measuring stuff
struct timespec start_time, stop_time;

float total_time = 0;

void start() {
        clock_gettime(CLOCK_REALTIME, &start_time);
}

void stop() {
        clock_gettime(CLOCK_REALTIME, &stop_time);
            total_time += stop_time.tv_sec - start_time.tv_sec + 
                        (stop_time.tv_nsec - start_time.tv_nsec) / 1e9;
}

void print_time() {
        printf("Measured time: %f\n", total_time);
}

void *thread_write_frame(void *a)
{
    pthread_mutex_lock(&mutex);
    write_frame(write_list.front());
    // small hack to remember entropy_ctx
    write_entropy = write_list.front()->e_ctx;
    destroy_cm_write(write_list.front());
    write_list.pop_front();
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

/* Read YUV frames */
static yuv_t* read_yuv(FILE *file)
{
    size_t len = 0;
    yuv_t *image = (yuv_t*)malloc(sizeof(yuv_t));

    printf("Reading...\n");

    /* Read Y' */
    image->Y = (uint8_t*)malloc(width*(height+8));
    len += fread(image->Y, 1, width*height, file);
    if(ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read U */
    image->U = (uint8_t*)malloc(width*height);
    len += fread(image->U, 1, (width*height)/4, file);
    if(ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read V */
    image->V = (uint8_t*)malloc(width*height);
    len += fread(image->V, 1, (width*height)/4, file);
    if(ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if(len != width*height*1.5)
    {
        fprintf(stderr, "Reached end of file.\n");
        return NULL;
    }

    return image;
}




static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
    /* Advance to next frame */
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, image);

    /* Check if keyframe */
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
    {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;

        fprintf(stderr, " (keyframe) ");
    }
    else
        cm->curframe->keyframe = 0;
    

    if (!cm->curframe->keyframe)
    {
        /* Motion Estimation */
        //start();
        //cuda_c63_motion_estimate(cm);
        //stop();
        //c63_motion_estimate(cm);
        /* Motion Compensation */
        //c63_motion_compensate(cm);
    }
   /* 
    dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[0], cm->padh[0], cm->curframe->residuals->Ydct, cm->quanttbl[0]);
    dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[1], cm->padh[1], cm->curframe->residuals->Udct, cm->quanttbl[1]);
    dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[2], cm->padh[2], cm->curframe->residuals->Vdct, cm->quanttbl[2]);
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[0]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[1]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[2]);
   */
    
    cuda_dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[0], cm->padh[0], cm->curframe->residuals->Ydct, 0, dct_in_data_y, dct_prediction_y, dct_out_data_y);
    cuda_dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[1], cm->padh[1], cm->curframe->residuals->Udct, 1, dct_in_data_uv, dct_prediction_uv, dct_out_data_uv);
    cuda_dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[2], cm->padh[2], cm->curframe->residuals->Vdct, 1, dct_in_data_uv, dct_prediction_uv, dct_out_data_uv);

    cuda_dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, 0, idct_in_data_y, idct_prediction_y, idct_out_data_y);
    cuda_dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, 1, idct_in_data_uv, idct_prediction_uv, idct_out_data_uv);
    cuda_dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, 1, idct_in_data_uv, idct_prediction_uv, idct_out_data_uv);
    
    // dump_image can be used here to check if the prediction is correct.
    //dump_image(cm->curframe->predicted, cm->width, cm->height, predfile);

    
    write_list.push_back(cm_copy_write(cm, &write_entropy));
    pthread_t t;
    pthread_create(&t, NULL, thread_write_frame, NULL);
    th_id_list.push_back(t);
    //write_frame(cm);

    ++cm->framenum;
    ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
    int i;
    struct c63_common *cm = (struct c63_common*)calloc(1, sizeof(struct c63_common));

    cm->width = width;
    cm->height = height;
    cm->padw[0] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
    cm->padh[0] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
    cm->padw[1] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
    cm->padh[1] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
    cm->padw[2] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
    cm->padh[2] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

    cm->mb_cols = cm->ypw / 8;
    cm->mb_rows = cm->yph / 8;


    /* Quality parameters */
    // ---------------------------------------------------------------
    // if you want to change this you need to prepare new __constant__
    // quant tables in cuda_dct.cu 
    cm->qp = 25;                 // Constant quantization factor. Range: [1..50]
    // ---------------------------------------------------------------
    cm->me_search_range = 16;    // Pixels in every direction
    cm->keyframe_interval = 100;  // Distance between keyframes


    /* Initialize quantization tables */
    for (i=0; i<64; ++i)
    {
        cm->quanttbl[0][i] = yquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[1][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[2][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    }

    return cm;
}

static void print_help()
{
    fprintf(stderr, "Usage: ./c63enc [options] input_file\n");
    fprintf(stderr, "Commandline options:\n");
    fprintf(stderr, "  -h                             height of images to compress\n");
    fprintf(stderr, "  -w                             width of images to compress\n");
    fprintf(stderr, "  -o                             Output file (.mjpg)\n");
    fprintf(stderr, "  [-f]                           Limit number of frames to encode\n");
    fprintf(stderr, "\n");

    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    pthread_mutex_init (&mutex, NULL);
    int c;
    yuv_t *image;

    if(argc == 1)
    {
        print_help();
    }

    while((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
    {
        switch(c)
        {
        case 'h':
            height = atoi(optarg);
            break;
        case 'w':
            width = atoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'f':
            limit_numframes = atoi(optarg);
            break;
        default:
            print_help();
            break;
        }
    }


    if(optind >= argc)
    {
        fprintf(stderr, "Error getting program options, try --help.\n");
        exit(EXIT_FAILURE);
    }


    ///

    //height = 288;
    //width = 352;
    //limit_numframes = 2;
    //output_file = "~/foreman.c63";

    ///
    
    outfile = fopen(output_file, "wb");
    if(outfile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

//    predfile = fopen("/tmp/pred.yuv", "wb");


    struct c63_common *cm = init_c63_enc(width, height);
    cm->e_ctx.fp = outfile;
    write_entropy = cm->e_ctx;


    /* Calculate the padded width and height */
    ypw = (uint32_t)(ceil(width/8.0f)*8);
    yph = (uint32_t)(ceil(height/8.0f)*8);
    upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
    uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
    vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
    vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

    input_file = argv[optind];

    ///
    //input_file = "~/foreman.yuv";
    ///

    if (limit_numframes)
        fprintf(stderr, "Limited to %d frames.\n", limit_numframes);

    FILE *infile = fopen(input_file, "rb");

    if(infile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }


    /* Encode input frames */
    cuda_dct_malloc(cm->padw[0] * cm->padh[0], &dct_in_data_y,
            &dct_prediction_y, &dct_out_data_y);
    cuda_dct_malloc(cm->padw[1] * cm->padh[1], &dct_in_data_uv,
            &dct_prediction_uv, &dct_out_data_uv);
    cuda_idct_malloc(cm->padw[0] * cm->padh[0], &idct_in_data_y,
            &idct_prediction_y, &idct_out_data_y);
    cuda_idct_malloc(cm->padw[1] * cm->padh[1], &idct_in_data_uv,
            &idct_prediction_uv, &idct_out_data_uv);
    int numframes = 0;
    while(!feof(infile))
    {
        image = read_yuv(infile);
        if (!image) {
            break;
        }

        fprintf(stderr, "Encoding frame %d, ", numframes);
        c63_encode_image(cm, image);

        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);

        fprintf(stderr, "Done!\n");

        ++numframes;
        if (limit_numframes && numframes >= limit_numframes)
            break;
    }
    cuda_dct_free(dct_in_data_y, dct_prediction_y, dct_out_data_y);
    cuda_dct_free(dct_in_data_uv, dct_prediction_uv, dct_out_data_uv);
    cuda_idct_free(idct_in_data_y, idct_prediction_y, idct_out_data_y);
    cuda_idct_free(idct_in_data_uv, idct_prediction_uv, idct_out_data_uv);

    while (!th_id_list.empty())
    {
        pthread_join(th_id_list.front(), NULL);
        th_id_list.pop_front();
    }

    fclose(outfile);
    fclose(infile);
//    fclose(predfile);


    print_time();
    return EXIT_SUCCESS;
}
