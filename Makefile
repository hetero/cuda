CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SDK_PATH ?= $(HOME)/NVIDIA_GPU_Computing_SDK

CXX = g++
CC = g++
NVCC = nvcc
LINK = g++

#CFLAGS = -O3 -Wall -g -D_XOPEN_SOURCE=600
#LDFLAGS = -lm 

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc

COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) -O3 -arch=sm_13 --ptxas-options=-v #--maxrregcount 16
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -O3 -Wall

OBJS_ENC = c63enc.o tables.o io.o c63_write.o common.o me.o dsp.o cuda_me.o cuda_dct.o cuda_idct.o
OBJS_DEC = c63dec.o tables.o io.o common.o me.o dsp.o

all: c63enc c63dec

cuda_me.o: cuda_me.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_dct.o: cuda_dct.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_idct.o: cuda_idct.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

c63enc: $(OBJS_ENC)
	$(LINK) -o c63enc $(OBJS_ENC) $(LIB_CUDA)

c63dec: $(OBJS_DEC)
	$(LINK) -o c63dec $(OBJS_DEC) $(LIB_CUDA)

clean:
	rm -f *.o c63enc c63dec
