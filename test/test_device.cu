#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float *device_in, float *device_out, int N) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    
    unsigned int tid = threadIdx.x;//0-255
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;// block + 0-255

    sdata[tid] = device_in[i];
    __syncthreads();

    // 进行规约操作
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }   
    __syncthreads();

    // 将结果写入 global memory
    if (tid == 0) device_out[blockIdx.x] = sdata[tid];
}



extern "C" void launch_reduce(float *device_in, float* device_out, int const N){
    int block_num = N / THREAD_PER_BLOCK;
    dim3 Grid(block_num, 1);// 32*1024*1024 /256
    dim3 Block(THREAD_PER_BLOCK, 1);// 256
    reduce0<<<Grid, Block>>>(device_in, device_out, N);
}
