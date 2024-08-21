#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <cuda_runtime.h>

// 声明 CUDA 内核函数
extern "C" void launch_reduce(float *device_in, float* device_out, int const N);

#define THREAD_PER_BLOCK 256

int main() {
    int const N = 32 * 1024 * 1024;
    int block_num = N / THREAD_PER_BLOCK;
    // 主机端内存分配
    float *a = (float *)malloc(N * sizeof(float));
    float *out = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
    float *res = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));

    // 设备端内存分配
    float *device_a;
    float *device_out;
    cudaMalloc((void **)&device_a, N * sizeof(float));
    cudaMalloc((void **)&device_out, (N / THREAD_PER_BLOCK) * sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }
    
    std::cout<<"This is checking init input data."<<std::endl;
    std::cout<<"test a[0]   = "<<a[0]<<std::endl;
    std::cout<<"test a[1]   = "<<a[1]<<std::endl;
    std::cout<<"test a[256] = "<<a[256]<<std::endl;
    std::cout<<" "<<std::endl;

    #pragma omp parallel for
    for (int i = 0; i < block_num; i++) {
        float cur = 0.0f;
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            cur += a[i * THREAD_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    
    cudaMemcpy(device_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 Grid(N / THREAD_PER_BLOCK, 1);
    // dim3 Block(THREAD_PER_BLOCK, 1);
    // reduce0<<<Grid, Block>>>(device_a, device_out);
    launch_reduce(device_a, device_out, N);

    cudaMemcpy(out, device_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"This is each block reduce. "<<std::endl;
    std::cout<<"test out[0] = "<<out[0]<<std::endl;
    std::cout<<"test out[1] = "<<out[1]<<std::endl;
    std::cout<<"test out[2] = "<<out[2]<<std::endl;

    // 检查结果
    bool correct = true;
    float tolerance = 1e-4;
    for (int i = 0; i <  block_num; i++) {
        if (fabs(out[i] - res[i]) > tolerance) {
            correct = false;
            // Output the index and values for debugging
            printf("Mismatch at index %d: result = %f, expected = %f\n", i, out[i], res[i]);
            break; // If you want to find only the first mismatch, otherwise remove this line
        }
    }

    if (correct) {
        std::cout << "The result is correct." << std::endl;
    } else {
        std::cout << "The result is incorrect." << std::endl;
    }

    free(a);
    free(out);
    free(res);
    cudaFree(device_a);
    cudaFree(device_out);

    return 0;
}
