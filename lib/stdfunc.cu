/* -*- cu -*- */
#include <stdio.h>
#include "stdfunc.cuh"
#include <gnuradio/gr_complex.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>

template <typename T>
__global__ void kernelClear(T* input, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        input[i] = 0;
    }
}

template <typename T>
__global__ void kernelMemSet(T* input, T value, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        input[i] = value;
    }
}

template <typename T>
void cuClearMemory(T* input, int length,
                   dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelClear<<<grid_size, block_size, 0, stream>>>(input, length);
    check_cuda_errors(cudaGetLastError());
}

template <typename T>
void cuMemSet(T* input, T value, int length,
              dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelMemSet<T><<<grid_size, block_size, 0, stream>>>(input, value, length);
    check_cuda_errors(cudaGetLastError());
}

#define IMPLEMENT_KERNEL(T)                          \
    template void cuMemSet<T>(T*, T, int, dim3, dim3, cudaStream_t); \
    template void cuClearMemory(T *input, int length, \
                               dim3 grid_size, dim3 block_size, cudaStream_t stream); \

IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(float)