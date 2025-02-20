/* -*- cu -*- */
#include "multi_ch.cuh"
#include <gnuradio/gr_complex.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>

__global__ void kernelApplyWindow(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < vector_length){
        output[i].x = input[i].x*win_coe[i];
        output[i].y = input[i].y*win_coe[i];
    }
}

__global__ void kernelComplex2MagSquare(cufftComplex* input, cufftComplex* output, int total_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < total_length){
        output[i].x = input[i].x*input[i].x + input[i].y*input[i].y;
        output[i].y = 0;
    }
}

__global__ void kernelLog10(cufftComplex* input, cufftComplex* output,int total_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < total_length){
        output[i].x = 10*log10f(input[i].x);
        output[i].y = 0;
    }
}

__global__ void kernelSquareSum(cufftComplex* input, float* output, 
                               int ch_num, int vector_length) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ch_num && j < vector_length) {
        atomicAdd(&output[i], input[j + i*vector_length].x);  // 原子操作
    }
}


void applyWindow_multi_ch(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length,
                 dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelApplyWindow<<<grid_size, block_size, 0, stream>>>(input, output, win_coe, vector_length);
    check_cuda_errors(cudaGetLastError());
}

void complex_to_mag_square(cufftComplex* input, cufftComplex* output, int total_length,
                           dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelComplex2MagSquare<<<grid_size, block_size, 0, stream>>>(input, output, total_length);
    check_cuda_errors(cudaGetLastError());
}

void Log10(cufftComplex* input, cufftComplex* output, int total_length,
           dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelLog10<<<grid_size, block_size, 0, stream>>>(input, output, total_length);
    check_cuda_errors(cudaGetLastError());
}

void SquareSum(cufftComplex* input, float* output, int ch_num, int vector_length,
               dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelSquareSum<<<grid_size, block_size, 0, stream>>>(input, output, ch_num, vector_length);
    check_cuda_errors(cudaGetLastError());
}
