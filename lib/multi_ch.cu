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

__global__ void kernelComplex2MagSquare(const cufftComplex* input, float* output, int total_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < total_length){
        output[i] = input[i].x*input[i].x + input[i].y*input[i].y;
    }
}

__global__ void kernelLog10(float* input, float* output,int total_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < total_length){
        output[i] = 10*log10f(input[i]);
    }
}

__global__ void kernelSquareSum(float* input, float* output, 
                               int ch_num, int vector_length) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ch_num && j < vector_length) {
        atomicAdd(&output[i], input[j + i*vector_length]); 
    }
}

__global__ void kernelEstimP(float* input, float* output, int ch_num){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < ch_num){
        output[i] = 10*log10f(input[i]);
    }
}

void applyWindow_multi_ch(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length,
                 dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelApplyWindow<<<grid_size, block_size, 0, stream>>>(input, output, win_coe, vector_length);
    check_cuda_errors(cudaGetLastError());
}

void complex_to_mag_square(const cufftComplex* input, float* output, int total_length,
                           dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelComplex2MagSquare<<<grid_size, block_size, 0, stream>>>(input, output, total_length);
    check_cuda_errors(cudaGetLastError());
}

void Log10(float* input, float* output, int total_length,
           dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelLog10<<<grid_size, block_size, 0, stream>>>(input, output, total_length);
    check_cuda_errors(cudaGetLastError());
}

void SquareSum(float* input, float* output, int ch_num, int vector_length,
               dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelSquareSum<<<grid_size, block_size, 0, stream>>>(input, output, ch_num, vector_length);
    check_cuda_errors(cudaGetLastError());
}

void EstimP(float* input, float* output, int ch_num,
            int grid_size, int block_size, cudaStream_t steram){
    kernelEstimP<<<grid_size, block_size,0,steram>>>(input, output, ch_num);        
    check_cuda_errors(cudaGetLastError());
}