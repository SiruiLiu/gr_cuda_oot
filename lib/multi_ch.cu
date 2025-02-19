#include "multi_ch.cuh"
#include <gnuradio/gr_complex.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <cufft.h>
#include <curand_mtgp32_kernel.h>

__global__ void kernelApplyWindow(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < vector_length){
        output[i].x = input[i].x*win_coe[i];
        output[i].y = input[i].y*win_coe[i];
    }
}

void applyWindow_multi_ch(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length,
                 dim3 grid_size, dim3 block_size, cudaStream_t stream){
    kernelApplyWindow<<<grid_size, block_size, 0, stream>>>(input, output, win_coe, vector_length);
    check_cuda_errors(cudaGetLastError());
}