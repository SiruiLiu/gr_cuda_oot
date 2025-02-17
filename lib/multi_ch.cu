#include "multi_ch.cuh"
#include <gnuradio/gr_complex.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>

__global__ void kernelIn2Out(gr_complex** input, gr_complex** output, int ch_num, int vector_length){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < ch_num){
        output[idx] = input[idx];
    }
}


void applyIn2Out(gr_complex** input, gr_complex** output, int ch_num, int vector_length,
                int grid_size, int block_size, cudaStream_t stream){
    kernelIn2Out<<<grid_size, block_size, 0, stream>>>(input, output, ch_num, vector_length);
    check_cuda_errors(cudaGetLastError());
}