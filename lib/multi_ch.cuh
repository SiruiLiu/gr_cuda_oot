#ifndef __MULTI_CH_H__
#define __MULTI_CH_H__

#include <gnuradio/gr_complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <gnuradio/io_signature.h>

void applyWindow_multi_ch(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length,
                          dim3 grid_size, dim3 block_size, cudaStream_t stream);

void complex_to_mag_square(cufftComplex* input, cufftComplex* output, int total_length,
                           dim3 grid_size, dim3 block_size, cudaStream_t stream);


void Log10(cufftComplex* input, cufftComplex* output, int total_length,
           dim3 grid_size, dim3 block_size, cudaStream_t stream);

void SquareSum(cufftComplex* input, float* output, int ch_num,
               dim3 grid_size, dim3 block_size, cudaStream_t stream);

#endif