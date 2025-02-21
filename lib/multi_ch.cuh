#ifndef __MULTI_CH_H__
#define __MULTI_CH_H__

#include <gnuradio/gr_complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <driver_types.h>
#include <gnuradio/io_signature.h>

void applyWindow_multi_ch(cufftComplex* input, cufftComplex* output, float* win_coe, int vector_length,
                          dim3 grid_size, dim3 block_size, cudaStream_t stream);

void complex_to_mag_square(const cufftComplex* input, float* output, int total_length,
                           dim3 grid_size, dim3 block_size, cudaStream_t stream);


void Log10(float* input, float* output, int total_length,
           dim3 grid_size, dim3 block_size, cudaStream_t stream);

void SquareSum(float* input, float* output, int ch_num, int vector_length,
               dim3 grid_size, dim3 block_size, cudaStream_t stream);

void EstimP(float* input, float* output, int ch_num,
              int grid_size, int block_size, cudaStream_t steram);
#endif