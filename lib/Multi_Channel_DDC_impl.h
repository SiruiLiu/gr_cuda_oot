/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H
#define INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H

#include "cufft_impl.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <gnuradio/cuda/Multi_Channel_DDC.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

class Multi_Channel_DDC_impl : public Multi_Channel_DDC
{
private:
    int            i_fft_num = 0;
    float          f_sr      = 1.0f;
    int            i_ch_n    = 0;
    cudaStream_t   stream;
    cufftHandle    plan1d;
    cublasHandle_t cublas_handle;
    int            i_min_grid_size_for_fft;
    int            i_block_size_for_fft;
    int            i_min_grid_size_for_in2out;
    int            i_block_size_for_in2out;
    float*         win_coe = nullptr;
    cudaDeviceProp prop;
    size_t*        p_complex_size_vector = nullptr;

public:
    Multi_Channel_DDC_impl(int channel_num, float sample_rate, int vector_length);
    ~Multi_Channel_DDC_impl();

    // Where all the action really happens
    int  work(int noutput_items, gr_vector_const_void_star& input_items,
              gr_vector_void_star& output_items);
    void allocateGPUSources();
    void allocateGPUSourcesforIn2Out();
    void allocateGPUSourcesforFFT();
};

}   // namespace cuda
}   // namespace gr

#endif /* INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H */
