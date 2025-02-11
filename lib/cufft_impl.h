/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_CUFFT_IMPL_H
#define INCLUDED_CUDA_CUFFT_IMPL_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cufft.h>

namespace gr {
namespace cuda {

class cufft_impl : public cufft, public cuda_block
{
private:
    // Nothing to declare in this block.
    int            i_fft_num;
    bool           b_forward;
    std::string    s_win_type;
    cudaStream_t   stream;
    cufftHandle    plan1d;
    cublasHandle_t cublas_handle;
    int            i_min_grid_size;
    int            i_block_size;
    float*         win_coe = nullptr;
    cudaDeviceProp prop;

protected:
    int calculate_output_stream_length(const gr_vector_int& ninput_items);

public:
    cufft_impl(int fft_num, const std::string& len_key, bool forward, std::string win_type);
    ~cufft_impl();

    // Where all the action really happens
    int work(int noutput_items, gr_vector_int& ninput_items, gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

}   // namespace cuda
}   // namespace gr

#endif /* INCLUDED_CUDA_CUFFT_IMPL_H */
