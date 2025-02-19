/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_CUFFT_SYNC_IMPL_H
#define INCLUDED_CUDA_CUFFT_SYNC_IMPL_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cufft_sync.h>

namespace gr {
namespace cuda {

class cufft_sync_impl : public cufft_sync, public cuda_block
{
private:
    int            i_fft_num;
    bool           b_forward;
    std::string    s_win_type;
    cudaStream_t   stream;
    cufftHandle    plan1d;
    cublasHandle_t cublas_handle;
    dim3           i_min_grid_size;
    dim3           i_block_size;
    float*         win_coe = nullptr;
    cudaDeviceProp prop;

public:
    cufft_sync_impl(int fft_num, bool forward, std::string win_type);
    ~cufft_sync_impl();

    // Where all the action really happens
    int work(int noutput_items, gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

}   // namespace cuda
}   // namespace gr

#endif /* INCLUDED_CUDA_CUFFT_SYNC_IMPL_H */
