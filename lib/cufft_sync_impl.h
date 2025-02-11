/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_CUFFT_SYNC_IMPL_H
#define INCLUDED_CUDA_CUFFT_SYNC_IMPL_H

#include <gnuradio/cuda/cufft_sync.h>

namespace gr {
namespace cuda {

class cufft_sync_impl : public cufft_sync
{
private:
    // Nothing to declare in this block.

public:
    cufft_sync_impl(int fft_num, bool forward, std::string win_type);
    ~cufft_sync_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_CUFFT_SYNC_IMPL_H */
