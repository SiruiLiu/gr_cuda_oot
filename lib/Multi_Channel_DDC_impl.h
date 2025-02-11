/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H
#define INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H

#include <gnuradio/cuda/Multi_Channel_DDC.h>

namespace gr {
namespace cuda {

class Multi_Channel_DDC_impl : public Multi_Channel_DDC
{
private:
    // Nothing to declare in this block.

public:
    Multi_Channel_DDC_impl(int channel_num);
    ~Multi_Channel_DDC_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H */
