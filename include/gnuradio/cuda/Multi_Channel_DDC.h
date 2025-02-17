/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_MULTI_CHANNEL_DDC_H
#define INCLUDED_CUDA_MULTI_CHANNEL_DDC_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief <+description of block+>
 * \ingroup cuda
 *
 */
class CUDA_API Multi_Channel_DDC : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<Multi_Channel_DDC> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::Multi_Channel_DDC.
     *
     * To avoid accidental use of raw pointers, cuda::Multi_Channel_DDC's
     * constructor is in a private implementation
     * class. cuda::Multi_Channel_DDC::make is the public interface for
     * creating new instances.
     */
    static sptr make(int channel_num, float sample_rate, int vector_length);
};

}   // namespace cuda
}   // namespace gr

#endif /* INCLUDED_CUDA_MULTI_CHANNEL_DDC_H */
