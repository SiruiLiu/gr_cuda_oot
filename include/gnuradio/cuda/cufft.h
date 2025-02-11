/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_CUFFT_H
#define INCLUDED_CUDA_CUFFT_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/tagged_stream_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief <+description of block+>
 * \ingroup cuda
 *
 */
class CUDA_API cufft : virtual public gr::tagged_stream_block
{
public:
    typedef std::shared_ptr<cufft> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::cufft.
     *
     * To avoid accidental use of raw pointers, cuda::cufft's
     * constructor is in a private implementation
     * class. cuda::cufft::make is the public interface for
     * creating new instances.
     */
    static sptr make(int fft_num, const std::string& len_key, bool forward, std::string win_type);
};

}   // namespace cuda
}   // namespace gr

#endif /* INCLUDED_CUDA_CUFFT_H */
