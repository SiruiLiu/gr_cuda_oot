/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_CUFFT_SYNC_H
#define INCLUDED_CUDA_CUFFT_SYNC_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief <+description of block+>
 * \ingroup cuda
 *
 */
class CUDA_API cufft_sync : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<cufft_sync> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::cufft_sync.
     *
     * To avoid accidental use of raw pointers, cuda::cufft_sync's
     * constructor is in a private implementation
     * class. cuda::cufft_sync::make is the public interface for
     * creating new instances.
     */
    static sptr make(int fft_num, bool forward, std::string win_type);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_CUFFT_SYNC_H */
