/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cufft_sync_impl.h"
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

// #pragma message("set the following appropriately and remove this warning")
using input_type = gr_complex;
// #pragma message("set the following appropriately and remove this warning")
using output_type = gr_complex;
cufft_sync::sptr cufft_sync::make(int fft_num, bool forward, std::string win_type)
{
    return gnuradio::make_block_sptr<cufft_sync_impl>(fft_num, forward, win_type);
}


/*
 * The private constructor
 */
cufft_sync_impl::cufft_sync_impl(int fft_num, bool forward, std::string win_type)
    : gr::sync_block(
          "cufft_sync",
          gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
          gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{}

/*
 * Our virtual destructor.
 */
cufft_sync_impl::~cufft_sync_impl() {}

int cufft_sync_impl::work(int noutput_items, gr_vector_const_void_star& input_items,
                          gr_vector_void_star& output_items)
{
    auto in  = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

#pragma message("Implement the signal processing in your block and remove this warning")
    // Do <+signal processing+>

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
