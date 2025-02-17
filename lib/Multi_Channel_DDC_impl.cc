/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "Multi_Channel_DDC_impl.h"
#include "cufft_impl.h"
#include "gnuradio/cuda/Multi_Channel_DDC.h"
#include "multi_ch.cuh"
#include <cstddef>
#include <cuda_runtime.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

// #pragma message("set the following appropriately and remove this warning")
using input_type = gr_complex;
// #pragma message("set the following appropriately and remove this warning")
using output_type = gr_complex;
Multi_Channel_DDC::sptr Multi_Channel_DDC::make(int channel_num, float sample_rate,
                                                int vector_length)
{
    return gnuradio::make_block_sptr<Multi_Channel_DDC_impl>(
        channel_num, sample_rate, vector_length);
}


/*
 * The private constructor
 */
Multi_Channel_DDC_impl::Multi_Channel_DDC_impl(int channel_num, float sample_rate,
                                               int vector_length)
    : gr::sync_block("Multi_Channel_DDC",
                     gr::io_signature::make(1 /* min inputs */, -1 /* max inputs */,
                                            sizeof(input_type) * vector_length, cuda_buffer::type),
                     gr::io_signature::make(1 /* min outputs */, -1 /*max outputs */,
                                            sizeof(output_type) * vector_length, cuda_buffer::type))
    , i_ch_n(channel_num)
    , f_sr(sample_rate)
    , i_fft_num(vector_length)
{
    check_cuda_errors(cudaStreamCreate(&this->stream));
    check_cuda_errors(
        cudaMallocAsync((void**)&this->win_coe, sizeof(float) * this->i_fft_num, this->stream));
    cufftResult_t r = cufftPlan1d(&this->plan1d, this->i_fft_num, CUFFT_C2C, 1);
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create fft plan");
    }
    cublasStatus_t status = cublasCreate(&this->cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUBLAS");
    }
    r                           = cufftSetStream(this->plan1d, this->stream);
    this->p_complex_size_vector = new size_t[this->i_ch_n];
    memset(
        this->p_complex_size_vector, (size_t)(sizeof(gr_complex) * this->i_fft_num), this->i_ch_n);
}

/*
 * Our virtual destructor.
 */
Multi_Channel_DDC_impl::~Multi_Channel_DDC_impl()
{
    cufftDestroy(this->plan1d);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(this->stream);
    cudaFree(this->win_coe);
    delete this->p_complex_size_vector;
    this->p_complex_size_vector = nullptr;
}

int Multi_Channel_DDC_impl::work(int noutput_items, gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    // auto in  = static_cast<const input_type*>(input_items[0]);
    // auto out = static_cast<output_type*>(output_items[0]);
    input_type*  in[this->i_ch_n];
    output_type* out[this->i_ch_n];

    cudaMemcpyBatchAsync(
        out, in, this->p_complex_size_vector, this->i_ch_n, 0, 0, 0, 0, this->stream);

    // #pragma message("Implement the signal processing in your block and remove this warning")
    // Do <+signal processing+>

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

void Multi_Channel_DDC_impl::allocateGPUSources()
{
    cudaGetDeviceProperties(&(this->prop), 0);
    this->allocateGPUSourcesforFFT();
    this->allocateGPUSourcesforIn2Out();
}
void Multi_Channel_DDC_impl::allocateGPUSourcesforIn2Out()
{
    if (this->prop.maxThreadsPerMultiProcessor > this->i_ch_n) {
        this->i_min_grid_size_for_in2out = 1;
        this->i_block_size_for_in2out    = 1;
    }
    else {
        this->i_min_grid_size_for_in2out =
            ceil((this->i_ch_n + this->prop.maxThreadsPerMultiProcessor - 1) /
                 this->prop.maxThreadsPerMultiProcessor);
        this->i_block_size_for_in2out = this->prop.maxThreadsPerMultiProcessor;
    }
}
void Multi_Channel_DDC_impl::allocateGPUSourcesforFFT()
{
    this->i_block_size_for_fft = this->prop.maxBlocksPerMultiProcessor;
    this->i_min_grid_size_for_fft =
        ceil((this->i_fft_num + this->i_block_size_for_fft - 1) / this->i_block_size_for_fft);
}

} /* namespace cuda */
} /* namespace gr */
