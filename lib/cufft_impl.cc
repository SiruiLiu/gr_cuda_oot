/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "apply_window.h"
#include "cufft_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <stdexcept>


namespace gr {
namespace cuda {

// #pragma message("set the following appropriately and remove this warning")
using input_type = gr_complex;
// #pragma message("set the following appropriately and remove this warning")
using output_type = gr_complex;
cufft::sptr cufft::make(int fft_num, const std::string& len_key, bool forward, std::string win_type)
{
    return gnuradio::make_block_sptr<cufft_impl>(fft_num, len_key, forward, win_type);
}


/*
 * The private constructor
 */
cufft_impl::cufft_impl(int fft_num, const std::string& len_key, bool forward, std::string win_type)
    : gr::tagged_stream_block("cufft",
                              gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */,
                                                     sizeof(input_type), cuda_buffer::type),
                              gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */,
                                                     sizeof(output_type), cuda_buffer::type),
                              len_key)
    , i_fft_num(fft_num)
    , b_forward(forward)
    , s_win_type(win_type)
{
    cudaGetDeviceProperties(&(this->prop), 0);
    this->i_block_size    = this->prop.maxBlocksPerMultiProcessor;
    this->i_min_grid_size = ceil((this->i_fft_num + this->i_block_size - 1) / this->i_block_size);
    check_cuda_errors(cudaStreamCreate(&this->stream));
    check_cuda_errors(
        cudaMallocAsync((void**)&this->win_coe, sizeof(float) * this->i_fft_num, this->stream));
    cufftResult_t r = cufftPlan1d(&this->plan1d, this->i_fft_num, CUFFT_C2C, 1);
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create fft plan");
    }
    r = cufftSetStream(this->plan1d, this->stream);
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to bind stream to fft plan");
    }
    cublasStatus_t status = cublasCreate(&this->cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUBLAS");
    }
    if (this->b_forward) {   // Only forward FFT needs window coefficients.
        std::cout << "Do window coefficients generating" << std::endl;
        if (this->s_win_type == "Hamming") {
            genHammingWindow(this->i_fft_num,
                             this->win_coe,
                             this->i_min_grid_size,
                             this->i_block_size,
                             this->stream);
            std::cout << "Finished" << std::endl;
        }
        else if (this->s_win_type == "Hanning") {
            genHanningWindow(this->i_fft_num,
                             this->win_coe,
                             this->i_min_grid_size,
                             this->i_block_size,
                             this->stream);
            std::cout << "Finished" << std::endl;
        }
        else if (this->s_win_type == "Blackman") {
            genBlackmanWindow(this->i_fft_num,
                              this->win_coe,
                              this->i_min_grid_size,
                              this->i_block_size,
                              this->stream);
            std::cout << "Finished" << std::endl;
        }
    }
}

/*
 * Our virtual destructor.
 */
cufft_impl::~cufft_impl() {}

int cufft_impl::calculate_output_stream_length(const gr_vector_int& ninput_items)
{
    // #pragma message("set the following appropriately and remove this warning")
    int noutput_items = ninput_items[0];
    return noutput_items;
}

int cufft_impl::work(int noutput_items, gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items, gr_vector_void_star& output_items)
{
    auto in  = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    noutput_items = ninput_items[0];

    if (this->b_forward) {
        if (this->s_win_type != "Rectangle") {
            ApplayWindow(this->i_fft_num,
                         this->win_coe,
                         (cuComplex*)in,
                         this->i_min_grid_size,
                         this->i_block_size,
                         this->stream);
        }
        cufftExecC2C(this->plan1d, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD);
        float scale = 1.0f / this->i_fft_num;
        cublasSscal(cublas_handle, 2 * this->i_fft_num, &scale, (float*)out, 1);
        // check_cuda_errors(cudaMemcpyAsync(
        // out, in, sizeof(gr_complex) * this->i_fft_num, cudaMemcpyDeviceToDevice, this->stream));
    }
    else {
        cufftExecC2C(this->plan1d, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE);
    }
    // #pragma message("Implement the signal processing in your block and remove this warning")
    // Do <+signal processing+>
    cudaStreamSynchronize(this->stream);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
