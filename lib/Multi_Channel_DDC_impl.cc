/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "Multi_Channel_DDC_impl.h"
#include "apply_window.cuh"
#include "cufft_impl.h"
#include "gnuradio/cuda/Multi_Channel_DDC.h"
#include "multi_ch.cuh"
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

using input_type  = gr_complex;
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
    check_cuda_errors(cudaStreamCreate(&this->stream));   // Create cuda stream for further process
    this->createFFTPlane();                               // Create multiple channel FFT plan.
    this->allocateGPUSources();
    this->genWinCoe();   // Generate window coefficients for FFT process.
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
}

int Multi_Channel_DDC_impl::work(int noutput_items, gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    // auto in  = static_cast<const input_type*>(input_items[0]);
    // auto out = static_cast<output_type*>(output_items[0]);
    input_type*  in[this->i_ch_n];
    output_type* out[this->i_ch_n];

    for (int i = 0; i < this->i_ch_n; i++) {
        in[i]  = (input_type*)input_items[i];
        out[i] = (output_type*)output_items[i];
        applyWindow_multi_ch((cufftComplex*)in[i],
                             this->p_fft_memory_block + i * this->i_fft_num,
                             this->win_coe,
                             this->i_fft_num,
                             this->i_grid_size_for_win,
                             this->i_block_size_for_win,
                             this->stream);
        cudaStreamSynchronize(this->stream);
    }
    this->cuFFTProcess();
    complex_to_mag_square(this->p_fft_memory_block,
                          this->p_fft_memory_block,
                          this->i_ch_n * this->i_fft_num,
                          this->i_grid_size_for_abs,
                          this->i_block_size_for_abs,
                          this->stream);
    Log10(this->p_fft_memory_block,
          this->p_fft_memory_block,
          i_ch_n * this->i_fft_num,
          i_grid_size_for_abs,
          i_block_size_for_abs,
          this->stream);

    for (int i = 0; i < this->i_ch_n; i++) {
        cudaMemcpyAsync(out[i],
                        this->p_fft_memory_block + i * this->i_fft_num,
                        sizeof(gr_complex) * this->i_fft_num,
                        cudaMemcpyDeviceToDevice,
                        this->stream);
        cudaStreamSynchronize(this->stream);
    }

    return noutput_items;
}

void Multi_Channel_DDC_impl::createFFTPlane()
{
    cufftResult_t r = cufftPlan1d(&this->plan1d,
                                  this->i_fft_num,
                                  CUFFT_C2C,
                                  this->i_ch_n);   // Create multiple channel FFT plan
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create fft plan");
    }
    r = cufftSetStream(this->plan1d, this->stream);
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to set stream to fft plan");
    }
}

void Multi_Channel_DDC_impl::genWinCoe()
{
    check_cuda_errors(cudaMallocAsync((void**)&this->win_coe,
                                      sizeof(float) * this->i_fft_num,
                                      this->stream));   // Allocate memory for windows coefficients
    if (this->prop.maxThreadsPerBlock > this->i_fft_num) {
        this->i_block_size_for_win = this->i_fft_num;
    }
    else {
        this->i_block_size_for_win.x = this->prop.maxThreadsPerBlock;
    }
    this->i_grid_size_for_win.x =
        ceil((this->i_fft_num + this->i_block_size_for_win.x - 1) / this->i_block_size_for_win.x);
    genBlackmanWindow(this->i_fft_num,
                      this->win_coe,
                      this->i_grid_size_for_win,
                      i_block_size_for_win,
                      this->stream);
    cudaStreamSynchronize(this->stream);
}

void Multi_Channel_DDC_impl::allocateGPUSources()
{
    cudaGetDeviceProperties(&(this->prop), 0);
    this->allocateGPUSourcesforFFT();
    this->allocateGPUSourcesforAbs();
    this->allocateGPUSourcesforEstim();
}

void Multi_Channel_DDC_impl::allocateGPUSourcesforFFT()
{
    check_cuda_errors(cudaMallocAsync(&this->p_fft_memory_block,
                                      sizeof(gr_complex) * this->i_fft_num * this->i_ch_n,
                                      this->stream));

    cublasStatus_t status = cublasCreate(&this->cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUBLAS");
    }
}

void Multi_Channel_DDC_impl::allocateGPUSourcesforAbs()
{
    int total_length = this->i_fft_num * this->i_ch_n;
    std::cout << "Total length: " << total_length << std::endl;
    if (this->prop.maxThreadsPerBlock > total_length) {
        this->i_block_size_for_abs.x = total_length;
    }
    else {
        this->i_block_size_for_abs.x = this->prop.maxThreadsPerBlock;
    }

    this->i_grid_size_for_abs.x =
        (total_length + this->i_block_size_for_abs.x - 1) / this->i_block_size_for_abs.x;
}

void Multi_Channel_DDC_impl::allocateGPUSourcesforEstim()
{
    check_cuda_errors(
        cudaMallocAsync(&this->p_square_sum, sizeof(float) * this->i_ch_n, this->stream));
}

void Multi_Channel_DDC_impl::cuFFTProcess()
{
    cufftExecC2C(this->plan1d, this->p_fft_memory_block, this->p_fft_memory_block, CUFFT_FORWARD);
    float scale = 1.0f / this->i_fft_num;
    cublasSscal(
        cublas_handle, 2 * this->i_fft_num * this->i_ch_n, &scale, (float*)p_fft_memory_block, 1);
}

} /* namespace cuda */
} /* namespace gr */
