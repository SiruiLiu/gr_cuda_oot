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
#include "stdfunc.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <npp.h>
#include <npps.h>

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
    this->allocateGPUSources();
}

/*
 * Our virtual destructor.
 */
Multi_Channel_DDC_impl::~Multi_Channel_DDC_impl()
{
    cufftDestroy(this->plan1d);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(this->stream);
    this->cuFree(this->win_coe);
    this->cuFree(p_square_sum);
    this->cuFree(this->p_fft_memory_block);
    this->cuFree(this->p_spectrum_block);
    this->cuFree(this->pMaximumVec);
    this->cuFree(this->bufForEstimMaximum);
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
                             this->stream);   // Apply window to data stream and send
                                              // to continuous memory block
        cudaStreamSynchronize(this->stream);
    }
    this->cuFFTProcess();
    this->cuEstimates();

    for (int i = 0; i < this->i_ch_n; i++) {
        cudaMemcpyAsync(out[i],
                        this->p_spectrum_block + i * this->i_fft_num,
                        sizeof(float) * this->i_fft_num,
                        cudaMemcpyDeviceToDevice,
                        this->stream);
        cudaStreamSynchronize(this->stream);
    }

    return noutput_items;
}
//! ************************************************ !//
//!                    下面为自定义函数                 !//
//! ************************************************ !//

/*
 * 分配GPU资源
 */
void Multi_Channel_DDC_impl::allocateGPUSources()
{
    cudaGetDeviceProperties(&(this->prop), 0);
    this->allocateGPUSourcesForFFT();
    this->allocateGPUSourcesForAbs();
    this->allocateGPUSourcesForEstim();
    this->genWinCoe();   // Generate window coefficients for FFT process.
}

/*
 * 创建FFT计划
 */
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

/*
 * 生成窗函数系数
 */
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

/*
 * 给GPU上的FFT处理分配计算资源
 */
void Multi_Channel_DDC_impl::allocateGPUSourcesForFFT()
{
    // 分配空间给FFT计算结果
    check_cuda_errors(cudaMallocAsync(&this->p_fft_memory_block,
                                      sizeof(gr_complex) * this->i_fft_num * this->i_ch_n,
                                      this->stream));
    // 创建cuBlas句柄
    cublasStatus_t status =
        cublasCreate(&this->cublas_handle);   // Prepare cublas handler for FFT normalization
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUBLAS");
    }
    this->createFFTPlane();   // 创建FFT计划
}

/*
 * 给GPU上的求模和求对数等操作分配计算资源
 */
void Multi_Channel_DDC_impl::allocateGPUSourcesForAbs()
{
    // 分配GPU计算资源
    int total_length = this->i_fft_num * this->i_ch_n;
    if (this->prop.maxThreadsPerBlock > total_length) {
        this->i_block_size_for_abs.x = total_length;
    }
    else {
        this->i_block_size_for_abs.x = this->prop.maxThreadsPerBlock;
    }

    this->i_grid_size_for_abs.x =
        (total_length + this->i_block_size_for_abs.x - 1) / this->i_block_size_for_abs.x;
}

/*
 * 给参数估计分配GPU计算资源
 */
void Multi_Channel_DDC_impl::allocateGPUSourcesForEstim()
{
    // 分配空间
    check_cuda_errors(
        cudaMallocAsync(&this->p_square_sum, sizeof(float) * this->i_ch_n, this->stream));
    check_cuda_errors(cudaMallocAsync(
        &this->p_spectrum_block, sizeof(float) * this->i_ch_n * this->i_fft_num, this->stream));
    check_cuda_errors(
        cudaMallocAsync(&this->pMaximumVec, sizeof(float) * this->i_ch_n, this->stream));
    nppsMaxGetBufferSize_32f(this->i_fft_num, &this->bufsizeForEstimMaximum);
    check_cuda_errors(
        cudaMallocAsync(&this->bufForEstimMaximum, bufsizeForEstimMaximum, this->stream));

    // 分配计算资源
    this->i_block_size_for_ssum.x = 32;
    this->i_block_size_for_ssum.y = 32;
    this->i_grid_size_for_ssum.x =
        (this->i_fft_num + this->i_block_size_for_ssum.x - 1) / this->i_block_size_for_ssum.x;
    this->i_grid_size_for_ssum.y =
        (this->i_ch_n + this->i_block_size_for_ssum.y - 1) / this->i_block_size_for_ssum.y;
}

/*
 * FFT处理和归一化
 */
void Multi_Channel_DDC_impl::cuFFTProcess()
{
    cufftExecC2C(this->plan1d,
                 this->p_fft_memory_block,
                 this->p_fft_memory_block,
                 CUFFT_FORWARD);            // FFT操作
    float scale = 1.0f / this->i_fft_num;   // 归一化因子，与FFT点数相关
    cublasSscal(cublas_handle,
                2 * this->i_fft_num * this->i_ch_n,
                &scale,
                (float*)p_fft_memory_block,
                1);   // 归一化
}

/*
 * 信号功率估计
 */
void Multi_Channel_DDC_impl::estimP()
{
    EstimP(this->p_square_sum,
           this->p_square_sum,
           this->i_ch_n,
           1,
           this->i_ch_n,
           this->stream);   // 估计功率的核函数
}

/*
 * 各通道最大值估计
 */
// todo 之后该函数应该放在带宽估计函数中
void Multi_Channel_DDC_impl::estimMaximumPerChannels()
{
    for (uint8_t i = 0; i < this->i_ch_n; i++) {
        nppsMax_32f(this->p_spectrum_block + i * this->i_fft_num,
                    this->i_fft_num,
                    &this->pMaximumVec[i],
                    this->bufForEstimMaximum);
    }
}

/*
 * 参数估计函数，包括求模，求对数，平滑，求功率，带宽，信噪比和频率等
 */
void Multi_Channel_DDC_impl::cuEstimates()
{
    complex_to_mag_square(this->p_fft_memory_block,
                          this->p_spectrum_block,
                          this->i_ch_n * this->i_fft_num,
                          this->i_grid_size_for_abs,
                          this->i_block_size_for_abs,
                          this->stream);   // 计算平方和
    cudaStreamSynchronize(this->stream);   // 同步操作
    cuClearMemory(this->p_square_sum,
                  this->i_ch_n,
                  this->i_grid_size_for_ssum,
                  this->i_block_size_for_ssum,
                  this->stream);           // 清空累加目的地址
    cudaStreamSynchronize(this->stream);   // 同步操作
    SquareSum(this->p_spectrum_block,
              this->p_square_sum,
              this->i_ch_n,
              this->i_fft_num,
              this->i_grid_size_for_ssum,
              this->i_block_size_for_ssum,
              this->stream);               // 按通道求和
    this->estimP();                        // 功率估计
    cudaStreamSynchronize(this->stream);   // 同步操作
    Log10(this->p_spectrum_block,
          this->p_spectrum_block,
          i_ch_n * this->i_fft_num,
          i_grid_size_for_abs,
          i_block_size_for_abs,
          this->stream);               // 计算对数
    this->estimMaximumPerChannels();   // 找出每个通道的最大值
    this->Display(this->pMaximumVec, this->i_ch_n);
    sleep(5);
}

} /* namespace cuda */
} /* namespace gr */