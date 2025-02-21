/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H
#define INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H

#include "cufft_impl.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <gnuradio/cuda/Multi_Channel_DDC.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/sync_block.h>
#include <npp.h>
#include <npps.h>

namespace gr {
namespace cuda {

class Multi_Channel_DDC_impl : public Multi_Channel_DDC
{
private:
    //* Normal variables
    int            i_fft_num = 0;
    float          f_sr      = 1.0f;
    int            i_ch_n    = 0;
    cudaStream_t   stream;
    cufftHandle    plan1d;
    cublasHandle_t cublas_handle;
    cudaDeviceProp prop;

    size_t bufsizeForEstimMaximum;   // 估计最大值需要分配的内存空间的大小

    //* Pointers
    float*        win_coe            = nullptr;
    cufftComplex* p_fft_memory_block = nullptr;
    float*        p_spectrum_block   = nullptr;
    float*        p_square_sum       = nullptr;

    Npp8u* bufForEstimMaximum = nullptr;   // NPP库估计最大值用到的显存空间指针
    float* pMaximumVec        = nullptr;   // 每个通道频谱最大值结果。

    //* GPU grid and block size variables
    dim3 i_block_size_for_win;
    dim3 i_grid_size_for_win;
    dim3 i_grid_size_for_abs;
    dim3 i_block_size_for_abs;
    dim3 i_block_size_for_ssum;
    dim3 i_grid_size_for_ssum;

protected:
    /**
     * @brief 分配GPU资源，这个函数包含了为各个计算
     * 操作分配GPU内存空间和计算资源的函数
     *
     */
    void allocateGPUSources();
    /**
     * @brief 给GPU的FFT处理创建计划
     */
    void createFFTPlane();
    /**
     * @brief 生成窗函数
     */
    void genWinCoe();
    /**
     * @brief 给FFT处理分配资源和空间
     */
    void allocateGPUSourcesForFFT();
    /**
     * @brief 给求模操作和求对数操作等分配GPU资源
     */
    void allocateGPUSourcesForAbs();
    /**
     * @brief 给参数估计计算分配GPU资源
     */
    void allocateGPUSourcesForEstim();
    /**
     * @brief 执行FFT操作
     */
    void cuFFTProcess();
    /**
     * @brief 执行参数估计
     */
    void cuEstimates();
    /**
     * @brief 功率估计
     */
    void estimP();
    /**
     * @brief 各通道峰值功率估计
     */
    void estimMaximumPerChannels();
    /**
     * @brief 打印GPU中数据的值
     *
     * @tparam T 待打印的值的数据类型
     * @param data 指向待打印的数据的首地址的指针
     * @param length 打印长度
     */
    template<typename T> inline void Display(const T* data, int length)
    {
        T* tmp = new T[length];
        cudaMemcpyAsync(tmp, data, sizeof(T) * length, cudaMemcpyDeviceToHost, this->stream);
        cudaStreamSynchronize(this->stream);
        for (int i = 0; i < length; i++) {
            std::cout << "Value of index " << i << " is " << tmp[i] << std::endl;
        }
    }
    /**
     * @brief 释放GPU资源，在cudaFree函数上封装了一层，避免悬空指针
     *
     * @param devPtr 待释放指针
     */
    inline void cuFree(void* devPtr)
    {
        cudaFree(devPtr);
        devPtr = nullptr;
    }

public:
    Multi_Channel_DDC_impl(int channel_num, float sample_rate, int vector_length);
    ~Multi_Channel_DDC_impl();

    // Where all the action really happens
    int work(int noutput_items, gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

}   // namespace cuda
}   // namespace gr

#endif /* INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H */
