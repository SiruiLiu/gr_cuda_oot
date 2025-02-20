/* -*- c++ -*- */
/*
 * Copyright 2025 Dayao.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H
#define INCLUDED_CUDA_MULTI_CHANNEL_DDC_IMPL_H

#include "cufft_impl.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <gnuradio/cuda/Multi_Channel_DDC.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/sync_block.h>

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

    //* Pointers
    float*         win_coe = nullptr;
    cudaDeviceProp prop;
    cufftComplex*  p_fft_memory_block = nullptr;
    float*         p_square_sum       = nullptr;

    //* GPU grid and block size variables
    dim3 i_block_size_for_win;
    dim3 i_grid_size_for_win;
    dim3 i_grid_size_for_abs;
    dim3 i_block_size_for_abs;
    dim3 i_block_size_for_ssum;
    dim3 i_grid_size_for_ssum;

protected:
    /**
     * @brief Create an CUDA FFT plan for FFT Process
     *
     */
    void createFFTPlane();
    /**
     * @brief Generate window coefficients
     *
     */
    void genWinCoe();
    /**
     * @brief Allocate memory for GPU sources, this functions
     * includes other gpu sources allcating functions for
     * specified operations
     *
     */
    void allocateGPUSources();
    /**
     * @brief Allocate GPU sources for FFT process
     *
     */
    void allocateGPUSourcesForFFT();
    /**
     * @brief Allocate GPU sources for calculating absolute values
     * and log results
     *
     */
    void allocateGPUSourcesForAbs();
    /**
     * @brief Allocate GPU sources for parameter estimation
     *
     */
    void allocateGPUSourcesForEstim();
    /**
     * @brief Do FFT process
     *
     */
    void cuFFTProcess();
    /**
     * @brief Do parameter estimation
     *
     */
    void cuEstimates();

    template<typename T> inline void Display(const T* data, int length)
    {
        T* tmp = new T[length];
        cudaMemcpyAsync(tmp, data, sizeof(T) * length, cudaMemcpyDeviceToHost, this->stream);
        cudaStreamSynchronize(this->stream);
        for (int i = 0; i < length; i++) {
            std::cout << "Value of index " << i << " is " << tmp[i] << std::endl;
        }
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
