#include <gnuradio/gr_complex.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>


template <typename T>
void cuClearMemory(T* input, int length,
                   dim3 grid_size, dim3 block_size, cudaStream_t stream);

template <typename T>
void cuMemSet(T* input, T value, int length,
              dim3 grid_size, dim3 block_size, cudaStream_t stream);
