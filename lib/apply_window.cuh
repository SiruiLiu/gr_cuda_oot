#ifndef __APPLY_WINDOW_H__
#define __APPLY_WINDOW_H__
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

void genHammingWindow(int win_width, float* out, dim3 grid_size, dim3 block_size,
                      cudaStream_t stream);
void genHanningWindow(int win_width, float* out, dim3 grid_size, dim3 block_size,
                      cudaStream_t stream);

void genBlackmanWindow(int win_width, float* out, dim3 grid_size, dim3 block_size,
                       cudaStream_t stream);
void ApplyWindow(int win_width, float* coe, cuComplex* out, dim3 grid_size, dim3 block_size,
                  cudaStream_t stream);

#endif