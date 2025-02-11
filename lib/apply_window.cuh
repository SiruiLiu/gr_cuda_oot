#ifndef __APPLY_WINDOW_H__
#define __APPLY_WINDOW_H__
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
void get_block_and_grid(const std::string& win_type, int* minGrid_win, int* minBlock_win,
                        int* minGrid_apply, int* minBlock_apply);
void genHammingWindow(int win_width, float* out, int grid_size, int block_size,
                      cudaStream_t stream);
void genHanningWindow(int win_width, float* out, int grid_size, int block_size,
                      cudaStream_t stream);

void genBlackmanWindow(int win_width, float* out, int grid_size, int block_size,
                       cudaStream_t stream);

void ApplayWindow(int win_width, float* coe, cuComplex* out, int grid_size, int block_size,
                  cudaStream_t stream);

#endif