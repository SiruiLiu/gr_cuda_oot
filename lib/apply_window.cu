#include <gnuradio/cuda/cuda_error.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>

__global__ void kernelHammingWindow(int win_width, float* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < win_width){
        out[i] = 0.54f - 0.46f * __cosf((2 * M_PI * i) / ((win_width - 1)*1.0f));
    }
}

__global__ void kernelHanningWindow(int win_width, float* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < win_width){
        out[i] = 0.5f * (1 - __cosf((2 * M_PI * i) / ((win_width - 1)*1.0f)));
    }
}

__global__ void kernelBlackmanWindow(int win_width, float* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < win_width){
        out[i] = 0.42f - 0.5f * __cosf((2 * M_PI * i) / ((win_width-1)*1.0f)) +
              0.08f * __cosf((4 * M_PI * i) / ((win_width-1)*1.0f));
    }
}

__global__ void kernelApplyWindow(int win_width, float* coe, cuComplex* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < win_width){
        out[i].x = out[i].x*coe[i];
        out[i].y = out[i].y*coe[i];
    }
}

void genHammingWindow(int win_width, float* out, int grid_size, int block_size, cudaStream_t stream){
    kernelHammingWindow<<<grid_size, block_size, 0, stream>>>(win_width, out);
    check_cuda_errors(cudaGetLastError());
}

void genHanningWindow(int win_width, float* out, int grid_size, int block_size, cudaStream_t stream){
    kernelHanningWindow<<<grid_size, block_size, 0, stream>>>(win_width, out);
    check_cuda_errors(cudaGetLastError());
}

void genBlackmanWindow(int win_width, float* out, int grid_size, int block_size, cudaStream_t stream){
    kernelBlackmanWindow<<<grid_size, block_size, 0, stream>>>(win_width, out);
    check_cuda_errors(cudaGetLastError());
}

void ApplayWindow(int win_width, float* coe, cuComplex* out, int grid_size, int block_size, cudaStream_t stream){
    kernelApplyWindow<<<grid_size, block_size, 0, stream>>>(win_width, coe, out);
    check_cuda_errors(cudaGetLastError());
}

void get_block_and_grid(const std::string& win_type, int* minGrid_win, int* minBlock_win, int* minGrid_apply, int* minBlock_apply)
{
    if(win_type == "Hamming"){
        check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
            minGrid_win, minBlock_win, genHammingWindow, 0, 0));
    }
    else if(win_type == "Hanning"){
        check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
            minGrid_win, minBlock_win, genHanningWindow, 0, 0));
    }
    else if(win_type == "Blackman"){
        check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
            minGrid_win, minBlock_win, genBlackmanWindow, 0, 0));
    }
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid_apply, minBlock_apply, ApplayWindow, 0, 0));
}