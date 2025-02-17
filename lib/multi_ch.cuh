#ifndef __MULTI_CH_H__
#define __MULTI_CH_H__

#include <gnuradio/gr_complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gnuradio/io_signature.h>

void applyIn2Out(gr_complex** input, gr_complex** output, int ch_num, 
                 int grid_size, int block_size, cudaStream_t stream);

#endif