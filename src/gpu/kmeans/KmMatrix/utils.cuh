#ifndef UTILS_CUH_
#define UTILS_CUH_

#include "GpuInfo.cuh"

namespace H2O4GPU {
namespace KMeans {

M_DEVINLINE size_t global_thread_idx () {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

M_DEVINLINE size_t grid_stride () {
  return blockDim.x * gridDim.x;
}

// This wrapper function is created to work around a possible bug in nvcc,
// which threats GpuInfo::ins() as calling base class method when used inside a
// class member function.
size_t get_blocks(size_t _mul, int _device=0) {
  return GpuInfo::ins().blocks(_mul, _device);
}

}  // KMeans
}  // H2O4GPU

#endif  // UTILS_CUH_