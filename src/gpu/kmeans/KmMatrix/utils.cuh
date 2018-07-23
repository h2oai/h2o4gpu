/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include "GpuInfo.cuh"

namespace H2O4GPU {
namespace KMeans {

M_DEVINLINE size_t global_thread_idx () {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

M_DEVINLINE size_t global_thread_idy () {
  return threadIdx.y + blockIdx.y * blockDim.y; 
}

M_DEVINLINE size_t grid_stride_x () {
  return blockDim.x * gridDim.x;
}

M_DEVINLINE size_t grid_stride_y () {
  return blockDim.y * gridDim.y;
}

template <typename T1, typename T2>
T1 M_HOSTDEVINLINE div_roundup(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}


// Work around for shared memory
// https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name
template <typename T>
struct KmSharedMem;

template <>
struct KmSharedMem<float> {
  __device__ float * ptr() {
    extern __shared__ __align__(sizeof(float)) float s_float[];
    return s_float;
  }
};

template <>
struct KmSharedMem<double> {
  __device__ double * ptr() {
    extern __shared__ __align__(sizeof(double)) double s_double[];
    return s_double;
  }
};

template <>
struct KmSharedMem<int> {
  __device__ int * ptr() {
    extern __shared__ __align__(sizeof(int)) int s_int[];
    return s_int;
  }
};


}  // KMeans
}  // H2O4GPU

#endif  // UTILS_CUH_