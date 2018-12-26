/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef GPU_INFO_HPP_
#define GPU_INFO_HPP_

#include "utils.cuh"

#include <cublas_v2.h>

#include <stdlib.h>
#include <stdio.h>

namespace h2o4gpu {
// Singleton class storing gpu info.
// Call GpuInfo::ins() to use the class;
class GpuInfo {
 private:
  int n_gpu_;
  int* n_sm_;  // number of gpu processors for each device
  cublasHandle_t* handles_;  // handle for each device

 public:
  GpuInfo () {
    safe_cuda(cudaGetDeviceCount(&n_gpu_));
    n_sm_ = (int*) malloc (n_gpu_);
    handles_ = (cublasHandle_t*) malloc (n_gpu_);

    for (int i = 0; i < n_gpu_; ++i) {
      cudaDeviceGetAttribute(&n_sm_[i], cudaDevAttrMultiProcessorCount, i);
      safe_cublas(cublasCreate(&handles_[i]));
    }
  }
  ~GpuInfo () {
    free (n_sm_);
    for (int i = 0; i < n_gpu_; ++i) {
      safe_cublas(cublasDestroy(handles_[i]));
    }
    free (handles_);
  }

  static GpuInfo& ins() {
    static GpuInfo obj;
    return obj;
  }

  // Call the following methods with GpuInfo::ins(). For example:
  // GpuInfo::ins().blocks(32)

  // Get number of blocks for grid strided loop kernel.
  // returns _mul * MultiProcessorCount[device].
  // FIXME, get active device
  size_t blocks (size_t _mul, int _device=0) {
    if (has_device(_device)) {
      return _mul * n_sm_[_device];
    } else {
      fprintf(stderr, "Doesn't have device: %d\n", _device);
      abort();
    }
  }
  // FIXME, ditto
  cublasHandle_t cublas_handle(int _device=0) {
    if (has_device(_device)) {
      return handles_[_device];
    } else {
      fprintf(stderr, "Doesn't have device: %d\n", _device);
      abort();
    }
  }

  bool has_device(int _device) {
    return _device < n_gpu_ && _device >= 0;
  }
};

}      // namespace h2o4gpu

#endif  // GPU_INFO_HPP_
