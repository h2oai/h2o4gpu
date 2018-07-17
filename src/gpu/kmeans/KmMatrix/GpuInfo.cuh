#ifndef GPU_INFO_HPP_
#define GPU_INFO_HPP_

#include "KmConfig.h"
#include "stdlib.h"

class GpuInfo {
 private:
  int n_gpu_;
  size_t* n_sm_;
 public:
  GpuInfo () {
    CUDA_CHECK(cudaGetDeviceCount(&n_gpu_));
    n_sm_ = (size_t*) malloc (n_gpu_);
  }
  ~GpuInfo () {
    free (n_sm_);
  }

  size_t blocks (size_t _mul, int _device=0) {
    if (has_device(_device)) {
      return _mul * n_sm_[_device];
    }
    return 0;
  }

  bool has_device(int _device) {
    return _device < n_gpu_ && _device > 0;
  }

  static GpuInfo& ins() {
    static GpuInfo obj;
    return obj;
  }

};

// const GpuInfoImpl GpuInfo::impl = GpuInfoImpl();

#endif  // GPU_INFO_HPP_
