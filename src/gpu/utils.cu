/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Modifications Copyright 2017 H2O.ai, Inc.
 ************************************************************************/
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "include/cuda_utils2.h"

#ifdef __cplusplus
extern "C" {
  #endif

  int cudaresetdevice(int wDev, int nDev) {
    if(nDev>0){
      int nVis = 0;
      CUDACHECK(cudaGetDeviceCount(&nVis));

      std::vector<int> dList(nDev);
      for (int i = 0; i < nDev; ++i){
        dList[i] = i % nVis;
        CUDACHECK(cudaSetDevice(dList[i]));
        CUDACHECK(cudaDeviceReset()); // reset device dList[i]
      }
    }  
    return(0);
  }

  #ifdef __cplusplus
}
#endif




#ifdef __cplusplus
extern "C" {
  #endif

    int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf) {
        // http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d_idx);
        *major = prop.major;
        *minor = prop.minor;
        *ratioperf = prop.singleToDoublePrecisionPerfRatio;
        return(0);
    }

  #ifdef __cplusplus
}
#endif
