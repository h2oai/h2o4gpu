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
