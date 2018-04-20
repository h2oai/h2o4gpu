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
#include <iostream>
#include <assert.h>
#include "nvml.h"

#include "include/cuda_utils2.h"

int cudaresetdevice(int wDev, int nDev) {
  if (nDev > 0) {
    int nVis = 0;
    CUDACHECK(cudaGetDeviceCount(&nVis));

    std::vector<int> dList(nDev);
    for (int i = 0; i < nDev; ++i) {
      dList[i] = i % nVis;
      CUDACHECK(cudaSetDevice(dList[i]));
      CUDACHECK(cudaDeviceReset()); // reset device dList[i]
    }
  }
  return (0);
}

int cudaresetdevice_bare(void) {
  cudaDeviceReset();
    return(0);
}

int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf) {
  // http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, d_idx);
  *major = prop.major;
  *minor = prop.minor;
  *ratioperf = prop.singleToDoublePrecisionPerfRatio;
  return (0);
}


void get_gpu_info_c(unsigned int *n_gpus, int *gpu_percent_usage, unsigned long long *gpu_total_memory, char **gpu_name) {

  nvmlReturn_t rv;
  rv = nvmlInit();
  assert(rv == NVML_SUCCESS);
  //unsigned int n_gpus;
  rv = nvmlDeviceGetCount(n_gpus);

  assert(rv == NVML_SUCCESS);

  //  int gpu_percent_usage[n_gpus];

  for (int i = 0; i < *n_gpus; ++i) {
    nvmlDevice_t device;
    nvmlReturn_t rv;
    rv = nvmlDeviceGetHandleByIndex(i, &device);
    assert(rv == NVML_SUCCESS);
    nvmlUtilization_t utilization;
    rv = nvmlDeviceGetUtilizationRates(device, &utilization);
    assert(rv == NVML_SUCCESS);
    gpu_percent_usage[i] = utilization.gpu;
    nvmlMemory_t memory;
    rv = nvmlDeviceGetMemoryInfo(device, &memory);
    assert(rv == NVML_SUCCESS);
    gpu_total_memory[i] = memory.total;
    rv = nvmlDeviceGetName(device, gpu_name[i], 30);
    assert(rv == NVML_SUCCESS);
  }

}
