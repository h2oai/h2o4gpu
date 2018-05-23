/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 ************************************************************************/
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <assert.h>

#include <cuda_runtime_api.h>
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
  cudaDeviceProp prop;
  cudaError_t error;
  error = cudaGetDeviceProperties(&prop, d_idx);
  if(error != cudaSuccess){
    std::cerr << "cudaGetDeviceProperties got error code " << (int)error << std::endl;
    std::cout << "cudaGetDeviceProperties got error code " << (int)error << std::endl;
    return(error);
  }
  *major = prop.major;
  *minor = prop.minor;
  *ratioperf = prop.singleToDoublePrecisionPerfRatio;
  return (0);
}


void get_gpu_info_c(unsigned int *n_gpus, int *gpu_percent_usage, unsigned long long *gpu_total_memory,
 unsigned long long *gpu_free_memory,
 char **gpu_name,
 int *majors, int *minors,
 unsigned int *num_pids, unsigned int *pids, unsigned long long *usedGpuMemorys) {

  bool verbose=false;
  if(verbose){
    std::cerr << "inside get_gpu_info_c c function" << std::endl;
  }

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

    if(verbose){
      std::cerr << "i=" << i << " usage=" << gpu_percent_usage[i] << std::endl;
    }

    nvmlMemory_t memory;
    rv = nvmlDeviceGetMemoryInfo(device, &memory);
    assert(rv == NVML_SUCCESS);
    gpu_total_memory[i] = memory.total;
    gpu_free_memory[i] = memory.free;

    rv = nvmlDeviceGetName(device, gpu_name[i], 30);
    assert(rv == NVML_SUCCESS);

#if (CUDART_VERSION >= 9000)
    rv = nvmlDeviceGetCudaComputeCapability(device, &majors[i], &minors[i]);
    assert(rv == NVML_SUCCESS);
#else
    int ratioperf;
    // if this gets called in process, it creates cuda context,
    // but can't assume user wants that to happen, so cripple the function for now
    //get_compute_capability(i, &majors[i], &minors[i], &ratioperf);
    majors[i] = -1;
    minors[i] = -1;
#endif

    if(verbose){
      std::cerr << "i=" << i << " majors=" << majors[i] << " minors=" << minors[i] << std::endl;
    }

    unsigned int max_pids=2000;
    unsigned int infoCount;
    //rv = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, NULL);
    //assert(rv == NVML_SUCCESS);
    infoCount = max_pids;
    nvmlProcessInfo_t infos[infoCount];
    unsigned int num_pid_local;
    num_pids[i] = infoCount;
    rv = nvmlDeviceGetComputeRunningProcesses(device, &num_pids[i], infos);
    assert(rv == NVML_SUCCESS);
    if(num_pids[i] > max_pids){
       std::cerr << "Too many pids: " << num_pids[i] << " .  Increase max_pids: " << max_pids << std::endl;
       assert(num_pids[i] <= max_pids);
    }
    for (unsigned int pidi=0; pidi < num_pids[i]; pidi++) {
        pids[pidi + i * max_pids] = infos[pidi].pid;
        usedGpuMemorys[pidi + i * max_pids] = infos[pidi].usedGpuMemory;

        if(verbose){
          std::cerr << "i=" << i << " pidi=" << pidi << " pids=" << pids[pidi + i * max_pids] << " gpumemory=" << usedGpuMemorys[pidi + i * max_pids] << std::endl;
        }

    }
  }

}
