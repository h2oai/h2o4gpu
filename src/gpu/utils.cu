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

#include "../common/logger.h"

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


int get_gpu_info_c(int verbose, unsigned int *n_gpus, int *gpu_percent_usage, unsigned long long *gpu_total_memory,
 unsigned long long *gpu_free_memory,
 char **gpu_name,
 int *majors, int *minors,
 unsigned int *num_pids, unsigned int *pids, unsigned long long *usedGpuMemorys,
 unsigned int *num_pids_usage, unsigned int *pids_usage, unsigned long long *usedGpuUsage) {

  log_verbose(verbose, "Inside get_gpu_info_c c function");

  nvmlReturn_t rv;
  rv = nvmlInit();

  if (rv != NVML_SUCCESS) {
    log_fatal(verbose, "Failed to initialize NVML: %s", nvmlErrorString(rv));
    return 1;
  }

  log_verbose(verbose, "Initialized NVML.");

  //unsigned int n_gpus;
  rv = nvmlDeviceGetCount(n_gpus);

  if (rv != NVML_SUCCESS) {
    log_fatal(verbose, "Failed to get device count: %s", nvmlErrorString(rv));
    return 1;
  }

  //  int gpu_percent_usage[n_gpus];

  log_verbose(verbose, "Getting info for %u devices", *n_gpus);

  for (int i = 0; i < *n_gpus; ++i) {
    nvmlDevice_t device;
    nvmlReturn_t rv;
    rv = nvmlDeviceGetHandleByIndex(i, &device);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get device %d by handle: %s", i, nvmlErrorString(rv));
      return 1;
    }

    nvmlUtilization_t utilization;
    rv = nvmlDeviceGetUtilizationRates(device, &utilization);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get device %d utilization: %s", i, nvmlErrorString(rv));
      return 1;
    }

    gpu_percent_usage[i] = utilization.gpu;

    log_verbose(verbose, "i= %d usage= %d", i, gpu_percent_usage[i]);

    nvmlMemory_t memory;
    rv = nvmlDeviceGetMemoryInfo(device, &memory);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get device %d memory info: %s", i, nvmlErrorString(rv));
      return 1;
    }

    log_verbose(verbose, "Device memory %i obtained.", i);

    gpu_total_memory[i] = memory.total;
    gpu_free_memory[i] = memory.free;

    rv = nvmlDeviceGetName(device, gpu_name[i], 100);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get device %d name: %s", i, nvmlErrorString(rv));
      return 1;
    }

#if (CUDART_VERSION >= 9000)
    rv = nvmlDeviceGetCudaComputeCapability(device, &majors[i], &minors[i]);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get device %d cuda compute capability: %s", i, nvmlErrorString(rv));
        return 1;
    }

#else
    int ratioperf;
    // if this gets called in process, it creates cuda context,
    // but can't assume user wants that to happen, so cripple the function for now
    //get_compute_capability(i, &majors[i], &minors[i], &ratioperf);
    majors[i] = -1;
    minors[i] = -1;
#endif

    log_verbose(verbose, "i= %d majors= %d", i, majors[i]);

    unsigned int max_pids=2000;
    unsigned int infoCount;
    infoCount = max_pids;
    nvmlProcessInfo_t infos[infoCount];
    num_pids[i] = infoCount;
    rv = nvmlDeviceGetComputeRunningProcesses(device, &num_pids[i], infos);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get device %d running processes: %s", i, nvmlErrorString(rv));
      return 1;
    }

    if(num_pids[i] > max_pids){
      log_debug(verbose, "Too many pids: %u. Increase max_pids: %u.", num_pids[i], max_pids );
      return 1;
    }
    for (unsigned int pidi=0; pidi < num_pids[i]; pidi++) {
      pids[pidi + i * max_pids] = infos[pidi].pid;
      usedGpuMemorys[pidi + i * max_pids] = infos[pidi].usedGpuMemory;

      log_verbose(verbose, "i=%d pidi=%u pids=%u gpumemory=%llu", i, pidi, pids[pidi + i * max_pids], usedGpuMemorys[pidi + i * max_pids]);
    }

#if (CUDART_VERSION >= 9000)
    nvmlProcessUtilizationSample_t utilization_perprocess[infoCount];
    unsigned int processSamplesCount;
    unsigned long long lastSeenTimeStamp = 0;

    num_pids_usage[i] = infoCount;
    rv = nvmlDeviceGetProcessUtilization(device, utilization_perprocess, &num_pids_usage[i], lastSeenTimeStamp);

    if (rv != NVML_SUCCESS) {
      log_fatal(verbose, "Failed to get per-process device %d utilization: %s", i, nvmlErrorString(rv));
      return 1;
    }

    if(num_pids_usage[i] > max_pids){
      log_debug(verbose, "Too many pids: %u. Increase max_pids: %u.", num_pids_usage[i], max_pids );
      return 1;
    }

    for (unsigned int pidi=0; pidi < num_pids_usage[i]; pidi++) {
      pids_usage[pidi + i * max_pids] = utilization_perprocess[pidi].pid;
      usedGpuUsage[pidi + i * max_pids] = utilization_perprocess[pidi].smUtil;

      log_verbose(verbose, "i=%d pidi=%u pids=%u gpuusage=%llu", i, pidi, pids_usage[pidi + i * max_pids], usedGpuUsage[pidi + i * max_pids]);
    }
#else
    pids_usage[i] = 0;
    usedGpuUsage[i] = 0;
#endif
  }

  return 0;
}
