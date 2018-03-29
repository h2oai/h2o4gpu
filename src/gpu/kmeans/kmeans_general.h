/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <thrust/device_vector.h>
#include "../../common/logger.h"
#include <iostream>

#define MAX_NGPUS 16

#define VERBOSE 0
#define CHECK 1
#define DEBUGKMEANS 0

// TODO(pseudotensor): Avoid throw for python exception handling.  Need to avoid all exit's and return exit code all the way back.
#define gpuErrchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__);
#define safe_cublas(ans) throw_on_cublas_error((ans), __FILE__, __LINE__);

#define CUDACHECK(cmd) do {                           \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));\
      fflush( stdout );                               \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

template<typename T>
void printVector(const char *msg, thrust::device_vector<T> dev_vec) {
  thrust::host_vector<T> host_vec = dev_vec;
  std::cout << "\n>>> " << msg << std::endl;
  std::cout.precision(17);
  for (int i = 0; i < host_vec.size(); ++i) {
    std::cout << std::fixed << host_vec[i] << " ";
  }
  std::cout << std::endl;
}
