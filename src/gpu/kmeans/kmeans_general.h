/*!
 * Copyright 2017-2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include "../../common/logger.h"
#include "stdio.h"
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

#define CUBLAS_CHECK(cmd) do {                                          \
    cublasStatus_t status = cmd;                                        \
    if ( status != CUBLAS_STATUS_SUCCESS) {                             \
      const char* errmsg = nullptr;                                     \
      switch(status) {                                                  \
        case CUBLAS_STATUS_NOT_INITIALIZED:                             \
          errmsg = "library not initialized";                           \
          break;                                                        \
                                                                        \
        case CUBLAS_STATUS_ALLOC_FAILED:                                \
          errmsg = "resource allocation failed";                        \
          break;                                                        \
                                                                        \
        case CUBLAS_STATUS_INVALID_VALUE:                               \
          errmsg = "an invalid numeric value was used as an argument";  \
          break;                                                        \
                                                                        \
        case CUBLAS_STATUS_ARCH_MISMATCH:                               \
          errmsg = "an absent device architectural feature is required"; \
          break;                                                        \
                                                                        \
        case CUBLAS_STATUS_MAPPING_ERROR:                               \
          errmsg = "an access to GPU memory space failed";              \
          break;                                                        \
                                                                        \
        case CUBLAS_STATUS_EXECUTION_FAILED:                            \
          errmsg = "the GPU program failed to execute";                 \
          break;                                                        \
                                                                        \
        case CUBLAS_STATUS_INTERNAL_ERROR:                              \
          errmsg = "an internal operation failed";                      \
          break;                                                        \
                                                                        \
        default:                                                        \
          errmsg = "unknown error";                                     \
          break;                                                        \
      }                                                                 \
      printf("%s", errmsg);                                             \
    }                                                                   \
  } while (false)
