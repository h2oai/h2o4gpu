#ifndef KM_CONFIG_H_
#define KM_CONFIG_H_

#define USE_CUDA() 1

#include "stdio.h"

// Matrix host dev
#define M_HOSTDEV __host__ __device__
#define M_DEV __device__
#define M_DEVINLINE __device__ __forceinline__
#define M_HOSTDEVINLINE __host__ __device__ __forceinline__

#define CUDA_CHECK(cmd) do {                            \
    cudaError_t e = cmd;                                \
    if( e != cudaSuccess ) {                            \
      printf("Cuda failure %s:%d '%s'\n",               \
             __FILE__,__LINE__,cudaGetErrorString(e));  \
      fflush( stdout );                                 \
      exit(EXIT_FAILURE);                               \
    }                                                   \
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

#define M_ERROR(msg)                            \
  printf("%s\n\t in %s, %u, %s\n", msg, __FILE__, __LINE__, __func__);  \
  abort();

#endif  // KM_CONFIG_H_
