/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef _CUDA_UTILS2_H
#define _CUDA_UTILS2_H

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

#define OK(cmd) {                                       \
    cudaError_t e = cmd;                                \
    if( e != cudaSuccess ) {                            \
      printf("Cuda failure %s:%d '%s'\n",               \
             __FILE__,__LINE__,cudaGetErrorString(e));  \
      exit(EXIT_FAILURE);                               \
    }                                                   \
}

#define SYNC_OK(cmd) {                                  \
    safe_cuda(cmd);                                            \
    safe_cuda(cudaDeviceSynchronize());                        \
    safe_cuda(cudaGetLastError());                             \
 } 

#ifdef SYNC
#define CUDACHECK(cmd) {                                \
    SYNC_OK(cmd);                                       \
 }
#endif

#ifndef SYNC
#define CUDACHECK(cmd) {                                \
    safe_cuda(cmd);                                            \
}
#endif


#endif
