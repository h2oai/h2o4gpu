/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
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


#endif
