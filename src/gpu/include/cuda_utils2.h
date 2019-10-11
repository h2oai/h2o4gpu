/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef _CUDA_UTILS2_H
#define _CUDA_UTILS2_H

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define CUDACHECK(cmd)                                        \
  do {                                                        \
    cudaError_t e = cmd;                                      \
    if (e != cudaSuccess) {                                   \
      printf("Cuda failure %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                          \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define OK(cmd)                                               \
  do {                                                        \
    cudaError_t e = cmd;                                      \
    if (e != cudaSuccess) {                                   \
      printf("Cuda failure %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                          \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define SYNC_OK(cmd)             \
  do {                           \
    OK(cmd);                     \
    OK(cudaDeviceSynchronize()); \
    OK(cudaGetLastError());      \
  } while (0)

#ifdef SYNC
#define CUDACHECK(cmd) \
  { SYNC_OK(cmd); }
#endif

#ifndef SYNC
#define CUDACHECK(cmd) \
  { OK(cmd); }
#endif

template <class T>
inline __host__ void compute1DInvokeConfig(size_t n, int *minGridSize,
                                           int *blockSize, T func,
                                           size_t dynamicSMemSize = 0,
                                           int blockSizeLimit = 0) {
  OK(cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                        dynamicSMemSize, blockSizeLimit));
  *minGridSize = int((n + *blockSize - 1) / *blockSize);
}

#endif
