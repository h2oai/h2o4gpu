#ifndef UTIL_CUH_
#define UTIL_CUH_

#define CHECKERR(err_string) \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s (error code %s)!\n", err_string, \
      cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

#endif  // UTIL_CUH_

