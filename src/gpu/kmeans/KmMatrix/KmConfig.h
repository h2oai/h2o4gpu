#ifndef KM_CONFIG_H_
#define KM_CONFIG_H_

#define USE_CUDA() 1

// Matrix host dev
#define M_HOSTDEV __host__ __device__
#define M_DEVINLINE __device__ __forceinline__
#define M_HOSTDEVINLINE __host__ __device__ __forceinline__

#define CUDA_CHECK(cmd) do {                          \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));\
      fflush( stdout );                               \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

#endif  // KM_CONFIG_H_
