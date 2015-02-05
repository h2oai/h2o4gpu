#ifndef CML_RAND_CUH_
#define CML_RAND_CUH_

#include <time.h>
#include <curand_kernel.h>

#include "cml/cml_defs.cuh"
#include "cml/cml_utils.cuh"

namespace cml {

namespace {
__global__ void setup_kernel(curandState *state, unsigned long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate(curandState *globalState, double *data, size_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < size; i += gridDim.x * blockDim.x)
    data[i] = curand_uniform_double(&globalState[tid]);
}

__global__ void generate(curandState *globalState, float *data, size_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < size; i += gridDim.x * blockDim.x)
    data[i] = curand_uniform(&globalState[tid]);
}

}  // namespace

template <typename T>
void rand(T *x, size_t size) {
  size_t num_rand = min(static_cast<unsigned int>(size), kMaxGridSize);
  curandState* devStates;
  cudaMalloc(&devStates, num_rand * sizeof(curandState));
  
  // Setup seeds.
  int grid_dim = calc_grid_dim(num_rand, kBlockSize);
  setup_kernel<<<grid_dim, kBlockSize>>>(devStates, time(NULL));

  // Generate random numbers.
  generate<<<grid_dim, kBlockSize>>>(devStates, x, size);
}

}  // namespace cml

#endif  // CML_RAND_CUH_

