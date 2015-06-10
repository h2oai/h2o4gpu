#ifndef CML_RAND_CUH_
#define CML_RAND_CUH_

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
  size_t num_rand = std::min(size, kMaxGridSize);
  curandState* devStates;
  cudaMalloc(&devStates, num_rand * sizeof(curandState));

  // Setup seeds.
  size_t block_dim = std::min(kBlockSize, num_rand);
  size_t grid_dim = calc_grid_dim(num_rand, block_dim);
  setup_kernel<<<grid_dim, block_dim>>>(devStates, 0);

  // Generate random numbers.
  generate<<<grid_dim, block_dim>>>(devStates, x, size);

  cudaFree(devStates);
}

}  // namespace cml

#endif  // CML_RAND_CUH_

