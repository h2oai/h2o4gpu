#ifndef CML_RAND_CUH_
#define CML_RAND_CUH_

#include <random>

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
void rand(T *x, size_t size, bool use_gpu) {
  if (use_gpu) {
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
  } else {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> dist(static_cast<T>(0),
        static_cast<T>(1));

    std::vector<T> x_temp(size);

    for (size_t i = 0; i < size; ++i) {
      x_temp[i] = dist(generator);
    }

    cudaMemcpy(x, x_temp.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  }
}

}  // namespace cml

#endif  // CML_RAND_CUH_

