/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <cublas_v2.h>
#include <curand_kernel.h>

namespace H2O4GPU {
namespace KMeans {
namespace kernel {

__global__ void setup_random_states(int _seed, curandState *_state,
                                    size_t _size) {
  int id = threadIdx.x + blockIdx.x * threadIdx.x;
  /* Each thread gets same seed, a different sequence
     number, no offset */
  if (id < _size)
    curand_init(_seed, id, 0, &_state[id]);
}

__global__ void generate_uniform_kernel(float *_res,
                                        curandState *_state,
                                        int _size) {
    int idx = threadIdx.x + blockIdx.x * threadIdx.x;
    if (idx < _size) {
      float x;
      curandState local_state = _state[idx];
      x = curand_uniform(&local_state);
      _state[idx] = local_state;
      _res[idx] = x;
    }
}

__global__ void generate_uniform_kernel(double *_res,
                                        curandState *_state,
                                        int _size) {
    int idx = threadIdx.x + blockIdx.x * threadIdx.x;
    if (idx < _size) {
      double x;
      curandState local_state = _state[idx];
      x = curand_uniform_double(&local_state);
      _state[idx] = local_state;
      _res[idx] = x;
    }
}

__global__ void generate_uniform_kernel(int *_res,
                                        curandState *_state,
                                        int _size) {
    int idx = threadIdx.x + blockIdx.x * threadIdx.x;
    if (idx < _size) {
      int x;
      curandState local_state = _state[idx];
      x = (int) curand_uniform_double(&local_state);
      _state[idx] = local_state;
      _res[idx] = x;
    }
}

}  // namespace kernel
}  // namespace KMeans
}  // namespace H2O4GPU