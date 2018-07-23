/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <thrust/random.h>
#include <random>

#include <curand_kernel.h>

#include "Generator.hpp"
#include "KmMatrix.hpp"
#include "utils.cuh"


namespace H2O4GPU {
namespace KMeans {

namespace kernel {
// Split the definition to avoid multiple definition.
__global__ void setup_random_states(int _seed, curandState *_state,
                                    size_t _size);

__global__ void generate_uniform_kernel(float *_res,
                                        curandState *_state,
                                        int _size);

__global__ void generate_uniform_kernel(double *_res,
                                        curandState *_state,
                                        int _size);
}

template <typename T>
struct UniformGenerator : public GeneratorBase<T> {
 private:
  // FIXME: Use KmMatrix
  curandState *dev_states_;
  size_t size_;
  // FIXME: Cache random_numbers_ in a safer way.
  KmMatrix<T> random_numbers_;
  int seed_;

  void initialize (size_t _size) {
    size_ = _size;
    random_numbers_ = KmMatrix<T> (1, size_);

    if (dev_states_ != nullptr) {
      CUDA_CHECK(cudaFree(dev_states_));
    }
    CUDA_CHECK(cudaMalloc((void **)&dev_states_, size_ * sizeof(curandState)));
    kernel::setup_random_states<<<div_roundup(size_, 256), 256>>>(
        seed_, dev_states_, size_);
  }

 public:
  UniformGenerator() : dev_states_ (nullptr), size_ (0) {
    std::random_device rd;
    seed_ = rd();
  }

  UniformGenerator (size_t _size, int _seed) {
    if (_size == 0) {
      M_ERROR("Zero size for generate is not allowed.");
    }
    initialize(_size);
  }

  UniformGenerator(int _seed) :
      seed_(_seed), dev_states_(nullptr), size_ (0) {}

  ~UniformGenerator () {
    if (dev_states_ != nullptr) {
      CUDA_CHECK(cudaFree(dev_states_));
    }
  }

  UniformGenerator(const UniformGenerator<T>& _rhs) = delete;
  UniformGenerator(UniformGenerator<T>&& _rhs) = delete;
  void operator=(const UniformGenerator<T>& _rhs) = delete;
  void operator=(UniformGenerator<T>&& _rhs) = delete;

  KmMatrix<T> generate() override {
    kernel::generate_uniform_kernel<<<div_roundup(size_, 256), 256>>>
        (random_numbers_.k_param().ptr, dev_states_, size_);
    return random_numbers_;
  }

  KmMatrix<T> generate(size_t _size) override {
    if (_size == 0) {
      M_ERROR("Zero size for generate is not allowed.");
    }
    if (_size != size_) {
      initialize(_size);
    }
    return generate();
  }
};
  
}  // H2O4GPU
}  // KMeans