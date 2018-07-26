/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <thrust/random.h>
#include <random>

#include "Generator.hpp"
#include "KmMatrix.hpp"
#include "../../utils/utils.cuh"


namespace h2o4gpu {
namespace Matrix {

template <typename T>
struct UniformGenerator : public GeneratorBase<T> {
 private:
  size_t size_;
  // FIXME: Cache random_numbers_ in a safer way.
  KmMatrix<T> random_numbers_;
  int seed_;

  void initialize (size_t _size) {
    size_ = _size;
    random_numbers_ = KmMatrix<T> (1, size_);
  }

 public:
  UniformGenerator() : size_ (0) {
    std::random_device rd;
    seed_ = rd();
  }

  UniformGenerator (size_t _size, int _seed) : seed_(_seed) {
    if (_size == 0) {
      h2o4gpu_error("Zero size for generate is not allowed.");
    }
    initialize(_size);
  }

  UniformGenerator(int _seed) :
      seed_(_seed), size_ (0) {}

  ~UniformGenerator () {}

  UniformGenerator(const UniformGenerator<T>& _rhs) = delete;
  UniformGenerator(UniformGenerator<T>&& _rhs) = delete;
  void operator=(const UniformGenerator<T>& _rhs) = delete;
  void operator=(UniformGenerator<T>&& _rhs) = delete;

  KmMatrix<T> generate() override {
    thrust::device_ptr<T> rn_ptr (random_numbers_.dev_ptr());
    thrust::transform(
        thrust::make_counting_iterator((size_t)0),
        thrust::make_counting_iterator(size_),
        rn_ptr,
        [=] __device__ (int idx) {
          thrust::default_random_engine rng(seed_);
          thrust::uniform_real_distribution<T> dist;
          rng.discard(idx);
          return dist(rng);
        });

    return random_numbers_;
  }

  KmMatrix<T> generate(size_t _size) override {
    if (_size == 0) {
      h2o4gpu_error("Zero size for generate is not allowed.");
    }
    if (_size != size_) {
      initialize(_size);
    }
    return generate();
  }
};

}  // namespace h2o4gpu
}  // namespace Matrix