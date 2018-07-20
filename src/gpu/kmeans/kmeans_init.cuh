/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#ifndef KMEANS_INIT_H_
#define KMEANS_INIT_H_

#include <cublas_v2.h>
#include <curand_kernel.h>

#include "KmMatrix/KmConfig.h"
#include "KmMatrix/KmMatrix.hpp"
#include "KmMatrix/utils.cuh"

namespace H2O4GPU{
namespace KMeans {

// Work around for shared memory
// https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name
template <typename T>
struct KmShardMem;

template <>
struct KmShardMem<float> {
  __device__ float * ptr() {
    extern __shared__ __align__(sizeof(float)) float s_float[];
    return s_float;
  }
};

template <>
struct KmShardMem<double> {
  __device__ double * ptr() {
    extern __shared__ __align__(sizeof(double)) double s_double[];
    return s_double;
  }
};

template <>
struct KmShardMem<int> {
  __device__ int * ptr() {
    extern __shared__ __align__(sizeof(int)) int s_int[];
    return s_int;
  }
};

template <typename T>
struct kMParam {
  T* ptr;
  size_t rows;
  size_t cols;

  kMParam(T* _ptr, size_t _rows, size_t _cols) :
      ptr (_ptr), rows (_rows), cols (_cols) {}
  kMParam(size_t _rows, size_t _cols):
      rows (_rows), cols (_cols) {}
  kMParam(size_t _cols) : cols (_cols) {}
};

template <typename T>
struct kVParam {
  T* ptr;
  size_t size;
  kVParam(T* _ptr, size_t _size) : ptr(_ptr), size(_size) {}
};

namespace kernel {

__global__ void setup_random_states(curandState *state, size_t size);
__global__ void generate_uniform_kernel(float *_res,
                                        curandState *_state,
                                        int _size);
__global__ void generate_uniform_kernel(double *_res,
                                        curandState *_state,
                                        int _size);
}

template <typename T>
struct UniformGenerator {
  // private:

  // FIXME: Use KmMatrix
  curandState *dev_states_;
  size_t size_;
  // FIXME: Cache random_numbers_ in a safer way.
  KmMatrix<T> random_numbers_;

  void initialize (size_t _size) {
    size_ = _size;
    random_numbers_ = KmMatrix<T> (1, size_);

    if (dev_states_ != nullptr) {
      CUDA_CHECK(cudaFree(dev_states_));
    }
    CUDA_CHECK(cudaMalloc((void **)&dev_states_, size_ * sizeof(curandState)));
    kernel::setup_random_states<<<div_roundup(size_, 256), 256>>>(
        dev_states_, size_);
  }

  // public:
  UniformGenerator() : dev_states_ (nullptr), size_ (0){}

  UniformGenerator (size_t _size) {
    if (_size == 0) {
      M_ERROR("Zero size for generate is not allowed.");
    }
    initialize(_size);
  }

  ~UniformGenerator () {
    if (dev_states_ != nullptr) {
      CUDA_CHECK(cudaFree(dev_states_));
    }
  }

  UniformGenerator(const UniformGenerator<T>& _rhs) = delete;
  UniformGenerator(UniformGenerator<T>&& _rhs) = delete;
  void operator=(const UniformGenerator<T>& _rhs) = delete;
  void operator=(UniformGenerator<T>&& _rhs) = delete;

  KmMatrix<T> generate() {
    kernel::generate_uniform_kernel<<<div_roundup(size_, 256), 256>>>
        (random_numbers_.k_param().ptr, dev_states_, size_);
    return random_numbers_;
  }
  KmMatrix<T> generate(size_t _size) {
    if (_size == 0) {
      M_ERROR("Zero size for generate is not allowed.");
    }
    if (_size != size_) {
      initialize(_size);
    }
    return generate();
  }
};

/*
 * Base class used for all K-Means initialization algorithms.
 */
template <typename T>
class KmeansInitBase {
 public:
  virtual ~KmeansInitBase() {}
  /*
   * Select k centroids from data.
   *
   * @param data data points stored in row major matrix.
   * @param k number of centroids.
   */
  virtual KmMatrix<T> operator()(KmMatrix<T>& data, size_t k) = 0;
};

/*
 * Each instance of KmeansLlInit corresponds to one dataset, if a new data set
 * is used, users need to create a new instance.
 *
 * k-means|| algorithm based on the paper:
 * <a href="http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf">
 * Scalable K-Means++
 * </a>
 *
 * @tparam Data type, supported types are float and double.
 */
template <typename T>
struct KmeansLlInit : public KmeansInitBase<T> {
 private:
  T over_sample_;
  int seed_;
  int k_;
  UniformGenerator<T> uniform_dist;

  // Buffer like variables
  // store the self dot product of each data point
  KmMatrix<T> data_dot_;
  // store distances between each data point and centroids
  KmMatrix<T> distance_pairs_;

  KmMatrix<T> probability(KmMatrix<T>& data, KmMatrix<T>& centroids);
 public:
  // sample_centroids should not be part of the interface, but following error
  // is generated when put in private section:
  // The enclosing parent function ("sample_centroids") for an extended
  // __device__ lambda cannot have private or protected access within its class
  KmMatrix<T> sample_centroids(KmMatrix<T>& data, KmMatrix<T>& centroids);

  /*
   * Initialize KmeansLlInit algorithm, with default:
   *  over_sample = 1.5,
   *  seed = 0,
   */
  KmeansLlInit () :
      over_sample_ (1.5f), seed_ (0), k_ (0) {}

  /*
   * Initialize KmeansLlInit algorithm, with default:
   *  seed = 0,
   *
   * @param over_sample over_sample rate.
   *    \f$p_x = over_sample \times \frac{d^2(x, C)}{\Phi_X (C)}\f$
   *    Note that when \f$over_sample != 1\f$, the probability for each data
   *    point doesn't add to 1.
   */
  KmeansLlInit (T _over_sample) :
      over_sample_ (_over_sample), seed_ (0), k_ (0) {}

  /*
   * Initialize KmeansLlInit algorithm.
   *
   * @param seed Seed used to generate threshold for sampling centroids.
   * @param over_sample over_sample rate.
   *    p_x = over_sample \times \frac{d^2(x, C)}{\Phi_X (C)}
   */
  KmeansLlInit (int _seed, T _over_sample) :
      seed_(_seed), seed_(_seed), k_(0) {}

  virtual ~KmeansLlInit () override {}

  /*
   * Select k centroids from data.
   * @param data data points stored in row major matrix.
   * @param k number of centroids.
   */
  KmMatrix<T> operator()(KmMatrix<T>& data, size_t k) override;
};


// FIXME: Make kmeans++ a derived class of KmeansInitBase

}  // namespace Kmeans
}  // namespace H2O4GPU

#endif  // KMEANS_INIT_H_