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

// Wrappers for Eigen matrix and vector
// template <typename T>
// struct EiMatrix;

// template <>
// struct EiMatrix <float> {
//   using type = Eigen::MatrixXf;
// };
// template <>
// struct EiMatrix <double> {
//   using type = Eigen::MatrixXd;
// };
// template <>
// struct EiMatrix<int> {
//   using type = Eigen::MatrixXi;
// };

// template <typename T>
// struct EiVector;
// template <>
// struct EiVector<float> {
//   using type = Eigen::VectorXf;
// };
// template <>
// struct EiVector<double> {
//   using type = Eigen::VectorXd;
// };
// template <>
// struct EiVector<int> {
//   using type = Eigen::VectorXi;
// };

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

#define MA_T(T)                                 \
  typename EiMatrix<T>::type
#define VE_T(T)                                 \
  typename EiVector<T>::type

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
struct Generator {
  // FIXME: Use KmMatrix
  curandState *dev_states_;
  size_t size_;
  // FIXME: Cache random_numbers_ in a safer way.
  KmMatrix<T> random_numbers_;

  Generator (size_t _size) : size_(_size) , random_numbers_(1, _size) {
    CUDA_CHECK(cudaMalloc((void **)&dev_states_, _size *
                          sizeof(curandState)));
    kernel::setup_random_states<<<div_roundup(size_, 256), 256>>>(
        dev_states_, size_);
  }
  ~Generator () {
    CUDA_CHECK(cudaFree(dev_states_));
  }

  KmMatrix<T> generate() {
    kernel::generate_uniform_kernel<<<div_roundup(size_, 256), 256>>>
        (random_numbers_.k_param().ptr, dev_states_, size_);
    return random_numbers_;
  }
};

template <typename T>
class KmeansInitBase {
 public:
  virtual ~KmeansInitBase() {}
  virtual KmMatrix<T> operator()(KmMatrix<T>& data, size_t k) = 0;
};

template <typename T>
struct KmeansLlInit : public KmeansInitBase<T> {
 private:
  double over_sample_;
  int seed_;
  int k_;
  // Buffer like variables
  // store the self dot product of each data point
  KmMatrix<T> data_dot_;
  // store distances between each data point and centroids
  KmMatrix<T> distance_pairs_;

  KmMatrix<T> probability(KmMatrix<T>& data, KmMatrix<T>& centroids);

 public:
  KmeansLlInit () : over_sample_ (2.0), seed_ (0), k_(0) {
    data_dot_.set_name ("data_dot");
    distance_pairs_.set_name ("distance pairs");
  }
  virtual ~KmeansLlInit () override {}

  KmMatrix<T> sample_centroids(KmMatrix<T>& data, KmMatrix<T>& centroids);
  KmMatrix<T> operator()(KmMatrix<T>& data, size_t k) override;
};


// FIXME: Make kmeans++ a derived class of KmeansInitBase

}  // namespace Kmeans
}  // namespace H2O4GPU

#endif  // KMEANS_INIT_H_