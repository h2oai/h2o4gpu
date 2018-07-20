/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <cub/device/device_select.cuh>

#include <random>
#include <limits>
#include <string>

#include <stdio.h>

#include "kmeans_init.cuh"

#include "KmMatrix/KmMatrix.hpp"
#include "KmMatrix/utils.cuh"
#include "KmMatrix/GpuInfo.cuh"
#include "KmMatrix/blas.cuh"


namespace H2O4GPU {
namespace KMeans {

namespace kernel {

__global__ void setup_random_states(curandState *state, size_t size) {
  int id = threadIdx.x + blockIdx.x * threadIdx.x;
  /* Each thread gets same seed, a different sequence
     number, no offset */
  if (id < size)
    curand_init(1234, id, 0, &state[id]);
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

/*
 * Compute min value for each row.
 * @tparam T Numeric type of the data
 * @param _res The output matrix with shape m x 1
 * @param _val The input matrix with shape m x n
 */
template <typename T, size_t BATCH_SIZE=64>
__global__ void row_min_sequential(kParam<T> _res, kParam<T> _val) {

  size_t idx = global_thread_idx();
  size_t stride = grid_stride_x () * _val.cols;

  size_t n_batches = div_roundup(_val.cols, 128);

  for (size_t i = idx; i < _val.size(); i += stride) {
    T min = std::numeric_limits<T>::max();

    for (size_t j = 0; j < _val.cols; ++j) {
      T tmp = _val.ptr[i+j];
      if (tmp < min)
        min = tmp;
      _res.ptr[idx] = tmp;
    }
  }
}

}  // namespace kernel


template <typename T>
struct DotOp {
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _val) {
    this->dot(_res, _val, _val);
  }
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs,
           KmMatrix<T>& _rhs) {
    constexpr T alpha = 1.0;
    constexpr T beta = 1.0;
    cublasHandle_t handle = GpuInfo::ins().cublas_handle();
    Blas::gemm(handle,
               CUBLAS_OP_T, CUBLAS_OP_N,  // FIXME
               _lhs.rows(), _rhs.cols(), _lhs.cols(),
               &alpha,
               _lhs.dev_ptr(), _lhs.cols(),
               _rhs.dev_ptr(), _rhs.cols(),
               &beta,
               _res.dev_ptr(), _res.cols());
  }
};

template <typename T>
struct VecBatchDotOp {
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _val) {
    this->dot(_res, _val, _val);
  }
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs, KmMatrix<T>& _rhs) {
    constexpr T alpha = 1.0;
    constexpr T beta = 1.0;
    cublasHandle_t handle = GpuInfo::ins().cublas_handle();
    Blas::gemm_strided_batched(
        handle,
        // k-means use row major, so transpose the second vector.
        CUBLAS_OP_N, CUBLAS_OP_T,
        1, 1, _rhs.cols(),  // m, n, k
        &alpha,
        _lhs.dev_ptr(), 1, _lhs.cols(),
        _rhs.dev_ptr(), 1, _rhs.cols(),
        &beta,
        _res.dev_ptr(), _res.cols(), 1,  // c should be columun vector
        _lhs.rows());
  }
};

// FIXME: Using struct for operations is just keeping the possibility to create
// some unified operations for KmMatrix. For example, let KmMatrix
// inherit those left associative ops, or create a inferface for elementwise
// operations.
template <typename T>
struct SumOp {
  T sum(KmMatrix<T>& _val) {
    T* raw_ptr = _val.dev_ptr();
    thrust::device_ptr<T> ptr (raw_ptr);
    T res = thrust::reduce(ptr, ptr + _val.size(), (T)0, thrust::plus<T>());
    return res;
  }
};

template <typename T>
struct MeanOp {
  T mean(KmMatrix<T>& _val) {
    T res = SumOp<T>().sum(_val);
    return res;
  }
};

template <typename T>
struct MulOp {
  void mul(KmMatrix<T>& _res, KmMatrix<T>& _lhs, T _rhs) {
    cublasHandle_t handle = GpuInfo::ins().cublas_handle();
    Blas::axpy(
        handle, _lhs.size(),  // handle, n
        &_rhs,                // alpha
        _lhs.dev_ptr(), 1,
        _res.dev_ptr(), 1);
  }
};

template <typename T>
struct MinOp {

  void min(KmMatrix<T>& _res, KmMatrix<T>& _val, KmMatrixDim _dim) {
    size_t blocks = GpuInfo::ins().blocks(32);
    if (_dim == KmMatrixDim::ROW) {
      kernel::row_min_sequential<<<blocks, 256, sizeof(T)*_val.cols()>>>(
          _res.k_param(), _val.k_param());
    } else {
      // FIXME
      M_ERROR("Not implemented");
    }
  }
};

namespace kernel {
// X^2 + Y^2, here only calculates the + operation.
template <typename T>
__global__ void construct_distance_pairs_kernel(
    kParam<T> _distance_pairs,
    kParam<T> _data_dots, kParam<T> _centroids_dots) {

  size_t idx = global_thread_idx();  // indexing data
  size_t idy = global_thread_idy();  // indexing centroids

  // FIXME: Is using shared memory necessary?

  size_t stride_x = grid_stride_x () * _data_dots.cols;
  // strides only for data.
  for (size_t i = idx; i < _data_dots.rows; i += stride_x) {
    if (i < _data_dots.rows && idy < _centroids_dots.rows ) {
      // i + idy: x^2 + y^2 between i^th data (a.k.a x) and idy^th
      // centroid (a.k.a y)
      _distance_pairs.ptr[i + idy] =
          _data_dots.ptr[idx] + _centroids_dots.ptr[idy];
    }
  }
}

}  // namespace kernel

// Extracted as an independent Op for k-means use.
template <typename T>
struct PairWiseDistanceOp {
  KmMatrix<T> data_dot_;
  KmMatrix<T> centroids_dot_;
  KmMatrix<T> distance_pairs_;

  bool initialized_;

  void initialize(size_t _n_data, size_t k, size_t _dim) {
    // FIXME
  }

  PairWiseDistanceOp () : initialized_(false) {}

  PairWiseDistanceOp (KmMatrix<T>& _data_dot, KmMatrix<T>& _centroids_dot,
                      KmMatrix<T>& _distance_pairs) :
      data_dot_(_data_dot), centroids_dot_(_centroids_dot),
      distance_pairs_(_distance_pairs), initialized_(true) {
    data_dot_.set_name ("data dot");
    centroids_dot_.set_name ("centroids_dot");
    distance_pairs_.set_name ("distance pairs");
  }

  KmMatrix<T> operator()(KmMatrix<T>& _data, KmMatrix<T>& _centroids) {

    kernel::construct_distance_pairs_kernel<<<
        dim3(GpuInfo::ins().blocks(32), div_roundup(_centroids.rows(), 16)),
        dim3(32, 16)>>>(  // FIXME: Tune this.
            distance_pairs_.k_param(),
            data_dot_.k_param(),
            centroids_dot_.k_param());

    CUDA_CHECK(cudaGetLastError());

    cublasHandle_t handle = GpuInfo::ins().cublas_handle();

    T alpha = -2.0;
    T beta = 1.0;

    Blas::gemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        // n, d, d/k
        _data.rows(), _data.cols(), _data.cols(),
        &alpha,
        _data.dev_ptr(), _data.rows(),
        _centroids.dev_ptr(), _centroids.cols(),
        &beta,
        distance_pairs_.dev_ptr(), distance_pairs_.rows());

    return distance_pairs_;
  }
};


template <typename T>
KmMatrix<T> KmeansLlInit<T>::probability(
    KmMatrix<T>& _data, KmMatrix<T>& _centroids) {

  KmMatrix<T> centroids_dot (_centroids.rows(), 1);

  VecBatchDotOp<T>().dot(centroids_dot, _centroids);

  // FIXME: Time this
  distance_pairs_ = KmMatrix<T>(_data.rows(), _centroids.rows());
  PairWiseDistanceOp<T> distance_op (data_dot_, centroids_dot, distance_pairs_);
  distance_pairs_ = distance_op(_data, _centroids);

  KmMatrix<T> min_distances (_data.rows(), 1);

  MinOp<T>().min(min_distances, distance_pairs_, KmMatrixDim::ROW);

  T cost = SumOp<T>().sum(min_distances);

  KmMatrix<T> prob (min_distances.rows(), 1);
  MulOp<T>().mul(prob, min_distances, over_sample_ / cost);

  return prob;
}


template <typename T>
KmMatrix<T> KmeansLlInit<T>::sample_centroids(
    KmMatrix<T>& _data, KmMatrix<T>& _prob) {

  KmMatrix<T> thresholds = uniform_dist.generate(_data.rows());

  T * thresholds_ptr = thresholds.dev_ptr();

  // If use kParam, nvcc complains:
  // identifier "H2O4GPU::KMeans::kParam<double> ::kParam" is undefined in
  // device code.
  T* prob_ptr = _prob.dev_ptr();

  auto prob_iter = thrust::make_counting_iterator(0);
  size_t n_new_centroids = thrust::count_if(
      thrust::device, prob_iter,
      prob_iter + _prob.size(),
      [=] __device__ (int idx) {
        float thresh = thresholds_ptr[idx];
        T prob_x = prob_ptr[idx];
        return prob_x > thresh;
      });

  KmMatrix<T> new_centroids(n_new_centroids, _data.cols());
  thrust::device_ptr<T> new_centroids_ptr (new_centroids.dev_ptr());

  thrust::device_ptr<T> data_ptr (_data.dev_ptr());

  size_t cols = _data.cols();
  // renew iterator
  prob_iter = thrust::make_counting_iterator(0);
  thrust::copy_if(
      thrust::device,
      data_ptr, data_ptr + _data.size(), prob_iter,
      new_centroids_ptr,
      [=] __device__(int idx) {
        size_t row = idx / cols;
        T thresh = thresholds_ptr[row];
        T prob_x = prob_ptr[row];
        return prob_x > thresh;
      });

  return new_centroids;
}

template <typename T>
KmMatrix<T>
KmeansLlInit<T>::operator()(KmMatrix<T>& _data, size_t _k) {

  if (seed_ < 0) {
    std::random_device rd;
    seed_ = rd();
  }
  k_ = _k;

  std::mt19937 generator(0);

  std::uniform_int_distribution<> distribution(0, _data.rows());
  size_t idx = distribution(generator);

  // Calculate X^2 (point-wise)
  data_dot_ = KmMatrix<T>(_data.rows(), 1);
  VecBatchDotOp<T>().dot(data_dot_, _data);

  // First centroid
  KmMatrix<T> centroids = _data.row(idx);

  KmMatrix<T> prob = probability(_data, centroids);

  T cost = SumOp<T>().sum(prob);

  for (size_t i = 0; i < std::log(cost); ++i) {
    prob = probability(_data, centroids);
    KmMatrix<T> new_centroids = sample_centroids(_data, prob);
    centroids = stack(centroids, new_centroids, KmMatrixDim::ROW);
  }

  if (centroids.rows() < k_) {
    // FIXME: When n_centroids < k
    // Get random selection in?
  }

  // FIXME: re-cluster
  // kmeans_plus_plus(centroids);
  return centroids;
}

#define INSTANTIATE(T)                                                  \
  template KmMatrix<T> KmeansLlInit<T>::operator()(                     \
      KmMatrix<T>& _data, size_t _k);                                   \
  template KmMatrix<T> KmeansLlInit<T>::probability(KmMatrix<T>& data,  \
                                                    KmMatrix<T>& centroids); \
  template KmMatrix<T> KmeansLlInit<T>::sample_centroids(               \
      KmMatrix<T>& data, KmMatrix<T>& centroids);                       \

INSTANTIATE(float)
INSTANTIATE(double)
// FIXME: int is not supported due to random kernel

}  // namespace Kmeans
}  // namespace H2O4GPU
