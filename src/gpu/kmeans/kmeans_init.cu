/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <thrust/device_vector.h>

#include <cub/device/device_select.cuh>
#include <cub/device/device_histogram.cuh>

#include <random>
#include <limits>
#include <string>

#include <stdio.h>

#include "kmeans_init.cuh"

#include "KmMatrix/KmMatrix.hpp"
#include "KmMatrix/Arith.hpp"
#include "KmMatrix/utils.cuh"
#include "KmMatrix/GpuInfo.cuh"
#include "KmMatrix/blas.cuh"

namespace H2O4GPU {
namespace KMeans {


namespace kernel {
// X^2 + Y^2, here only calculates the + operation.
template <typename T>
__global__ void construct_distance_pairs_kernel(
    kParam<T> _distance_pairs,
    kParam<T> _data_dots, kParam<T> _centroids_dots) {

  size_t idx = global_thread_idx();  // indexing data
  size_t idy = global_thread_idy();  // indexing centroids

  // FIXME: Is using shared memory necessary?

  size_t stride_x = grid_stride_x ();
  // strides only for data.
  for (size_t i = idx; i < _data_dots.rows; i += stride_x) {
    if (idy < _centroids_dots.rows ) {
      // i + idy: x^2 + y^2 between i^th data (a.k.a x) and idy^th
      // centroid (a.k.a y)
      _distance_pairs.ptr[i*_centroids_dots.rows + idy] =
          _data_dots.ptr[idx] + _centroids_dots.ptr[idy];
    }
  }
}

}  // namespace kernel

namespace detail {

template <typename T>
void PairWiseDistanceOp<T>::initialize(KmMatrix<T>& _data_dot,
                                       KmMatrix<T>& _centroids_dot,
                                       KmMatrix<T>& _distance_pairs) {
  data_dot_ = _data_dot;
  centroids_dot_ = _centroids_dot;
  distance_pairs_ = _distance_pairs;
  initialized_ = true;
}

template <typename T>
PairWiseDistanceOp<T>::PairWiseDistanceOp (KmMatrix<T>& _data_dot,
                                           KmMatrix<T>& _centroids_dot,
                                           KmMatrix<T>& _distance_pairs) :
    data_dot_(_data_dot), centroids_dot_(_centroids_dot),
    distance_pairs_(_distance_pairs), initialized_(true) {}

template <typename T>
KmMatrix<T> PairWiseDistanceOp<T>::operator()(KmMatrix<T>& _data,
                                              KmMatrix<T>& _centroids) {

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
      _centroids.rows(), _data.rows(), _data.cols(),
      &alpha,
      _centroids.dev_ptr(), _centroids.cols(),
      _data.dev_ptr(), _data.cols(),
      &beta,
      distance_pairs_.dev_ptr(), _centroids.rows());

  return distance_pairs_;
}

}  // namespace detail



/* ============== Class member functions ============== */

// We use counting to construct the weight as described in the paper. Counting
// is performed by histogram algorithm.
// For re-cluster, the paper suggests using K-Means++, but that will require
// copying data back to host. So we simply use those selected centroids with
// highest probability.

// FIXME:
// Operations performed in K-Means|| loop leads to a-approximation.
// Intuitively, choosing those centroids with highest probability should not
// break this property. But I haven't made the argument.
// And benchmarking should be performed to check the result.
template <typename T>
KmMatrix<T> KmeansLlInit<T>::recluster(KmMatrix<T>& _centroids) {
  KmMatrix<int> min_indices = ArgMinOp<T>().argmin(_centroids, KmMatrixDim::ROW);
  KmMatrix<int> weights (1, _centroids.rows());

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  // determine the temp_storage_bytes
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      min_indices.dev_ptr(),
                                      weights.dev_ptr(),
                                      _centroids.rows(),
                                      (T)0.0,
                                      (T)_centroids.rows(),
                                      (int)_centroids.rows());

  CUDA_CHECK(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      min_indices.dev_ptr(),
                                      weights.dev_ptr(),
                                      _centroids.rows(),
                                      (T)0.0,
                                      (T)_centroids.rows(),
                                      (int)_centroids.rows());
  CUDA_CHECK(cudaFree(d_temp_storage));

  // Sort the indices by weights in ascending order, then use those at front
  // as result.
  thrust::sort_by_key(thrust::device,
                      weights.dev_ptr(),
                      weights.dev_ptr() + weights.size(),
                      min_indices.dev_ptr(),
                      thrust::greater<int>());

  int * min_indices_ptr = min_indices.dev_ptr();

  KmMatrix<T> centroids (k_, _centroids.cols());
  int cols = _centroids.cols();
  size_t k = k_;

  thrust::copy_if(
      thrust::device,
      _centroids.dev_ptr(), _centroids.dev_ptr() + _centroids.size(),
      centroids.dev_ptr(),
      [=] __device__ (int idx) {
        size_t row = idx / cols;
        for (size_t i = 0; i < k; ++i) {
          if (row == min_indices_ptr[i])
            return true;
        }
        return false;
      });

  return centroids;
}

template <typename T>
KmMatrix<T> KmeansLlInit<T>::probability(
    KmMatrix<T>& _data, KmMatrix<T>& _centroids) {

  KmMatrix<T> centroids_dot (_centroids.rows(), 1);

  VecBatchDotOp<T>().dot(centroids_dot, _centroids);

  // FIXME: Time this
  distance_pairs_ = KmMatrix<T>(_data.rows(), _centroids.rows());
  detail::PairWiseDistanceOp<T> distance_op (
      data_dot_, centroids_dot, distance_pairs_);
  distance_pairs_ = distance_op(_data, _centroids);

  KmMatrix<T> min_distances = MinOp<T>().min(distance_pairs_, KmMatrixDim::ROW);

  T cost = SumOp<T>().sum(min_distances);

  KmMatrix<T> prob (min_distances.rows(), 1);
  MulOp<T>().mul(prob, min_distances, over_sample_ * k_ / cost);

  return prob;
}


template <typename T>
KmMatrix<T> KmeansLlInit<T>::sample_centroids(
    KmMatrix<T>& _data, KmMatrix<T>& _prob) {

  KmMatrix<T> thresholds = generator_->generate(_data.rows());

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

  if (_k > _data.size()) {
    char err_msg[128];
    sprintf(
        err_msg,
        "k must be less than or equal to the number of data points"
        ", k: %u, data points: %u",
        _k, _data.rows());
    M_USER_ERROR(err_msg);
  }

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
    M_ERROR("Not implemented.");
  }

  centroids = recluster(centroids);
  return centroids;
}

#define INSTANTIATE(T)                                                  \
  template KmMatrix<T> KmeansLlInit<T>::operator()(                     \
      KmMatrix<T>& _data, size_t _k);                                   \
  template KmMatrix<T> KmeansLlInit<T>::recluster(                      \
      KmMatrix<T>& centroids);                                          \
  template KmMatrix<T> KmeansLlInit<T>::probability(                    \
      KmMatrix<T>& data, KmMatrix<T>& centroids);                       \
  template KmMatrix<T> KmeansLlInit<T>::sample_centroids(               \
      KmMatrix<T>& data, KmMatrix<T>& centroids);                       \

INSTANTIATE(float)
INSTANTIATE(double)

#undef INSTANTIATE

namespace detail {

#define INSTANTIATE(T)                                          \
  template PairWiseDistanceOp<T>::PairWiseDistanceOp (          \
      KmMatrix<T>& _data_dot,                                   \
      KmMatrix<T>& _centroids_dot,                              \
      KmMatrix<T>& _distance_pairs);                            \
  template void PairWiseDistanceOp<T>::initialize(              \
      KmMatrix<T>& _data_dot,                                   \
      KmMatrix<T>& _centroids_dot,                              \
      KmMatrix<T>& _distance_pairs);                            \
  template KmMatrix<T> PairWiseDistanceOp<T>::operator()(       \
      KmMatrix<T>& _data,                                       \
      KmMatrix<T>& _centroids);

INSTANTIATE(float)
INSTANTIATE(double)

#undef INSTANTIATE
}
// FIXME: int is not supported due to random kernel

}  // namespace Kmeans
}  // namespace H2O4GPU
