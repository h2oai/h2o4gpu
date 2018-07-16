/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <random>

#define EIGNE_USE_GPU
#include "Eigen/Dense"

#include <stdio.h>

#include "kmeans_general.h"
#include "kmeans_h2o4gpu.h"

#include "kmeans_init.cuh"
#include "KmMatrix/KmMatrix.hpp"

namespace H2O4GPU {
namespace KMeans {

template <typename T>
__device__ __forceinline__
T min_distance(VE_T(T) *x, MA_T(T) *centroids) {

  KmShardMem<T> shared;
  T * _distances = shared.ptr();

  size_t n_rows = centroids->rows();
  for (size_t i = 0; i < centroids->rows(); ++i) {
    auto temp = *x - centroids->row(i);
    _distances[i] = temp.dot(temp);
  }

  __syncthreads();

  Eigen::Map<MA_T(T)> _distances_vec(_distances, n_rows, 1);
  T result = _distances_vec.minCoeff();
  return result;
}

template <typename T>
__global__
void potential_kernel(kVParam<T> _dis, kMParam<T> _data, kMParam<T> _cent) {

  MA_T(T) data = Eigen::Map<MA_T(T)>(_data.ptr, _data.rows, _data.cols);
  MA_T(T) centroids = Eigen::Map<MA_T(T)>(_cent.ptr, _cent.rows, _cent.cols);

  Eigen::Map<VE_T(T)> distances(_dis.ptr, _dis.size);

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < _dis.size) {
    distances(tid) = min_distance<T>(&( (VE_T(T)) data.row(tid)),
                                        &centroids);
    printf("distance[%u] %f\n", tid, distances(tid));
  }
}

template <typename T>
T KmeansLlInit<T>::potential(MA_T(T)& data, MA_T(T)& centroids) {

  VE_T(T) distances (data.rows());

  T* d_distances, * d_data, *d_centroids;

  CUDACHECK(cudaMalloc((void**)&d_distances, sizeof(T) * distances.size()));
  CUDACHECK(cudaMalloc((void**)&d_data, sizeof(T) * data.size()));
  CUDACHECK(cudaMalloc((void**)&d_centroids, sizeof(T) * centroids.size()));

  CUDACHECK(cudaMemcpy(d_distances, (void*)distances.data(),
                       sizeof(T) * distances.size(),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_data, (void*)data.data(),
                       sizeof(T) * data.size(),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_centroids, (void*)centroids.data(),
                       sizeof(T) * centroids.size(),
                       cudaMemcpyHostToDevice));

  potential_kernel<T><<<256, div_roundup(data.rows(), 256),
      sizeof(T)*centroids.rows()>>>(
          kVParam<T>(d_distances, distances.size()),
          kMParam<T>(d_data, data.rows(), data.cols()),
          kMParam<T>(d_centroids, centroids.rows(), centroids.cols()));

  CUDACHECK(cudaDeviceSynchronize());

  thrust::device_ptr<T> distances_vec (d_distances);

  T * temp = new T[distances.size()];
  CUDACHECK(cudaMemcpy(temp, d_distances, sizeof(T)*distances.size(), cudaMemcpyDeviceToHost));

  T res = thrust::reduce(distances_vec, distances_vec + distances.size(), (T)0,
                         thrust::plus<T>());

  CUDACHECK(cudaFree(d_distances));
  CUDACHECK(cudaFree(d_data));

  CUDACHECK(cudaFree(d_centroids));

  CUDACHECK(cudaGetLastError());

  return res;
}

template <typename T>
T KmeansLlInit<T>::probability(MA_T(T)& data, MA_T(T)& controids) {

}

template <typename T>
struct InplaceMulOp {
  T a;
  InplaceMulOp(T _a) : a(_a) {}

  __host__ __device__
  void operator()(T x) {
    // *x = *x * a;
  }
};

template <typename T>
MA_T(T) KmeansLlInit<T>::sample_centroids(MA_T(T)& data, MA_T(T)& centroids) {
  VE_T(T) distances (data.rows());

  T* d_distances, * d_data, *d_centroids;

  CUDACHECK(cudaMalloc((void**)&d_distances, sizeof(T) * distances.size()));
  CUDACHECK(cudaMalloc((void**)&d_data, sizeof(T) * data.size()));
  CUDACHECK(cudaMalloc((void**)&d_centroids, sizeof(T) * centroids.size()));

  CUDACHECK(cudaMemcpy(d_distances, (void*)distances.data(),
                       sizeof(T) * distances.size(),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_data, (void*)data.data(),
                       sizeof(T) * data.size(),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_centroids, (void*)centroids.data(),
                       sizeof(T) * centroids.size(),
                       cudaMemcpyHostToDevice));

  potential_kernel<T><<<256, div_roundup(data.rows(), 256),
      sizeof(T)*centroids.rows()>>>(
          kVParam<T>(d_distances, distances.size()),
          kMParam<T>(d_data, data.rows(), data.cols()),
          kMParam<T>(d_centroids, centroids.rows(), centroids.cols()));

  CUDACHECK(cudaDeviceSynchronize());

  thrust::device_ptr<T> distances_vec (d_distances);

  // T * temp = new T[distances.size()];
  // CUDACHECK(cudaMemcpy(temp, d_distances, sizeof(T)*distances.size(), cudaMemcpyDeviceToHost));

  T pot = thrust::reduce(distances_vec, distances_vec + distances.size(), (T)0,
                         thrust::plus<T>());

  thrust::device_ptr<T>& prob_vec = distances_vec;
  thrust::for_each(prob_vec, prob_vec + distances.size(), InplaceMulOp<T>(1/pot));

  CUDACHECK(cudaDeviceSynchronize());

  size_t _cols = data.cols();
  size_t _rows = data.rows();

  std::cout << "distances.size()" << distances.size() << std::endl;
  auto pot_cent_filter_counter = thrust::make_counting_iterator(0);
  size_t n_new_centroids =
      thrust::count_if(pot_cent_filter_counter, pot_cent_filter_counter + distances.size()-1,
                       [=] __device__(int idx) {
                         thrust::default_random_engine rng(0);
                         thrust::uniform_real_distribution<> dist(0.0, 1.0);
                         rng.discard(idx + _rows);
                         T threshold = (T) dist (rng);
                         printf("count:thresh[%u]: %f\n", idx, threshold);
                         // T prob = d_distances[idx] / pot;
                         T prob = 0.5;
                         printf("count:prob[%u]: %f\n", idx, prob);
                         return prob > threshold; });

  std::cout << "n_new_centroids:" << n_new_centroids << std::endl;
  thrust::device_vector<T> d_new_centroids (n_new_centroids * data.cols());
  auto range = thrust::make_counting_iterator(0);
  thrust::device_ptr<T> d_data_vec (d_data);

  // thrust::copy_if(
  //     d_data_vec, d_data_vec + data.size(),
  //     range,
  //     d_new_centroids.begin(),
  //     [=] __device__ (int idx) {
  //       size_t row = idx / _cols;
  //       thrust::default_random_engine rng(seed);
  //       thrust::uniform_real_distribution<> dist(0.0f, 1.0f);
  //       rng.discard(row);
  //       T threshold = (T) dist (rng);
  //       printf("copy:thresh[%u]: %f", row, threshold);
  //       T prob = d_distances[row];
  //       return prob > threshold;});

  thrust::host_vector<T> h_new_centroids (n_new_centroids);
  thrust::copy(d_new_centroids.begin(), d_new_centroids.end(),
               h_new_centroids.begin());

  size_t old_rows = centroids.rows();
  centroids.conservativeResize(data.rows() + n_new_centroids, Eigen::NoChange);

  for (size_t i = 0; i < n_new_centroids; i ++) {
    centroids.row(i+old_rows) = Eigen::Map<MA_T(T)> (h_new_centroids.data(), 1, data.cols());
  }

  CUDACHECK(cudaFree(d_distances));
  CUDACHECK(cudaFree(d_data));

  CUDACHECK(cudaFree(d_centroids));

  CUDACHECK(cudaGetLastError());

  return centroids;
}

template <typename T>
KmMatrix<T>
KmeansLlInit<T>::operator()(H2O4GPU::KMeans::KmMatrix<T>& data) {

  if (seed < 0) {
    std::random_device rd;
    seed = rd();
  }

  std::mt19937 generator(0);
  thrust::host_vector<T> vec (4);

  std::uniform_int_distribution<> distribution(0, data.rows());
  size_t idx = distribution(generator);

  KmMatrix<T> centroids = data.row(idx);
  std::cout << "centroids" << std::endl;
  std::cout << centroids << std::endl;

  // MA_T(T) centroids = data.row(idx);

  // std::cout << "data\n" << data << std::endl;
  // T pot = potential(data, centroids);
  // std::cout << "pot: " << pot << std::endl;

  // for (size_t i = 0; i < std::log(pot); ++i) {
  //   sample_centroids(data, centroids);
  //   std::cout << "new centroids" << std::endl;
  //   std::cout << centroids << std::endl;
  // }

  // re-cluster
  // kmeans_plus_plus(centroids);
  return data;
}

#define INSTANTIATE(T)                                                  \
  template KmMatrix<T> KmeansLlInit<T>::operator()(                     \
      KmMatrix<T>& data);                                               \
  template MA_T(T) KmeansLlInit<T>::sample_centroids(                   \
      MA_T(T)& data, MA_T(T)& centroids);                               \
  template T KmeansLlInit<T>::probability(MA_T(T)& data, MA_T(T)& controids); \


INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

}  // namespace Kmeans
}  // namespace H2O4GPU
