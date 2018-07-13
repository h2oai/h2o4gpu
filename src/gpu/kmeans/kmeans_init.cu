/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <random>

#include <Eigen/Dense>

#include <stdio.h>

#include "kmeans_general.h"
#include "array.cuh"
#include "kmeans_h2o4gpu.h"

#include "kmeans_init.cuh"

namespace H2O4GPU {
namespace KMeans {

using namespace Array;

// K-Means|| implementation
template <typename T>
__device__ float vector_dot(T lhs_start, T lhs_end, T rhs_start) {
  float result = 0;
  for (T lhs_iter = lhs_start, rhs_iter = rhs_start;
       lhs_iter != lhs_end;
       ++lhs_iter, ++rhs_iter) {
    result += (*lhs_iter) * (*rhs_iter);
  }
  return result;
}

template <typename T>
__global__ void min_distance(T* __restrict__ result,
                             T* __restrict__ data, size_t stride,
                             T* __restrict__ cendroids, size_t n_centroids) {
  for (size_t i = 0; i < n_centroids; ++i) {
    result[i] =
        vector_dot(data, data+stride, data);
        // vector_dot(data, data+stride, &cendroids[i*stride]) +
        // vector_dot(&cendroids[i*stride], &cendroids[(i+1)*stride], &cendroids[i*stride]);
  }
  T minimum = std::numeric_limits<T>::max();
  for (size_t i = 0; i < n_centroids; ++i) {
    if (result[i] < minimum) {
      minimum = result[i];
    }
  }
}


template <typename T>
CUDAArray<T> KmeansLlInit<T>::sample_centroids(CUDAArray<T>& data, CUDAArray<T>& prob) {
  size_t n_new_centroids = thrust::count_if(
      data.device_vector().begin(), data.device_vector().end(),
      [=] __device__ (int idx) {
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<> dist(0.0, 1.0);
        rng.discard(idx);
        T threshold = (T)dist(rng);
        // T prob = prob[i];
        T prob = 0.1f;
        return prob > threshold;
      });

  CUDAArray<T> centroids (n_new_centroids);
  thrust::copy_if(data.device_vector().begin(), data.device_vector().end(),
                  centroids.device_vector().begin(),
                  [=] __device__ (int idx) {
                    thrust::default_random_engine rng(seed);
                    thrust::uniform_real_distribution<> dist(0.0, 1.0);
                    // rng.discard(row + device * rows_per_gpu);
                    T prob_threshold = (T)dist(rng);

                    // T prob_x = ((2.0 * k * min_costs_ptr[row]) / total_min_cost);
                    T prob_x = 0.1f;

                    return prob_x > prob_threshold;
                  });
  return centroids;
}

template <typename T>
CUDAArray<T> KmeansLlInit<T>::operator()(CUDAArray<T>& data) {
  if (seed < 0) {
    std::random_device rd;
    seed = rd();
  }

  std::mt19937 generator(seed);
  std::uniform_int_distribution<> distribution(0, data.dims()[0] - 1);
  size_t idx = distribution(generator);
  CUDAArray<T> centroids = data.index(idx);

  CUDAArray<T> distances (Dims(data.dims()[0], 1, 0, 0));

  distances.print();

  min_distance<<<256, data.size() / 256>>>(
      distances.get(), data.get(), data.dims()[1],
      centroids.get(), 1);

  cudaDeviceSynchronize();

  // T potential = * min_element(distances.begin(), distances.end());
  T potential = 1.0f;
  // for (size_t i = 0; i < log(potential); ++i) {
  //   min_distance(distances.device_ptr(),
  //                data.device_ptr(), data.stride(),
  //                centroids.device_ptr(), centroids.size());
  //   T potential = * thrust::min_element(distances.begin(), distances.end());
  //   T potential = 1.0f;
  //   CUDAArray<T> prob = div(distances, potential);

  //   CUDAArray<T> new_centroids = sample_centroids(data, prob);
  //   thrust::copy(new_centroids.begin(), new_centroids.end(), centroids.begin());
  // }

  // re-cluster
  // kmeans_plus_plus(centroids);
}

#define INSTANTIATE(T)                                                  \
  template CUDAArray<T> KmeansLlInit<T>::operator()(CUDAArray<T>& data); \
  template CUDAArray<T> KmeansLlInit<T>::sample_centroids(              \
      CUDAArray<T>& data, CUDAArray<T>& prob);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace Kmeans
}  // namespace H2O4GPU
