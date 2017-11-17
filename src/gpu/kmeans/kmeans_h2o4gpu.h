/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include "kmeans_labels.h"
#include "kmeans_centroids.h"

#define CUDACHECK(cmd) do {                           \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));\
      fflush( stdout );                               \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

/**
 * Calculates closest centroid for each record and counts how many points are assigned to each centroid.
 * @tparam T
 * @param verbose
 * @param num_gpu
 * @param rows_per_gpu
 * @param cols
 * @param data
 * @param data_dots
 * @param centroids
 * @param weights
 * @param pairwise_distances
 * @param labels
 */
template<typename T>
void count_pts_per_centroid(
    int verbose,
    int num_gpu, int rows_per_gpu, int cols,
    thrust::device_vector<T> **data,
    thrust::device_vector<T> **data_dots,
    thrust::host_vector<T> centroids,
    thrust::host_vector<T> &weights
) {
  int k = centroids.size() / cols;
  for (int i = 0; i < num_gpu; i++) {
    thrust::host_vector<int> weights_tmp(weights.size());

    CUDACHECK(cudaSetDevice(i));
    thrust::device_vector<T> pairwise_distances(rows_per_gpu * k);
    thrust::device_vector<T> centroid_dots(k);
    thrust::device_vector<T> d_centroids = centroids;

    kmeans::detail::calculate_distances(verbose, 0, rows_per_gpu, cols, k,
                                        *data[i],
                                        d_centroids,
                                        *data_dots[i],
                                        centroid_dots,
                                        pairwise_distances);

    thrust::device_vector<int> counts(k);
    auto counting = thrust::make_counting_iterator(0);
    auto counts_ptr = thrust::raw_pointer_cast(counts.data());
    auto pairwise_distances_ptr = thrust::raw_pointer_cast(pairwise_distances.data());
    thrust::for_each(counting, counting + rows_per_gpu, [=]__device__(int idx){
      int closest_centroid_idx = 0;
      T best_distance = pairwise_distances_ptr[idx];
      // FIXME potentially slow due to striding
      for (int i = 1; i < k; i++) {
        T distance = pairwise_distances_ptr[idx + i * rows_per_gpu];

        if (distance < best_distance) {
          best_distance = distance;
          closest_centroid_idx = i;
        }
      }
      atomicAdd(&counts_ptr[closest_centroid_idx], 1);
    });
    CUDACHECK(cudaDeviceSynchronize());

    kmeans::detail::memcpy(weights_tmp, counts);
    kmeans::detail::streamsync(i);
    for (int p = 0; p < k; p++) {
      weights[p] += weights_tmp[p];
    }
  }
}