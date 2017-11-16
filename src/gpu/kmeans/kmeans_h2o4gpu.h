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
    thrust::device_vector<T> centroid_dots(k);
    thrust::device_vector<T> d_centroids = centroids;
    thrust::device_vector<int> counts(k);

    // Get info about available memory
    // This part of the algo can be very memory consuming
    // We might need to batch it
    // TODO not using batch_calculate_distances because nvcc is complaining about using the __device__
    // lambda inside a lambda
    size_t free_byte;
    size_t total_byte;
    CUDACHECK(cudaMemGetInfo( &free_byte, &total_byte ));
    free_byte *= 0.8;

    double required_byte = rows_per_gpu * k * sizeof(T);

    int runs = std::ceil( required_byte / free_byte );
    int offset = 0;
    for(int run = 0; run < runs; run++) {
      int rows_per_run = free_byte / (k * sizeof(T));

      if (run + 1 == runs) {
        rows_per_run = rows_per_gpu % rows_per_run;
      }

      thrust::device_vector<T> pairwise_distances(rows_per_run * k);
      kmeans::detail::calculate_distances(verbose, 0, rows_per_run, cols, k,
                                          *data[i], offset,
                                          d_centroids,
                                          *data_dots[i],
                                          centroid_dots,
                                          pairwise_distances);

      auto counting = thrust::make_counting_iterator(0);
      auto counts_ptr = thrust::raw_pointer_cast(counts.data());
      auto pairwise_distances_ptr = thrust::raw_pointer_cast(pairwise_distances.data());
      thrust::for_each(counting, counting + rows_per_run, [=]__device__(int idx){
        int closest_centroid_idx = 0;
        T best_distance = pairwise_distances_ptr[idx];
        // FIXME potentially slow due to striding
        for (int i = 1; i < k; i++) {
          T distance = pairwise_distances_ptr[idx + i * rows_per_run];

          if (distance < best_distance) {
            best_distance = distance;
            closest_centroid_idx = i;
          }
        }
        atomicAdd(&counts_ptr[closest_centroid_idx], 1);
      });
      offset += rows_per_run;
    }
    CUDACHECK(cudaDeviceSynchronize());

    kmeans::detail::memcpy(weights_tmp, counts);
    kmeans::detail::streamsync(i);
    for (int p = 0; p < k; p++) {
      weights[p] += weights_tmp[p];
    }
  }
}