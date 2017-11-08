/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <atomic>
#include <signal.h>
#include <string>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "kmeans_centroids.h"
#include "kmeans_labels.h"
#include "kmeans_general.h"

namespace kmeans {

//! kmeans clusters data into k groups
/*!

  \param n Number of data points
  \param d Number of dimensions
  \param k Number of clusters
  \param data Data points, in row-major order. This vector must have
  size n * d, and since it's in row-major order, data point x occupies
  positions [x * d, (x + 1) * d) in the vector. The vector is passed
  by reference since it is shared with the caller and not copied.
  \param labels Cluster labels. This vector has size n.
  The vector is passed by reference since it is shared with the caller
  and not copied.
  \param centroids Centroid locations, in row-major order. This
  vector must have size k * d, and since it's in row-major order,
  centroid x occupies positions [x * d, (x + 1) * d) in the
  vector. The vector is passed by reference since it is shared
  with the caller and not copied.
  \param distances Distances from points to centroids. This vector has
  size n. It is passed by reference since it is shared with the caller
  and not copied.
  \param threshold This controls early termination of the kmeans
  iterations. If the ratio of points being reassigned to a different
  centroid is less than the threshold, than the iterations are
  terminated. Defaults to 1e-3.
  \param max_iterations Maximum number of iterations to run
  \return The number of iterations actually performed.
 */

template<typename T>
int kmeans(
    int verbose,
    volatile std::atomic_int *flag,
    int n, int d, int k,
    thrust::device_vector<T> **data,
    thrust::device_vector<int> **labels,
    thrust::device_vector<T> **centroids,
    thrust::device_vector<T> **distances,
    thrust::device_vector<T> **data_dots,
    std::vector<int> dList,
    int n_gpu,
    int max_iterations,
    double threshold = 1e-3,
    bool do_per_iter_check = true) {

  thrust::device_vector<T> *centroid_dots[MAX_NGPUS];
  thrust::device_vector<T> *pairwise_distances[MAX_NGPUS];
  thrust::device_vector<int> *labels_copy[MAX_NGPUS];
  thrust::device_vector<int> *range[MAX_NGPUS];
  thrust::device_vector<int> *indices[MAX_NGPUS];
  thrust::device_vector<int> *counts[MAX_NGPUS];

  thrust::host_vector<int> h_counts(k);
  thrust::host_vector<int> h_counts_tmp(k);
  thrust::host_vector<T> h_centroids(k * d);
  h_centroids = *centroids[0]; // all should be equal
  thrust::host_vector<T> h_centroids_tmp(k * d);

  int h_changes[MAX_NGPUS], *d_changes[MAX_NGPUS];
  T h_distance_sum[MAX_NGPUS], *d_distance_sum[MAX_NGPUS];

  for (int q = 0; q < n_gpu; q++) {
    if (verbose) {
      fprintf(stderr, "Before kmeans() Allocation: gpu: %d\n", q);
      fflush(stderr);
    }

    safe_cuda(cudaSetDevice(dList[q]));
    safe_cuda(cudaMalloc(&d_changes[q], sizeof(int)));
    safe_cuda(cudaMalloc(&d_distance_sum[q], sizeof(T)));

    try {
      centroid_dots[q] = new thrust::device_vector<T>(k);
      pairwise_distances[q] = new thrust::device_vector<T>(n / n_gpu * k);
      labels_copy[q] = new thrust::device_vector<int>(n / n_gpu);
      range[q] = new thrust::device_vector<int>(n / n_gpu);
      counts[q] = new thrust::device_vector<int>(k);
      indices[q] = new thrust::device_vector<int>(n / n_gpu);
    } catch (thrust::system_error &e) {
      // output an error message and exit
      std::stringstream ss;
      ss << "Unable to allocate memory for gpu: " << q << " n/n_gpu: " << n / n_gpu << " k: " << k << " d: " << d
         << " error: " << e.what() << std::endl;
      return (-1);
      // throw std::runtime_error(ss.str());
    } catch (std::bad_alloc &e) {
      // output an error message and exit
      std::stringstream ss;
      ss << "Unable to allocate memory for gpu: " << q << " n/n_gpu: " << n / n_gpu << " k: " << k << " d: " << d
         << " error: " << e.what() << std::endl;
      return (-1);
      //throw std::runtime_error(ss.str());
    }

    if (verbose) {
      fprintf(stderr, "Before Create and save range for initializing labels: gpu: %d\n", q);
      fflush(stderr);
    }
    //Create and save "range" for initializing labels
    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(n / n_gpu),
                 (*range[q]).begin());

    if (verbose) {
      fprintf(stderr, "Before make_self_dots: gpu: %d\n", q);
      fflush(stderr);
    }
  }

  if (verbose) {
    fprintf(stderr, "Before kmeans() Iterations\n");
    fflush(stderr);
  }

  int i = 0;
  bool done = false;
  for (; i < max_iterations; i++) {
    if (*flag) continue;

    for (int q = 0; q < n_gpu; q++) {
      safe_cuda(cudaSetDevice(dList[q]));

      detail::calculate_distances(verbose, q, n / n_gpu, d, k,
                                  *data[q], *centroids[q], *data_dots[q],
                                  *centroid_dots[q], *pairwise_distances[q]);

      detail::relabel(n / n_gpu, k, *pairwise_distances[q], *labels[q], *distances[q], d_changes[q]);

      mycub::sum_reduce(*distances[q], d_distance_sum[q]);

      detail::memcpy(*labels_copy[q], *labels[q]);
      detail::find_centroids(q,
                             n / n_gpu,
                             d,
                             k,
                             *data[q],
                             *labels_copy[q],
                             *centroids[q],
                             *range[q],
                             *indices[q],
                             *counts[q],
                             n_gpu <= 1
      );
    }

    // Scale the centroids on host
    if (n_gpu > 1) {
      //Average the centroids from each device
      for (int p = 0; p < k; p++) h_counts[p] = 0.0;
      for (int q = 0; q < n_gpu; q++) {
        safe_cuda(cudaSetDevice(dList[q]));
        detail::memcpy(h_counts_tmp, *counts[q]);
        detail::streamsync(dList[q]);
        for (int p = 0; p < k; p++) h_counts[p] += h_counts_tmp[p];
      }

      // Zero the centroids only if any of the GPUs actually updated them
      for (int p = 0; p < k; p++) {
        for (int r = 0; r < d; r++) {
          if (h_counts_tmp[p] != 0) {
            h_centroids[p * d + r] = 0.0;
          }
        }
      }

      for (int q = 0; q < n_gpu; q++) {
        safe_cuda(cudaSetDevice(dList[q]));
        detail::memcpy(h_centroids_tmp, *centroids[q]);
        detail::streamsync(dList[q]);
        for (int p = 0; p < k; p++) {
          for (int r = 0; r < d; r++) {
            if (h_counts_tmp[p] != 0) {
              h_centroids[p * d + r] += h_centroids_tmp[p * d + r];
            }
          }
        }
      }

      for (int p = 0; p < k; p++) {
        for (int r = 0; r < d; r++) {
          // If 0 counts that means we leave the original centroids
          if (h_counts[p] == 0) {
            h_counts[p] = 1;
          }
          h_centroids[p * d + r] /= h_counts[p];
        }
      }

      //Copy the averaged centroids to each device
      for (int q = 0; q < n_gpu; q++) {
        safe_cuda(cudaSetDevice(dList[q]));
        detail::memcpy(*centroids[q], h_centroids);
      }
    }

    // whether to perform per iteration check
    if (do_per_iter_check) {
      double distance_sum = 0.0;
      int moved_points = 0.0;
      for (int q = 0; q < n_gpu; q++) {
        safe_cuda(cudaSetDevice(dList[q])); //  unnecessary
        safe_cuda(cudaMemcpyAsync(h_changes + q, d_changes[q], sizeof(int), cudaMemcpyDeviceToHost, cuda_stream[q]));
        safe_cuda(cudaMemcpyAsync(h_distance_sum + q,
                                  d_distance_sum[q],
                                  sizeof(T),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream[q]));
        detail::streamsync(dList[q]);
        if (verbose >= 2) {
          std::cout << "Device " << dList[q] << ":  Iteration " << i << " produced " << h_changes[q]
                    << " changes and the total_distance is " << h_distance_sum[q] << std::endl;
        }
        distance_sum += h_distance_sum[q];
        moved_points += h_changes[q];
      }
      if (i > 0) {
        double fraction = (double) moved_points / n;
#define NUMSTEP 10
        if (verbose > 1 && (i <= 1 || i % NUMSTEP == 0)) {
          std::cout << "Iteration: " << i << ", moved points: " << moved_points << std::endl;
        }
        if (fraction < threshold) {
          if (verbose) { std::cout << "Threshold triggered. Terminating early." << std::endl; }
          done = true;
        }
      }
    }

    if (*flag) {
      fprintf(stderr, "Signal caught. Terminated early.\n");
      fflush(stderr);
      *flag = 0; // set flag
      done = true;
    }

    if (done) break;
  }

// Final relabeling - uses final centroids
  for (int q = 0; q < n_gpu; q++) {
    safe_cuda(cudaSetDevice(dList[q]));

    detail::calculate_distances(verbose, q, n / n_gpu, d, k,
                                *data[q], *centroids[q], *data_dots[q],
                                *centroid_dots[q], *pairwise_distances[q]);

    detail::relabel(n / n_gpu, k, *pairwise_distances[q], *labels[q], *distances[q], d_changes[q]);
  }

  for (int q = 0; q < n_gpu; q++) {
    safe_cuda(cudaSetDevice(dList[q]));
    safe_cuda(cudaFree(d_changes[q]));
    delete (pairwise_distances[q]);
    delete (centroid_dots[q]);
    delete (labels_copy[q]);
    delete (range[q]);
    delete (counts[q]);
    delete (indices[q]);
  }

  if (verbose) {
    fprintf(stderr,
            "Iterations: %d\n", i);
    fflush(stderr);
  }
  return 0;
}

}

