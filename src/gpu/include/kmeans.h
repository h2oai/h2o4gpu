#pragma once
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "centroids.h"
#include "kmeans_labels.h"

template<typename T>
void print_array(T& array, int m, int n) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      typename T::value_type value = array[i * n + j];
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
}

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
    \param init_from_labels If true, the labels need to be initialized
    before calling kmeans. If false, the centroids need to be
    initialized before calling kmeans. Defaults to true, which means
    the labels must be initialized.
    \param threshold This controls early termination of the kmeans
    iterations. If the ratio of points being reassigned to a different
    centroid is less than the threshold, than the iterations are
    terminated. Defaults to 1e-3.
    \param max_iterations Maximum number of iterations to run
    \return The number of iterations actually performed.
   */

  template<typename T>
    int kmeans(
        int n, int d, int k,
        thrust::device_vector<T>** data,
        thrust::device_vector<int>** labels,
        thrust::device_vector<T>** centroids,
        thrust::device_vector<T>** distances,
        int n_gpu,
        int max_iterations,
        bool init_from_labels=true,
        double threshold=1e-3) {
      thrust::device_vector<T> *data_dots[16];
      thrust::device_vector<T> *centroid_dots[16];
      thrust::device_vector<T> *pairwise_distances[16];
      thrust::device_vector<int> *labels_copy[16];
      thrust::device_vector<int> *range[16];
      thrust::device_vector<int> *indices[16];
      thrust::device_vector<int> *counts[16];

      thrust::host_vector<T> h_centroids( k * d );
      thrust::host_vector<T> h_centroids_tmp( k * d );
      int h_changes[16], *d_changes[16];
      T h_distance_sum[16], *d_distance_sum[16];


      for (int q = 0; q < n_gpu; q++) {

        cudaSetDevice(q);
        cudaMalloc(&d_changes[q], sizeof(int));
        cudaMalloc(&d_distance_sum[q], sizeof(T));
        detail::labels_init();
        data_dots[q] = new thrust::device_vector <T>(n/n_gpu);
        centroid_dots[q] = new thrust::device_vector<T>(n/n_gpu);
        pairwise_distances[q] = new thrust::device_vector<T>(n/n_gpu * k);
        labels_copy[q] = new thrust::device_vector<int>(n/n_gpu * d);
        range[q] = new thrust::device_vector<int>(n/n_gpu);
        counts[q] = new thrust::device_vector<int>(k);
        indices[q] = new thrust::device_vector<int>(n/n_gpu);
        //Create and save "range" for initializing labels
        thrust::copy(thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n/n_gpu), 
            (*range[q]).begin());

        detail::make_self_dots(n/n_gpu, d, *data[q], *data_dots[q]);
        if (init_from_labels) {
          detail::find_centroids(n/n_gpu, d, k, *data[q], *labels[q], *centroids[q], *range[q], *indices[q], *counts[q]);
        }
      }

      int i=0;
      for(; i < max_iterations; i++) {
        //Average the centroids from each device
        if (n_gpu > 1) {
          for (int p = 0; p < k * d; p++) h_centroids[p] = 0.0;
          for (int q = 0; q < n_gpu; q++) {
            cudaSetDevice(q);
            detail::memcpy(h_centroids_tmp, *centroids[q]);
            detail::streamsync(q);
            for (int p = 0; p < k * d; p++) h_centroids[p] += h_centroids_tmp[p];
          }
          for (int p = 0; p < k * d; p++) h_centroids[p] /= n_gpu;
          //Copy the averaged centroids to each device 
          for (int q = 0; q < n_gpu; q++) {
            cudaSetDevice(q);
            detail::memcpy(*centroids[q],h_centroids);
          }
        }
        for (int q = 0; q < n_gpu; q++) {
          cudaSetDevice(q);

          detail::calculate_distances(n/n_gpu, d, k,
              *data[q], *centroids[q], *data_dots[q],
              *centroid_dots[q], *pairwise_distances[q]);

          detail::relabel(n/n_gpu, k, *pairwise_distances[q], *labels[q], *distances[q], d_changes[q]);
          //TODO remove one memcpy
          detail::memcpy(*labels_copy[q], *labels[q]);
          detail::find_centroids(n/n_gpu, d, k, *data[q], *labels[q], *centroids[q], *range[q], *indices[q], *counts[q]);
          detail::memcpy(*labels[q], *labels_copy[q]);
          //T d_distance_sum[q] = thrust::reduce(distances[q].begin(), distances[q].end())
          mycub::sum_reduce(*distances[q], d_distance_sum[q]);
        }
        double distance_sum = 0.0;
        int moved_points = 0.0;
        for (int q = 0; q < n_gpu; q++) {
          cudaMemcpyAsync(h_changes+q, d_changes[q], sizeof(int), cudaMemcpyDeviceToHost, cuda_stream[q]);
          cudaMemcpyAsync(h_distance_sum+q, d_distance_sum[q], sizeof(T), cudaMemcpyDeviceToHost, cuda_stream[q]);
          detail::streamsync(q);
          //std::cout << "Device " << q << ":  Iteration " << i << " produced " << h_changes[q]
          //  << " changes and the total_distance is " << h_distance_sum[q] << std::endl;
          distance_sum += h_distance_sum[q];
          moved_points += h_changes[q];
        }
        if (i > 0) {
          double fraction = (double)moved_points / n;
          std::cout << "Iteration: " << i << ", moved points: " << moved_points << std::endl;
          if (fraction < threshold) {
            std::cout << "Threshold triggered. Terminating early." << std::endl;
            return i + 1;
          }
        }
      }
      for (int q = 0; q < n_gpu; q++) {
        cudaSetDevice(q);
        cudaFree(d_changes[q]);
        detail::labels_close();
        delete(pairwise_distances[q]);
        delete(data_dots[q]);
        delete(centroid_dots[q]);
      }
      return i;
    }
}
