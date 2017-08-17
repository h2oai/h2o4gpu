// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once

#include <atomic>
#include <thrust/device_vector.h>
#include "centroids.h"

namespace kmeans {

    //! kmeans clusters data into k groups
    /*!

      \param verbose Verbosity level. 0 does not print information, 1 does.
      \param flag
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
      \param dList GPU ids
      \param n_gpu Number of GPUs to be used
      \param max_iterations Maximum number of iterations to run
      \param init_from_labels If true, the labels need to be initialized
      before calling kmeans. If false, the centroids need to be
      initialized before calling kmeans. Defaults to true, which means
      the labels must be initialized.
      \param threshold This controls early termination of the kmeans
      iterations. If the ratio of points being reassigned to a different
      centroid is less than the threshold, than the iterations are
      terminated. Defaults to 1e-3.
      \return 0
     */

    template<typename T>
    int kmeans(
            int verbose,
            volatile std::atomic_int *flag,
            int n, int d, int k,
            thrust::device_vector <T> **data,
            thrust::device_vector<int> **labels,
            thrust::device_vector <T> **centroids,
            thrust::device_vector <T> **distances,
            std::vector<int> dList,
            int n_gpu,
            int max_iterations,
            int init_from_labels = 0,
            double threshold = 1e-3
    );
}
