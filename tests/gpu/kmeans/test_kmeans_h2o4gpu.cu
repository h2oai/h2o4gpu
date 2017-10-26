#include "gtest/gtest.h"

#include "../src/gpu/kmeans/kmeans_h2o4gpu.h"
#include "../src/gpu/kmeans/kmeans_labels.h"
#include <thrust/host_vector.h>

TEST(KMeans, CountsPerCentroids) {
  // GIVEN
  int k = 2;
  int d = 2;
  int n = 4;

  // Setup data
  thrust::host_vector<float> h_data(n*d);
  h_data[0] = 0.0f; h_data[1] = 0.0f; // [0,0]
  h_data[2] = 0.0f; h_data[3] = 0.0f; // [0,0]
  h_data[4] = 1.0f; h_data[5] = 1.0f; // [1,1]
  h_data[6] = 1.0f; h_data[7] = 1.0f; // [1,1]
  thrust::device_vector<float> *d_data = new thrust::device_vector<float>();
  *d_data = h_data;

  // Data dots
  thrust::device_vector<float> *d_data_dots = new thrust::device_vector<float>(n);
  kmeans::detail::make_self_dots(n, d, *d_data, *d_data_dots);

  // Centroids
  thrust::host_vector<float> h_centroids(k*d); // TODO ordering??
  h_centroids[0] = -1.0f; h_centroids[1] = -1.0f; // [0,0]
  h_centroids[2] = 2.0f; h_centroids[3] = 2.0f; // [1,1]

  thrust::host_vector<float> h_weights(k);

  count_pts_per_centroid(0, 1, n, d, &d_data, &d_data_dots, h_centroids, h_weights);

  // THEN
  ASSERT_FLOAT_EQ(2.0f, h_weights.data()[0]);
  ASSERT_FLOAT_EQ(2.0f, h_weights.data()[1]);

  SUCCEED();

}