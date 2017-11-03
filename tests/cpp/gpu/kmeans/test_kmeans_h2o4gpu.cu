#include "gtest/gtest.h"

#include "../src/gpu/kmeans/kmeans_h2o4gpu.h"
#include "../src/gpu/kmeans/kmeans_labels.h"
#include <thrust/host_vector.h>

TEST(KMeans, CountsPerCentroids) {
  // GIVEN
  int d = 2;
  int n = 4;
  int k = 2;

  // Setup data
  thrust::host_vector<float> h_data(n*d);
  h_data[0] = 0.0f; h_data[1] = 0.0f;
  h_data[2] = 2.0f; h_data[3] = 2.0f;
  h_data[4] = 2.0f; h_data[5] = 3.0f;
  h_data[6] = 3.0f; h_data[7] = 2.0f;
  thrust::device_vector<float> *d_data = new thrust::device_vector<float>();
  *d_data = h_data;

  // Data dots
  thrust::device_vector<float> *d_data_dots = new thrust::device_vector<float>(n);
  kmeans::detail::make_self_dots(n, d, *d_data, *d_data_dots);

  // Centroids
  thrust::host_vector<float> h_centroids(k*d);
  h_centroids[0] = 0.0f; h_centroids[1] = 0.0f;
  h_centroids[2] = 2.0f; h_centroids[3] = 2.0f;

  thrust::host_vector<float> h_weights(k);

  // WHEN
  count_pts_per_centroid(0, 1, n, d, &d_data, &d_data_dots, h_centroids, h_weights);

  // THEN
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[0]);
  ASSERT_FLOAT_EQ(3.0f, h_weights.data()[1]);

  SUCCEED();

}

TEST(KMeans, CentroidPerDataPoint) {
  // GIVEN
  int d = 2;
  int n = 6;

  // Setup data
  thrust::host_vector<float> h_data(n*d);
  h_data[0] = 1.0f; h_data[1] = 2.0f;
  h_data[2] = 1.0f; h_data[3] = 4.0f;
  h_data[4] = 1.0f; h_data[5] = 0.0f;
  h_data[6] = 4.0f; h_data[7] = 2.0f;
  h_data[8] = 4.0f; h_data[9] = 4.0f;
  h_data[10] = 4.0f; h_data[11] = 0.0f;
  thrust::device_vector<float> *d_data = new thrust::device_vector<float>();
  *d_data = h_data;

  // Data dots
  thrust::device_vector<float> *d_data_dots = new thrust::device_vector<float>(n);
  kmeans::detail::make_self_dots(n, d, *d_data, *d_data_dots);

  // Centroids
  thrust::host_vector<float> h_centroids(n*d);
  h_centroids[0] = 1.0f; h_centroids[1] = 2.0f;
  h_centroids[2] = 1.0f; h_centroids[3] = 4.0f;
  h_centroids[4] = 1.0f; h_centroids[5] = 0.0f;
  h_centroids[6] = 4.0f; h_centroids[7] = 2.0f;
  h_centroids[8] = 4.0f; h_centroids[9] = 4.0f;
  h_centroids[10] = 4.0f; h_centroids[11] = 0.0f;

  thrust::host_vector<float> h_weights(n);

  // WHEN
  count_pts_per_centroid(0, 1, n, d, &d_data, &d_data_dots, h_centroids, h_weights);

  // THEN
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[0]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[1]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[2]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[3]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[4]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[5]);

  SUCCEED();

}