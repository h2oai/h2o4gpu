#include "gtest/gtest.h"

// TODO: make it less vague(use proper include)
#include "../gpu/kmeans/kmeans_h2o4gpu.h"
#include "../gpu/kmeans/kmeans_labels.h"
#include "cuda_utils2.h"
#include <thrust/host_vector.h>

TEST(KMeans, PartialCopyDataOrderR) {
  // 0 1 2
  // 3 4 5
  // 6 7 8
  float src[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  float dst[6] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<float> d_dst(6);
  copy_data(0, 'r', d_dst, &src[0], 0, 3, 2, 3);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_dst = d_dst;
  for (int i = 0; i < 6; ++i) {
    ASSERT_FLOAT_EQ(h_dst[i], dst[i]) << i;
  }
}

TEST(KMeans, PartialCopyDataOrderC) {
  // 0 1 2
  // 3 4 5
  // 6 7 8
  float src[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
  float dst[6] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<float> d_dst(6);
  copy_data(0, 'c', d_dst, &src[0], 0, 3, 2, 3);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_dst = d_dst;
  for (int i = 0; i < 6; ++i) {
    ASSERT_FLOAT_EQ(h_dst[i], dst[i]) << i;
  }
}

TEST(KMeans, CopyDataOrderR) {
  // 0 1
  // 2 3
  float src[4] = {0, 1, 2, 3};
  float dst[4] = {0, 1, 2, 3};
  thrust::device_vector<float> d_dst(4);
  copy_data(0, 'r', d_dst, &src[0], 0, 2, 2, 2);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_dst = d_dst;
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(h_dst[i], dst[i]) << i;
  }
}

TEST(KMeans, CopyDataOrderC) {
  // 0 1
  // 2 3
  float src[4] = {0, 2, 1, 3};
  float dst[4] = {0, 1, 2, 3};
  thrust::device_vector<float> d_dst(4);
  copy_data(0, 'c', d_dst, &src[0], 0, 2, 4, 2);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_dst = d_dst;
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(h_dst[i], dst[i]) << i;
  }
}

TEST(KMeans, CountsPerCentroids) {
  // GIVEN
  int d = 2;
  int n = 4;
  int k = 2;

  // Setup data
  thrust::host_vector<float> h_data(n * d);
  h_data[0] = 0.0f;
  h_data[1] = 0.0f;
  h_data[2] = 2.0f;
  h_data[3] = 2.0f;
  h_data[4] = 2.0f;
  h_data[5] = 3.0f;
  h_data[6] = 3.0f;
  h_data[7] = 2.0f;
  thrust::device_vector<float> *d_data = new thrust::device_vector<float>();
  *d_data = h_data;

  // Data dots
  thrust::device_vector<float> *d_data_dots =
      new thrust::device_vector<float>(n);
  kmeans::detail::make_self_dots(n, d, *d_data, *d_data_dots);

  // Centroids
  thrust::host_vector<float> h_centroids(k * d);
  h_centroids[0] = 0.0f;
  h_centroids[1] = 0.0f;
  h_centroids[2] = 2.0f;
  h_centroids[3] = 2.0f;

  thrust::host_vector<float> h_weights(k);

  // WHEN
  count_pts_per_centroid(0, 1, n, d, &d_data, &d_data_dots, h_centroids,
                         h_weights);

  delete (d_data);
  delete (d_data_dots);

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
  thrust::host_vector<float> h_data(n * d);
  h_data[0] = 1.0f;
  h_data[1] = 2.0f;
  h_data[2] = 1.0f;
  h_data[3] = 4.0f;
  h_data[4] = 1.0f;
  h_data[5] = 0.0f;
  h_data[6] = 4.0f;
  h_data[7] = 2.0f;
  h_data[8] = 4.0f;
  h_data[9] = 4.0f;
  h_data[10] = 4.0f;
  h_data[11] = 0.0f;
  thrust::device_vector<float> *d_data = new thrust::device_vector<float>();
  *d_data = h_data;

  // Data dots
  thrust::device_vector<float> *d_data_dots =
      new thrust::device_vector<float>(n);
  kmeans::detail::make_self_dots(n, d, *d_data, *d_data_dots);

  // Centroids
  thrust::host_vector<float> h_centroids(n * d);
  h_centroids[0] = 1.0f;
  h_centroids[1] = 2.0f;
  h_centroids[2] = 1.0f;
  h_centroids[3] = 4.0f;
  h_centroids[4] = 1.0f;
  h_centroids[5] = 0.0f;
  h_centroids[6] = 4.0f;
  h_centroids[7] = 2.0f;
  h_centroids[8] = 4.0f;
  h_centroids[9] = 4.0f;
  h_centroids[10] = 4.0f;
  h_centroids[11] = 0.0f;

  thrust::host_vector<float> h_weights(n);

  // WHEN
  count_pts_per_centroid(0, 1, n, d, &d_data, &d_data_dots, h_centroids,
                         h_weights);

  // THEN
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[0]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[1]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[2]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[3]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[4]);
  ASSERT_FLOAT_EQ(1.0f, h_weights.data()[5]);

  delete (d_data);
  delete (d_data_dots);

  SUCCEED();
}