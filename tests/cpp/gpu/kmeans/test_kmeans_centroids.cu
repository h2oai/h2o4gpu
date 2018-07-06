#include "gtest/gtest.h"

#include "../../../../src/gpu/kmeans/kmeans_centroids.h"
#include <thrust/host_vector.h>

TEST(KMeansCentroids, CalculateCentroids) {
  // GIVEN
  int k = 2;
  int d = 2;
  int n = 4;

  // Setup data
  thrust::host_vector<float> dataHost(n*d);
  dataHost[0] = 0.0f; dataHost[1] = 0.0f; // [0,0]
  dataHost[2] = 0.0f; dataHost[3] = 1.0f; // [0,1]
  dataHost[4] = 1.0f; dataHost[5] = 1.0f; // [1,1]
  dataHost[6] = 1.0f; dataHost[7] = 0.0f; // [1,1]

  thrust::device_vector<float> dataDevice(n*d);
  dataDevice = dataHost;

  // Setup counts
  thrust::device_vector<int> countsDevice(k);
  countsDevice[0] = 0;
  countsDevice[1] = 0;

  // Setup labels
  thrust::host_vector<int> labelsHost(n);
  labelsHost[0] = 0; // label for [0,0] -> 0
  labelsHost[1] = 0; // label for [0,1] -> 0
  labelsHost[2] = 1; // label for [1,1] -> 1
  labelsHost[3] = 1; // label for [1,0] -> 1
  thrust::device_vector<int> labelsDevice(n);
  labelsDevice = labelsHost;

  // Setup indices
  thrust::host_vector<int> indicesHost(n);
  indicesHost[0] = 0; indicesHost[1] = 1; indicesHost[2] = 2; indicesHost[3] = 3;
  thrust::device_vector<int> indicesDevice(n);
  indicesDevice = indicesHost;

  // Setup centroids
  thrust::host_vector<float> centroidsHost(d*k);
  centroidsHost[0] = 0.0f; centroidsHost[1] = 0.0f; centroidsHost[2] = 0.0f; centroidsHost[3] = 0.0f;
  thrust::device_vector<float> centroidsDevice(d*k);
  centroidsDevice = centroidsHost;

  int n_threads_x = 64;
  int n_threads_y = 16;
  kmeans::detail::calculate_centroids <<< dim3(1, 30), dim3(n_threads_x, n_threads_y), 0 >>> (
    n, d, k,
    thrust::raw_pointer_cast(dataDevice.data()),
    thrust::raw_pointer_cast(labelsDevice.data()),
    thrust::raw_pointer_cast(indicesDevice.data()),
    thrust::raw_pointer_cast(centroidsDevice.data()),
    thrust::raw_pointer_cast(countsDevice.data())
  );

  // THEN
  centroidsHost = centroidsDevice;

  ASSERT_FLOAT_EQ(0.0f, centroidsHost.data()[0]);
  ASSERT_FLOAT_EQ(1.0f, centroidsHost.data()[1]);
  ASSERT_FLOAT_EQ(2.0f, centroidsHost.data()[2]);
  ASSERT_FLOAT_EQ(1.0f, centroidsHost.data()[3]);

  SUCCEED();

}

// Calculating centroids "on 2 GPUs" should yield the same result as
// calculating centroids on 1 GPU from all data
TEST(KMeansCentroids, CalculateCentroids2GPU) {
  /**
   * CALCULATE CENTROIDS IN 2 TURNS, EACH TIME FROM HALF THE DATA
   * */
  // GIVEN
  int k = 2; int d = 2; int n = 3;

  thrust::host_vector<float> dataHost(n*d);
  thrust::device_vector<float> dataDevice(n*d);
  thrust::device_vector<int> countsDevice(k);
  thrust::host_vector<int> labelsHost(n);
  thrust::device_vector<int> labelsDevice(n);
  thrust::host_vector<float> centroidsHost(d*k);
  thrust::device_vector<float> centroidsDevice(d*k);
  thrust::host_vector<float> centroidsHostFirst(d*k);
  thrust::host_vector<int> indicesHost(n);
  thrust::device_vector<int> indicesDevice(n);
  int n_threads_x = 64; int n_threads_y = 16;

  indicesHost[0] = 0; indicesHost[1] = 1; indicesHost[2] = 2;
  dataHost[0] = 4.0f; dataHost[1] = 2.0f; // [4,2]
  dataHost[2] = 1.0f; dataHost[3] = 0.0f; // [1,0]
  dataHost[4] = 4.0f; dataHost[5] = 0.0f; // [4,0]
  countsDevice[0] = 0; countsDevice[1] = 0;
  labelsHost[0] = 0; labelsHost[1] = 0;  labelsHost[2] = 0;
  centroidsHost[0] = 0.0f; centroidsHost[1] = 0.0f; centroidsHost[2] = 0.0f; centroidsHost[3] = 0.0f;
  indicesDevice = indicesHost;
  dataDevice = dataHost;
  labelsDevice = labelsHost;
  centroidsDevice = centroidsHost;

  // Run on "gpu1"
  kmeans::detail::calculate_centroids <<< dim3(1, 30), dim3(n_threads_x, n_threads_y), 0 >>> (
      n, d, k,
      thrust::raw_pointer_cast(dataDevice.data()),
      thrust::raw_pointer_cast(labelsDevice.data()),
      thrust::raw_pointer_cast(indicesDevice.data()),
      thrust::raw_pointer_cast(centroidsDevice.data()),
      thrust::raw_pointer_cast(countsDevice.data())
  );

  centroidsHostFirst = centroidsDevice;

  // Setup data for "gpu2"
  dataHost[0] = 4.0f; dataHost[1] = 4.0f; // [4,4]
  dataHost[2] = 1.0f; dataHost[3] = 4.0f; // [1,4]
  dataHost[4] = 1.0f; dataHost[5] = 2.0f; // [1,2]
  countsDevice[0] = 0; countsDevice[1] = 0;
  labelsHost[0] = 0; labelsHost[1] = 1; labelsHost[2] = 1;
  centroidsHost[0] = 0.0f; centroidsHost[1] = 0.0f; centroidsHost[2] = 0.0f; centroidsHost[3] = 0.0f;
  dataDevice = dataHost;
  labelsDevice = labelsHost;
  centroidsDevice = centroidsHost;

  kmeans::detail::memzero(countsDevice); kmeans::detail::memzero(centroidsDevice);

  // Run on "gpu2"
  kmeans::detail::calculate_centroids <<< dim3(1, 30), dim3(n_threads_x, n_threads_y), 0 >>> (
      n, d, k,
      thrust::raw_pointer_cast(dataDevice.data()),
      thrust::raw_pointer_cast(labelsDevice.data()),
      thrust::raw_pointer_cast(indicesDevice.data()),
      thrust::raw_pointer_cast(centroidsDevice.data()),
      thrust::raw_pointer_cast(countsDevice.data())
  );

  centroidsHost = centroidsDevice;

  centroidsHost.data()[0] += centroidsHostFirst.data()[0];
  centroidsHost.data()[1] += centroidsHostFirst.data()[1];
  centroidsHost.data()[2] += centroidsHostFirst.data()[2];
  centroidsHost.data()[3] += centroidsHostFirst.data()[3];

  /**
   * CALCULATE CENTROIDS IN 1 TURN, FROM ALL DATA
   * */
  k = 2; d = 2; n = 6;

  // Setup data
  thrust::host_vector<float> dataHost2(n*d);
  dataHost2[0] = 4.0f; dataHost2[1] = 2.0f; // [0,0]
  dataHost2[2] = 1.0f; dataHost2[3] = 0.0f; // [0,1]
  dataHost2[4] = 4.0f; dataHost2[5] = 0.0f; // [1,1]
  dataHost2[6] = 4.0f; dataHost2[7] = 4.0f; // [1,1]
  dataHost2[8] = 1.0f; dataHost2[9] = 4.0f; // [1,1]
  dataHost2[10] = 1.0f; dataHost2[11] = 2.0f; // [1,1]

  thrust::device_vector<float> dataDevice2(n*d);
  dataDevice2 = dataHost2;

  // Setup counts
  thrust::device_vector<int> countsDevice2(k);

  // Setup labels
  thrust::host_vector<int> labelsHost2(n);
  labelsHost2[0] = 0; labelsHost2[1] = 0; labelsHost2[2] = 0; labelsHost2[3] = 0; labelsHost2[4] = 1; labelsHost2[5] = 1;
  thrust::device_vector<int> labelsDevice2(n);
  labelsDevice2 = labelsHost2;

  // Setup indices
  thrust::host_vector<int> indicesHost2(n);
  indicesHost2[0] = 0; indicesHost2[1] = 1; indicesHost2[2] = 2; indicesHost2[3] = 3; indicesHost2[4] = 4; indicesHost2[5] = 5;
  thrust::device_vector<int> indicesDevice2(n);
  indicesDevice2 = indicesHost2;

  // Setup centroids
  thrust::device_vector<float> centroidsDevice2(d*k);

  kmeans::detail::memzero(countsDevice2);
  kmeans::detail::memzero(centroidsDevice2);

  kmeans::detail::calculate_centroids <<< dim3(1, 30), dim3(n_threads_x, n_threads_y), 0 >>> (
    n, d, k,
    thrust::raw_pointer_cast(dataDevice2.data()),
    thrust::raw_pointer_cast(labelsDevice2.data()),
    thrust::raw_pointer_cast(indicesDevice2.data()),
    thrust::raw_pointer_cast(centroidsDevice2.data()),
    thrust::raw_pointer_cast(countsDevice2.data())
  );

  // THEN
  thrust::host_vector<float> centroidsHost2(d*k);
  centroidsHost2 = centroidsDevice2;

  ASSERT_FLOAT_EQ(centroidsHost2.data()[0], centroidsHost.data()[0]);
  ASSERT_FLOAT_EQ(centroidsHost2.data()[1], centroidsHost.data()[1]);
  ASSERT_FLOAT_EQ(centroidsHost2.data()[2], centroidsHost.data()[2]);
  ASSERT_FLOAT_EQ(centroidsHost2.data()[3], centroidsHost.data()[3]);

  SUCCEED();

}

TEST(KMeansCentroids, RevertCentroidZeroing) {
  // GIVEN
  int k = 3;
  int d = 2;

  // Setup counts
  thrust::host_vector<int> countsHost(k);
  countsHost[0] = 1;
  countsHost[1] = 0;
  countsHost[2] = 1;
  thrust::device_vector<int> countsDevice(k);
  countsDevice = countsHost;

  // Setup tmp centroids (original)
  thrust::host_vector<float> tmp_centroidsHost(d*k);
  tmp_centroidsHost[0] = 1.0f;
  tmp_centroidsHost[1] = 1.0f;
  tmp_centroidsHost[2] = 2.0f;
  tmp_centroidsHost[3] = 2.0f;
  tmp_centroidsHost[4] = 3.0f;
  tmp_centroidsHost[5] = 3.0f;
  thrust::device_vector<float> tmp_centroidsDevice(d*k);
  tmp_centroidsDevice = tmp_centroidsHost;

  // Setup centroids
  thrust::host_vector<float> centroidsHost(d*k);
  centroidsHost[0] = 5.0f;
  centroidsHost[1] = 5.0f;
  centroidsHost[2] = 0.0f;
  centroidsHost[3] = 0.0f;
  centroidsHost[4] = 4.0f;
  centroidsHost[5] = 4.0f;
  thrust::device_vector<float> centroidsDevice(d*k);
  centroidsDevice = centroidsHost;

  // WHEN
  kmeans::detail::revert_zeroed_centroids << < dim3((d - 1) / 32 + 1, (k - 1) / 32 + 1), dim3(32, 32), 0 >> > (
  d, k,
  thrust::raw_pointer_cast(tmp_centroidsDevice.data()),
  thrust::raw_pointer_cast(centroidsDevice.data()),
  thrust::raw_pointer_cast(countsDevice.data())
  );

  // THEN
  centroidsHost = centroidsDevice;

  ASSERT_FLOAT_EQ(5.0f, centroidsHost.data()[0]);
  ASSERT_FLOAT_EQ(5.0f, centroidsHost.data()[1]);
  ASSERT_FLOAT_EQ(2.0f, centroidsHost.data()[2]);
  ASSERT_FLOAT_EQ(2.0f, centroidsHost.data()[3]);
  ASSERT_FLOAT_EQ(4.0f, centroidsHost.data()[4]);
  ASSERT_FLOAT_EQ(4.0f, centroidsHost.data()[5]);

  SUCCEED();
}

TEST(KMeansCentroids, CentroidsScaling) {
  // GIVEN
  int k = 2;
  int d = 2;

  // Setup counts
  thrust::host_vector<int> countsHost(k);
  countsHost[0] = 4;
  countsHost[1] = 2;

  thrust::device_vector<int> countsDevice(k);
  countsDevice = countsHost;

  // Setup centroids
  thrust::host_vector<float> centroidsHost(d*k);
  centroidsHost[0] = 1.0f;
  centroidsHost[1] = 2.0f;
  centroidsHost[2] = 3.0f;
  centroidsHost[3] = 4.0f;

  thrust::device_vector<float> centroidsDevice(d*k);
  centroidsDevice = centroidsHost;

  // WHEN
  kmeans::detail::scale_centroids << < dim3((d - 1) / 32 + 1, (k - 1) / 32 + 1), dim3(32, 32), 0 >> > (
  d, k,
  thrust::raw_pointer_cast(countsDevice.data()),
  thrust::raw_pointer_cast(centroidsDevice.data())
  );

  // THEN
  centroidsHost = centroidsDevice;

  ASSERT_FLOAT_EQ(0.25f, centroidsHost.data()[0]);
  ASSERT_FLOAT_EQ(0.5f, centroidsHost.data()[1]);
  ASSERT_FLOAT_EQ(1.5f, centroidsHost.data()[2]);
  ASSERT_FLOAT_EQ(2.0f, centroidsHost.data()[3]);

  SUCCEED();
}