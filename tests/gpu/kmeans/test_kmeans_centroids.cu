#include "gtest/gtest.h"

#include "src/gpu/kmeans/kmeans_centroids.h"
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

TEST(KMeansCentroids, CalculateCentroids2GPU) {
  // GIVEN
  int k = 2;
  int d = 2;
  int n = 3;

  thrust::host_vector<float> dataHost(n*d);
  thrust::device_vector<float> dataDevice(n*d);
  thrust::device_vector<int> countsDevice(k);
  thrust::host_vector<int> labelsHost(n);
  thrust::device_vector<int> labelsDevice(n);
  thrust::host_vector<float> centroidsHost(d*k);
  thrust::device_vector<float> centroidsDevice(d*k);
  int n_threads_x = 64;
  int n_threads_y = 16;
  thrust::host_vector<float> finalCentroidsHost(d*k);

  // Setup indices
  thrust::host_vector<int> indicesHost(n);
  indicesHost[0] = 0; indicesHost[1] = 1; indicesHost[2] = 2;
  thrust::device_vector<int> indicesDevice(n);
  indicesDevice = indicesHost;

  // Setup data for "gpu1"
  dataHost[0] = 4.0f; dataHost[1] = 2.0f; // [4,2]
  dataHost[2] = 1.0f; dataHost[3] = 0.0f; // [1,0]
  dataHost[4] = 4.0f; dataHost[5] = 0.0f; // [4,0]
  countsDevice[0] = 0;
  countsDevice[1] = 0;
  labelsHost[0] = 0; // label for [0,0] -> 0
  labelsHost[1] = 0; // label for [0,1] -> 0
  labelsHost[2] = 0; // label for [0,1] -> 0
  centroidsHost[0] = 0.0f; centroidsHost[1] = 0.0f; centroidsHost[2] = 0.0f; centroidsHost[3] = 0.0f;
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

  finalCentroidsHost = centroidsDevice;

  printf("%g ", finalCentroidsHost.data()[0]);
  printf("%g ", finalCentroidsHost.data()[1]);
  printf("%g ", finalCentroidsHost.data()[2]);
  printf("%g\n", finalCentroidsHost.data()[3]);

  // Setup data for "gpu2"
  dataHost[0] = 4.0f; dataHost[1] = 4.0f; // [4,4]
  dataHost[2] = 1.0f; dataHost[3] = 4.0f; // [1,4]
  dataHost[4] = 1.0f; dataHost[5] = 2.0f; // [1,2]
  countsDevice[0] = 0;
  countsDevice[1] = 0;
  labelsHost[0] = 0;
  labelsHost[1] = 1;
  labelsHost[2] = 1;
  centroidsHost[0] = 0.0f; centroidsHost[1] = 0.0f; centroidsHost[2] = 0.0f; centroidsHost[3] = 0.0f;
  dataDevice = dataHost;
  labelsDevice = labelsHost;
  centroidsDevice = centroidsHost;

  kmeans::detail::memzero(countsDevice);
  kmeans::detail::memzero(centroidsDevice);

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

  printf("%g ",  centroidsHost.data()[0]);
  printf("%g ",  centroidsHost.data()[1]);
  printf("%g ",  centroidsHost.data()[2]);
  printf("%g\n", centroidsHost.data()[3]);

  // THEN
/*  ASSERT_FLOAT_EQ(0.0f, finalCentroidsHost.data()[0] + centroidsHost.data()[0]);
  ASSERT_FLOAT_EQ(1.0f, finalCentroidsHost.data()[1] + centroidsHost.data()[1]);
  ASSERT_FLOAT_EQ(2.0f, finalCentroidsHost.data()[2] + centroidsHost.data()[2]);
  ASSERT_FLOAT_EQ(1.0f, finalCentroidsHost.data()[3] + centroidsHost.data()[3]);*/

  SUCCEED();

}

TEST(KMeansCentroids, CalculateCentroids2) {
  // GIVEN
  int k = 2;
  int d = 2;
  int n = 6;

  // Setup data
  thrust::host_vector<float> dataHost(n*d);
  dataHost[0] = 4.0f; dataHost[1] = 2.0f; // [0,0]
  dataHost[2] = 1.0f; dataHost[3] = 0.0f; // [0,1]
  dataHost[4] = 4.0f; dataHost[5] = 0.0f; // [1,1]
  dataHost[6] = 4.0f; dataHost[7] = 4.0f; // [1,1]
  dataHost[8] = 1.0f; dataHost[9] = 4.0f; // [1,1]
  dataHost[10] = 1.0f; dataHost[11] = 2.0f; // [1,1]

  thrust::device_vector<float> dataDevice(n*d);
  dataDevice = dataHost;

  // Setup counts
  thrust::device_vector<int> countsDevice(k);

  // Setup labels
  thrust::host_vector<int> labelsHost(n);
  labelsHost[0] = 0;
  labelsHost[1] = 0;
  labelsHost[2] = 0;
  labelsHost[3] = 0;
  labelsHost[4] = 1;
  labelsHost[5] = 1;
  thrust::device_vector<int> labelsDevice(n);
  labelsDevice = labelsHost;

  // Setup indices
  thrust::host_vector<int> indicesHost(n);
  indicesHost[0] = 0; indicesHost[1] = 1; indicesHost[2] = 2; indicesHost[3] = 3; indicesHost[4] = 4; indicesHost[5] = 5;
  thrust::device_vector<int> indicesDevice(n);
  indicesDevice = indicesHost;

  // Setup centroids
  thrust::device_vector<float> centroidsDevice(d*k);

  kmeans::detail::memzero(countsDevice);
  kmeans::detail::memzero(centroidsDevice);

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
  thrust::host_vector<float> centroidsHost(d*k);
  centroidsHost = centroidsDevice;

  printf("%g ", centroidsHost.data()[0]);
  printf("%g ", centroidsHost.data()[1]);
  printf("%g ", centroidsHost.data()[2]);
  printf("%g ", centroidsHost.data()[3]);

  SUCCEED();

}