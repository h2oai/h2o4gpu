#include "gtest/gtest.h"

#include "src/gpu/kmeans/kmeans_centroids.h"
#include <thrust/host_vector.h>

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

int
main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}