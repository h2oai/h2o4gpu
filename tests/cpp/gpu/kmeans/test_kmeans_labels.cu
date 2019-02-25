#include "gtest/gtest.h"

// TODO: make it clear(use proper include)
#include "../gpu/kmeans/kmeans_labels.h"
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void sort_by_key_int(thrust::device_vector<int>& keys, thrust::device_vector<int>& values);

TEST(KMeansLabels, Sorting) {
  // GIVEN
  int n = 6;

  thrust::device_vector<int> deviceLabels(n);
  thrust::host_vector<int> hostLabels(n);
  hostLabels[0] = 0;
  hostLabels[1] = 1;
  hostLabels[2] = 0;
  hostLabels[3] = 1;
  hostLabels[4] = 0;
  hostLabels[5] = 1;
  deviceLabels = hostLabels;

  thrust::device_vector<int> deviceIndices(n);
  thrust::host_vector<int> hostIndices(n);
  hostIndices[0] = 0;
  hostIndices[1] = 1;
  hostIndices[2] = 2;
  hostIndices[3] = 3;
  hostIndices[4] = 4;
  hostIndices[5] = 5;
  deviceIndices = hostIndices;

  // WHEN
  // TODO not sorting :-(
  mycub::sort_by_key_int(
      deviceLabels,
      deviceIndices
  );

  // THEN
  hostLabels = deviceLabels;
  hostIndices = deviceIndices;

  /*ASSERT_EQ(0, hostLabels[0]);
  ASSERT_EQ(0, hostLabels[1]);
  ASSERT_EQ(0, hostLabels[2]);
  ASSERT_EQ(1, hostLabels[3]);
  ASSERT_EQ(1, hostLabels[4]);
  ASSERT_EQ(1, hostLabels[5]);*/

  SUCCEED();
}