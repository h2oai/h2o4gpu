/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <gtest/gtest.h>

#include "../../../../src/gpu/kmeans/array.cuh"
#include "../../../../src/gpu/kmeans/kmeans_init.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

TEST(KmeansLL, KmeansLLInit) {

  int k = 2;

  H2O4GPU::Array::Dims dims {4, 4, 0, 0};
  H2O4GPU::KMeans::KmeansLlInit<double> kmeans_ll_init;

  thrust::host_vector<double> _h_data (16);

  for (size_t i = 0; i < 4; ++i) {
    _h_data[i] = i;
  }
  for (size_t i = 4; i < 8; ++i) {
    _h_data[i] = i - 2;
  }

  for (size_t i = 8; i < 12; ++i) {
    _h_data[i] = i;
  }
  for (size_t i = 12; i < 16; ++i) {
    _h_data[i] = i + 2;
  }

  thrust::device_vector<double> _d_data;
  _d_data = _h_data;

  H2O4GPU::Array::CUDAArray<double> data (_d_data, dims);

  kmeans_ll_init (data);

  std::cout << "Host" << std::endl;
  for (size_t i = 0; i < 16; ++i) {
    std::cout << _h_data[i] << ',';
  }
  std::cout << std::endl;
}