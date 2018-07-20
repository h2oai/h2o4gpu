/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <gtest/gtest.h>

// #include "../../../../src/gpu/kmeans/Eigen/Dense"
#include "../../../../src/gpu/kmeans/KmMatrix/KmMatrix.hpp"
#include "../../../../src/gpu/kmeans/kmeans_init.cuh"

#include <thrust/device_vector.h>
#include <iostream>

TEST(KmeansLL, KmeansLLInit) {

  int k = 2;

  H2O4GPU::KMeans::KmeansLlInit<double> kmeans_ll_init;

  thrust::host_vector<double>  _h_data (16);

  for (size_t i = 0; i < 4; ++i) {
    _h_data[i] = double(i);
  }
  for (size_t i = 4; i < 8; ++i) {
    _h_data[i] = double(i - 2);
  }

  for (size_t i = 8; i < 12; ++i) {
    _h_data[i] = double(i);
  }
  for (size_t i = 12; i < 16; ++i) {
    _h_data[i] = double(i + 2);
  }

  H2O4GPU::KMeans::KmMatrix<double> h_data (_h_data, 4, 4);

  auto result = kmeans_ll_init (h_data, 1.0f);
  result.set_name ("kmeans result");
  std::cout << result << std::endl;
}