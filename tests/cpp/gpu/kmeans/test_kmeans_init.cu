/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <gtest/gtest.h>

#include "../../../../src/gpu/kmeans/KmMatrix/KmMatrix.hpp"
#include "../../../../src/gpu/kmeans/KmMatrix/Generator.hpp""
#include "../../../../src/gpu/kmeans/kmeans_init.cuh"

#include <thrust/device_vector.h>
#include <iostream>
#include <memory>

using namespace H2O4GPU::KMeans;

template <typename T>
struct GeneratorMock : GeneratorBase<T> {
 public:
  KmMatrix<T> generate() override {}

  KmMatrix<T> generate(size_t _size) override {
    thrust::host_vector<T> random_numbers (_size);
    for (size_t i = 0; i < _size; ++i) {
      random_numbers[i] = 1 / _size;
    }
    KmMatrix<T> res (random_numbers, 1, _size);
    return res;
  }
};

TEST(KmeansLL, KmeansLLInit) {
  int k = 2;
  std::unique_ptr<GeneratorBase<double>> mock_ptr (new GeneratorMock<double>);
  KmeansLlInit<double> kmeans_ll_init (mock_ptr, 2.5);

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

  auto result = kmeans_ll_init(h_data, 1.0f);
  result.set_name("kmeans with mock");
  std::cout << result << std::endl;
}
