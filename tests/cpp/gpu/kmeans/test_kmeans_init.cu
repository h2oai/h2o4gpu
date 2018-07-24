/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <gtest/gtest.h>

#include "../../../../src/gpu/kmeans/KmMatrix/KmMatrix.hpp"
#include "../../../../src/gpu/kmeans/KmMatrix/Generator.hpp"
#include "../../../../src/gpu/kmeans/KmMatrix/Arith.hpp"
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

TEST(KmeansLL, PairWiseDistance) {

  thrust::host_vector<double> h_data (20);
  for (size_t i = 0; i < 20; ++i) {
    h_data[i] = i * 2;
  }
  KmMatrix<double> data (h_data, 4, 5);
  data.set_name ("data");

  thrust::host_vector<double> h_centroids(10);
  for (size_t i = 0; i < 10; ++i) {
    h_centroids[i] = i;
  }
  KmMatrix<double> centroids (h_centroids, 2, 5);
  centroids.set_name ("centroids");

  KmMatrix<double> data_dot (4, 1);
  data_dot.set_name ("data_dot");
  VecBatchDotOp<double>().dot(data_dot, data);
  std::cout << data_dot << std::endl;

  KmMatrix<double> centroids_dot (2, 1);
  centroids_dot.set_name ("centroids dot");
  VecBatchDotOp<double>().dot(centroids_dot, centroids);
  std::cout << centroids_dot << std::endl;


  thrust::host_vector<float> h_pairs (8);
  for (size_t i = 0; i < 8; ++i) {
    h_pairs[i] = 1;
  }
  KmMatrix<double> distance_pairs (h_pairs, 4, 2);

  KmMatrix<double> res = detail::PairWiseDistanceOp<double>(
      data_dot, centroids_dot, distance_pairs)(data, centroids);
  res.set_name ("pw res");
  std::cout << res << std::endl;

  std::vector<float> h_sol
    {
      151, 376,
      1051, 1276,
      2951, 3176,
      5851, 6076
    };
  KmMatrix<double> sol (h_sol, 4, 2);

  ASSERT_TRUE(sol == res);
}

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

  // auto result = kmeans_ll_init(h_data, 2);
  // result.set_name("kmeans with mock");
  // std::cout << result << std::endl;
}
