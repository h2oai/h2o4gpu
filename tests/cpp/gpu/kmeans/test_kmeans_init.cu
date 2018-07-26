/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <gtest/gtest.h>

#include "../../../../src/gpu/matrix/KmMatrix/KmMatrix.hpp"
#include "../../../../src/gpu/matrix/KmMatrix/Generator.hpp"
#include "../../../../src/gpu/matrix/KmMatrix/Arith.hpp"
#include "../../../../src/gpu/kmeans/kmeans_init.cuh"
#include "../../../../src/common/utils.h"

#include <thrust/device_vector.h>
#include <iostream>
#include <memory>

using namespace h2o4gpu::kMeans;
using namespace h2o4gpu::Matrix;

template <typename T>
struct GeneratorMock : RandomGeneratorBase<T> {
 public:
  KmMatrix<T> generate() override {
    h2o4gpu_error("Not implemented");
    KmMatrix<T> res;
    return res;
  }

  KmMatrix<T> generate(size_t _size) override {
    thrust::host_vector<T> random_numbers (_size);
    for (size_t i = 0; i < _size; ++i) {
      if ( i % 2 == 0)
        random_numbers[i] = 0.8;
      else
        random_numbers[i] = 0.2;
    }
    KmMatrix<T> res (random_numbers, 1, _size);
    return res;
  }
};

TEST(KmeansRandom, Init) {
  thrust::host_vector<float> h_data (20);
  for (size_t i = 0; i < 20; ++i) {
    h_data[i] = i * 2;
  }
  KmMatrix<float> data (h_data, 4, 5);
  std::unique_ptr<RandomGeneratorBase<float>> gen (new GeneratorMock<float>());
  KmeansRandomInit<float> init (gen);

  auto res = init(data, 2);

  std::vector<float> h_sol =
      {
        30, 32, 34, 36, 38,
        0,  2,  4,  6,  8
      };
  KmMatrix<float> sol (h_sol, 2, 5);
  ASSERT_TRUE(sol == res);
}

// r --gtest_filter=KmeansLL.PairWiseDistance
TEST(KmeansLL, PairWiseDistance) {

  thrust::host_vector<double> h_data (20);
  for (size_t i = 0; i < 20; ++i) {
    h_data[i] = i * 2;
  }
  KmMatrix<double> data (h_data, 4, 5);

  thrust::host_vector<double> h_centroids(10);
  for (size_t i = 0; i < 10; ++i) {
    h_centroids[i] = i;
  }
  KmMatrix<double> centroids (h_centroids, 2, 5);

  KmMatrix<double> data_dot (4, 1);
  VecBatchDotOp<double>().dot(data_dot, data);

  KmMatrix<double> centroids_dot (2, 1);
  VecBatchDotOp<double>().dot(centroids_dot, centroids);

  thrust::host_vector<float> h_pairs (8);
  for (size_t i = 0; i < 8; ++i) {
    h_pairs[i] = 0;
  }
  KmMatrix<double> distance_pairs (h_pairs, 4, 2);

  KmMatrix<double> res = detail::PairWiseDistanceOp<double>(
      data_dot, centroids_dot, distance_pairs)(data, centroids);

  std::vector<float> h_sol
    {
      30.,   55.,
      730.,  255.,
      2430., 1455.,
      5130., 3655.,
    };
  KmMatrix<double> sol (h_sol, 4, 2);

  ASSERT_TRUE(sol == res);
}

// r --gtest_filter=KmeansLL.GreedyRecluster
TEST(KmeansLL, GreedyRecluster) {
  thrust::host_vector<double> h_centroids (20);
  // close points
  for (size_t i = 0; i < 5; ++i) {
    h_centroids[i] = i + 4;
  }
  for (size_t i = 5; i < 10; ++i) {
    h_centroids[i] = i;
  }
  for (size_t i = 10; i < 15; ++i) {
    h_centroids[i] = i - 4;
  }
  // satelite
  for (size_t i = 15; i < 20; ++i) {
    h_centroids[i] = i;
  }

  KmMatrix<double> centroids (h_centroids, 4, 5);
  KmMatrix<double> res = detail::GreedyRecluster<double>::recluster(centroids,
                                                                    2);
  std::vector<double> h_sol =
      {
        4,    5,    6,    7,    8,
        5,    6,    7,    8,    9,
      };
  KmMatrix<double> sol (h_sol, 2, 5);
  ASSERT_TRUE(res == sol);
}

// r --gtest_filter=KmeansLL.KmeansLLInit
TEST(KmeansLL, KmeansLLInit) {
  std::unique_ptr<RandomGeneratorBase<double>> mock_ptr (new GeneratorMock<double>);
  KmeansLlInit<double> kmeans_ll_init (mock_ptr, 2.5);
  thrust::host_vector<double> h_data (30);
  // We split the points into two groups, but the result is statistic.
  for (size_t i = 0; i < 5; ++i) {
    h_data[i] = i + 4;
  }
  for (size_t i = 5; i < 10; ++i) {
    h_data[i] = i;
  }
  for (size_t i = 10; i < 15; ++i) {
    h_data[i] = i - 4;
  }


  for (size_t i = 15; i < 20; ++i) {
    h_data[i] = i + 4;
  }
  for (size_t i = 20; i < 25; ++i) {
    h_data[i] = i;
  }
  for (size_t i = 25; i < 30; ++i) {
    h_data[i] = i - 4;
  }

  KmMatrix<double> data (h_data, 6, 5);

  auto res = kmeans_ll_init(data, 2);

  std::vector<double> h_sol =
      {
        4,    5,    6,    7,    8,
        19,   20,   21,   22,   23,
      };
  KmMatrix<double> sol (h_sol, 2, 5);
  ASSERT_TRUE(sol == res);
}
