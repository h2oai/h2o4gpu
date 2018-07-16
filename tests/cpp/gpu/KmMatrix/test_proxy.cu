#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include "../../../../src/gpu/kmeans/KmMatrix/KmMatrix.hpp"

#include <iostream>

// r --gtest_filter=KmMatrix.KmMatrixProxy
TEST(KmMatrix, KmMatrixProxy) {
  size_t rows = 12, cols = 16;
  thrust::host_vector<double> vec (rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    vec[i] = i;
  }

  H2O4GPU::KMeans::KmMatrix<double> mat (vec, rows, cols);
  mat.set_name ("mat");

  H2O4GPU::KMeans::KmMatrix<double> row = mat.row(1);
  row.set_name ("row");

  thrust::host_vector<double> res (cols);

  for (size_t i = 0, v = 16; v < 32; ++i, ++v) {
    res[i] = v;
    std::cout << v << ' ';
  }

  H2O4GPU::KMeans::KmMatrix<double> res_mat (res, 1, cols);
  res_mat.set_name("res_mat");

  ASSERT_TRUE(res_mat == row);
}

// r --gtest_filter=KmMatrix.KmMatrix
TEST(KmMatrix, KmMatrix) {
  thrust::host_vector<double> vec (2048 * 1024);
  for (size_t i = 0; i < 2048 * 1024; ++i) {
    vec[i] = i;
  }
  H2O4GPU::KMeans::KmMatrix<double> mat (vec, 2048, 1024);

  ASSERT_TRUE (vec == vec);

  thrust::host_vector<double> vec2 (2048 * 1024);
  for (size_t i = 0; i < 2048 * 1024; ++i) {
    vec2[i] = i + 1;
  }

  ASSERT_FALSE(vec == vec2);
}