#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include "../../../../src/gpu/matrix/KmMatrix/KmMatrix.hpp"

// r --gtest_filter=KmMatrix.KmMatrixHostProxy
TEST(KmMatrix, KmMatrixProxyHostEqual) {
  size_t rows = 12, cols = 16;
  thrust::host_vector<double> vec (rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    vec[i] = i;
  }

  h2o4gpu::Matrix::KmMatrix<double> mat (vec, rows, cols);

  h2o4gpu::Matrix::KmMatrix<double> row = mat.row(1);

  thrust::host_vector<double> res (cols);

  for (size_t i = 0, v = 16; v < 32; ++i, ++v) {
    res[i] = v;
  }

  h2o4gpu::Matrix::KmMatrix<double> res_mat (res, 1, cols);

  ASSERT_TRUE(res_mat == row);
}

// r --gtest_filter=KmMatrix.KmMatrixDevProxy
// FIXME
TEST(KmMatrix, KmMatrixProxyDevEqual) {
  size_t rows = 12, cols = 16;
  thrust::host_vector<double> vec (rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    vec[i] = i;
  }

  h2o4gpu::Matrix::KmMatrix<double> mat (vec, rows, cols);
  mat.set_name ("mat");

  mat.dev_ptr();

  h2o4gpu::Matrix::KmMatrix<double> row = mat.row(1);
  row.set_name ("row");

  thrust::host_vector<double> res (cols);

  for (size_t i = 0, v = 16; v < 16 + cols; ++i, ++v) {
    res[i] = v;
  }

  h2o4gpu::Matrix::KmMatrix<double> res_mat (res, 1, cols);
  res_mat.set_name("res");

  ASSERT_TRUE(res_mat == row);
}
