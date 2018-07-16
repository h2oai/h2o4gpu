#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include "../../../../src/gpu/kmeans/KmMatrix/KmMatrix.hpp"

// r --gtest_filter=KmMatrix.KmMatrix
TEST(KmMatrix, KmMatrixEqual) {
  thrust::host_vector<double> vec (2048 * 1024);
  for (size_t i = 0; i < 2048 * 1024; ++i) {
    vec[i] = i;
  }
  H2O4GPU::KMeans::KmMatrix<double> mat (vec, 2048, 1024);

  ASSERT_TRUE (mat == mat);

  thrust::host_vector<double> vec2 (2048 * 1024);
  for (size_t i = 0; i < 2048 * 1024; ++i) {
    vec2[i] = i + i;
  }
  H2O4GPU::KMeans::KmMatrix<double> mat2 (vec2, 2048, 1024);

  ASSERT_FALSE(mat == mat2);
}

TEST(KmMatrix, KmMatrixAssig) {
  thrust::host_vector<double> vec (2048 * 1024);
  for (size_t i = 0; i < 2048 * 1024; ++i) {
    vec[i] = i;
  }

  H2O4GPU::KMeans::KmMatrix<double> mat0 (vec, 2048, 1024);
  H2O4GPU::KMeans::KmMatrix<double> mat1 = mat0;
  H2O4GPU::KMeans::KmMatrix<double> mat2;

  mat2 = mat0;

  ASSERT_TRUE(mat0 == mat1);
  ASSERT_TRUE(mat1 == mat2);
}

TEST(KmMatrix, KmMatrixUtils) {
  thrust::host_vector<double> vec (12 * 16);
  H2O4GPU::KMeans::KmMatrix<double> mat (vec, 12, 16);

  ASSERT_EQ(mat.rows(), 12);
  ASSERT_EQ(mat.cols(), 16);
  ASSERT_EQ(mat.size(), 12 * 16);
}

TEST(KmMatrix, KmMatrixKparam) {
  thrust::host_vector<double> vec (12 * 16);
  thrust::fill(vec.begin(), vec.end(), 1);
  H2O4GPU::KMeans::KmMatrix<double> mat (vec, 12, 16);

  H2O4GPU::KMeans::kParam<double> param = mat.k_param();
  ASSERT_EQ(param.ptr, mat.dev_ptr());
  ASSERT_EQ(param.rows, 12);
  ASSERT_EQ(param.cols, 16);
}