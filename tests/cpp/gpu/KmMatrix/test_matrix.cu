#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include "../../../../src/gpu/kmeans/KmMatrix/KmMatrix.hpp"
#include <cmath>

// r --gtest_filter=KmMatrix.KmMatrixEqual
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

TEST(KmMatrix, KmMatrixRows) {
  thrust::host_vector<double> vec (12 * 16);
  for (size_t i = 0; i < 12 * 16; ++i) {
    vec[i] = i;
  }
  H2O4GPU::KMeans::KmMatrix<double> mat (vec, 12, 16);

  thrust::host_vector<double> h_index (4, 1);
  h_index[0] = 0;
  h_index[1] = 2;
  h_index[2] = 9;
  h_index[3] = 1;
  H2O4GPU::KMeans::KmMatrix<double> index (h_index, 4, 1);

  H2O4GPU::KMeans::KmMatrix<double> rows = mat.rows(index);

  thrust::host_vector<double> h_sol (4 * 16);
  for (size_t i = 0; i < 16; ++i) {
    h_sol[i] = vec[i];
  }
  for (size_t i = 16; i < 32; ++i) {
    h_sol[i] = vec[16 * 2 + (i - 16)];
  }
  for (size_t i = 32; i < 48; ++i) {
    h_sol[i] = vec[16 * 9 + (i - 32)];
  }
  for (size_t i = 48; i < 64; ++i) {
    h_sol[i] = vec[16 * 1 + (i - 48)];
  }

  H2O4GPU::KMeans::KmMatrix<double> sol (h_sol, 4, 16);

  ASSERT_TRUE(rows == sol);
}

TEST(KmMatrix, SizeError) {
  thrust::host_vector<double> vec (12 * 16);
  ASSERT_THROW(
      H2O4GPU::KMeans::KmMatrix<double> mat (vec, 12, 4),
      std::runtime_error);

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

TEST(KmMatrix, KmMatrixCycle) {
  size_t rows = 2048, cols = 1024;
  thrust::host_vector<double> vec (rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    vec[i] = i;
  }
  // Tweak this one to see if memory grows, there should be a better way to
  // test memory leak.
  size_t iters = std::pow(16, 1);
  H2O4GPU::KMeans::KmMatrix<double> mat0 (vec, rows, cols);
  mat0.dev_ptr();
  for (size_t i = 0; i < iters; ++i) {
    H2O4GPU::KMeans::KmMatrix<double> mat1 = mat0;
    H2O4GPU::KMeans::KmMatrix<double> mat2 = mat1;
    mat0 = mat2;
  }
}

// r --gtest_filter=KmMatrix.Stack
TEST(KmMatrix, Stack) {
  constexpr size_t rows = 16, cols = 16;
  thrust::host_vector<double> vec (rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    vec[i] = i;
  }
  H2O4GPU::KMeans::KmMatrix<double> mat(vec, rows, cols);

  thrust::host_vector<double> vec1 (rows * cols);
  for (size_t i = rows * cols; i < 2 * rows * cols; ++i) {
    vec1[i - rows * cols] = i;
  }
  H2O4GPU::KMeans::KmMatrix<double> mat1(vec1, rows, cols);

  H2O4GPU::KMeans::KmMatrix<double> calculated =
      H2O4GPU::KMeans::stack(mat, mat1, H2O4GPU::KMeans::KmMatrixDim::ROW);

  thrust::host_vector<double> res (2 * rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    res[i] = i;
  }
  for (size_t i = rows * cols; i < 2 * rows * cols; ++i) {
    res[i] = i;
  }

  H2O4GPU::KMeans::KmMatrix<double> res_mat (res, 2 * rows, cols);

  ASSERT_TRUE(calculated == res_mat);
}
