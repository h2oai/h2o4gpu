#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../gpu/arima/arima.h"
#include "cuda_utils2.h"

TEST(ARIMA, least_squares_solver_2x2) {
  const int rows = 2;
  const int cols = 2;
  thrust::device_vector<float> A(rows * cols, 0.0);
  thrust::device_vector<float> B(rows, 0.0);

  for (auto i = 0; i < rows * cols; ++i) A[i] = float(i + 1);
  B[0] = 1;
  B[1] = 0;

  //   for (auto i = 0; i < rows; ++i) B[i] = float(i + 1);

  h2o4gpu::LeastSquaresSolver solver(rows, cols);
  solver.Solve(thrust::raw_pointer_cast(A.data()),
               thrust::raw_pointer_cast(B.data()));

  OK(cudaDeviceSynchronize());

  thrust::host_vector<float> h_B = B;

  // A stored in column-major order
  //  || 1  3 || * x = || 1 ||
  //  || 2  4 ||       || 0 ||

  ASSERT_FLOAT_EQ(-2.0f, h_B[0]);
  ASSERT_FLOAT_EQ(1.0f, h_B[1]);
}

TEST(ARIMA, least_squares_solver_3x2) {
  const int rows = 3;
  const int cols = 2;
  thrust::device_vector<float> A(rows * cols);
  thrust::device_vector<float> B(rows);

  for (auto i = 0; i < rows * cols; ++i) A[i] = float(i + 1);
  B[0] = 1;
  B[1] = 0;
  B[2] = 0;

  //   for (auto i = 0; i < rows; ++i) B[i] = float(i + 1);

  h2o4gpu::LeastSquaresSolver solver(rows, cols);
  solver.Solve(thrust::raw_pointer_cast(A.data()),
               thrust::raw_pointer_cast(B.data()));

  OK(cudaDeviceSynchronize());

  thrust::host_vector<float> h_B = B;

  // A stored in column-major order
  //  || 1  4 || * x = || 1 ||
  //  || 2  5 ||       || 0 ||
  //  || 3  6 ||       || 0 ||

  ASSERT_FLOAT_EQ(-0.94444454f, h_B[0]);
  ASSERT_FLOAT_EQ(0.44444454f, h_B[1]);
}