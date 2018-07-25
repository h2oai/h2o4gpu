#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <vector>

#include "../../../../src/gpu/matrix/KmMatrix/KmMatrix.hpp"
#include "../../../../src/gpu/matrix/KmMatrix/Arith.hpp"

#include <iostream>

using namespace h2o4gpu::Matrix;

constexpr float esp = 0.001f;

TEST(KmMatrix, ArithDot) {
  thrust::host_vector<float> h_data (9);
  for (size_t i = 0; i < 9; ++i) {
    h_data[i] = (float) i * i;
  }
  KmMatrix<float> mat (h_data, 3, 3);
  KmMatrix<float> res (3, 3);

  DotOp<float>().dot(res, mat);

  std::vector<float> answer_vec
    {
      153.0f, 212.0f, 281.0f,
      1044.0f, 1490.0f, 2036.0f,
      2745.0f, 3956.0f, 5465.0f
    };

  KmMatrix<float> answer (answer_vec, 3, 3);

  ASSERT_TRUE(answer == res);
}

TEST(KmMatrix, ArithVecBatchDot) {
  thrust::host_vector<float> h_data (20);
  for (size_t i = 0; i < 20; ++i) {
    h_data[i] = (float) i * i;
  }

  KmMatrix<float> data (h_data, 4, 5);

  KmMatrix<float> res (4, 1);
  VecBatchDotOp<float>().dot(res, data);

  thrust::host_vector<float> h_sol (4);
  h_sol[0] = 354;
  h_sol[1] = 14979;
  h_sol[2] = 112354;
  h_sol[3] = 434979;
  KmMatrix<float> sol (h_sol, 4, 1);

  ASSERT_TRUE(res == sol);
}

TEST(KmMatrix, ArithSum) {
  thrust::host_vector<float> h_data (16);
  for (size_t i = 0; i < 16; ++i) {
    h_data[i] = (float) i * i;
  }
  KmMatrix<float> mat (h_data, 4, 4);
  float res = SumOp<float>().sum(mat);
  EXPECT_NEAR(res, 1240.0f, esp);
}

TEST(KmMatrix, ArithMean) {
  thrust::host_vector<float> h_data (16);
  for (size_t i = 0; i < 16; ++i) {
    h_data[i] = (float) i * i;
  }
  KmMatrix<float> mat (h_data, 4, 4);
  float res = MeanOp<float>().mean(mat);
  EXPECT_NEAR(res, 77.5, esp);
}

TEST(KmMatrix, ArithMul) {
  thrust::host_vector<float> h_data (16);
  for (size_t i = 0; i < 16; ++i) {
    h_data[i] = (float) i * i;
  }
  KmMatrix<float> mat (h_data, 4, 4);
  KmMatrix<float> res (4, 4);
  MulOp<float>().mul(res, mat, 2.0f);

  thrust::host_vector<float> h_sol(16);
  for (size_t i = 0; i < 16; ++i) {
    h_sol[i] = h_data[i] * 2.0f;
  }
  KmMatrix<float> solution {h_sol, 4, 4};

  ASSERT_TRUE(res == solution);
}

TEST(KmMatrix, ArithArgMin) {
  std::vector<float> h_data
    {
      1.0f, 3.0f, 2.0f, 0.0f,
      3.0f, 1.0f, 0.0f, 2.0f,
      1.0f, 1.0f, 1.0f, 1.0f
    };

  KmMatrix<float> mat (h_data, 3, 4);
  KmMatrix<int> res = ArgMinOp<float>().argmin(mat, KmMatrixDim::ROW);

  std::vector<int> solution_vec {3, 2, 0};
  KmMatrix<int> solution (solution_vec, 3, 1);

  ASSERT_TRUE(res == solution);
}

TEST(KmMatrix, ArithMin) {
  std::vector<float> h_data
    {
      1.0f, 3.0f, 2.0f, 0.0f,
      3.0f, 1.0f, 0.0f, 2.0f,
      1.0f, 1.0f, 1.0f, 1.0f
    };
  KmMatrix<float> mat (h_data, 3, 4);

  KmMatrix<float> res = MinOp<float>().min(mat, KmMatrixDim::ROW);

   std::vector<float> solution_vec {0.0f, 0.0f, 1.0f};
  KmMatrix<float> solution (solution_vec, 3, 1);

  ASSERT_TRUE(res == solution);
}