#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../gpu/arima/arima.h"
#include "cuda_utils2.h"

TEST(ARIMA, ts_data_to_matrix) {
  const int length = 7;
  const int depth = 3;
  const int lda = 6;
  thrust::device_vector<float> ts_data(length);
  for (auto i = 0; i < length; ++i) ts_data[i] = float(i);
  thrust::device_vector<float> A(depth * lda, -1.0);

  h2o4gpu::ARIMAModel<float>::AsMatrix(thrust::raw_pointer_cast(ts_data.data()),
                                       thrust::raw_pointer_cast(A.data()),
                                       depth, lda, length);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_A = A;

  ASSERT_FLOAT_EQ(0.0f, h_A[0]);
  ASSERT_FLOAT_EQ(1.0f, h_A[1]);
  ASSERT_FLOAT_EQ(2.0f, h_A[2]);
  ASSERT_FLOAT_EQ(3.0f, h_A[3]);
  ASSERT_FLOAT_EQ(4.0f, h_A[4]);
  ASSERT_FLOAT_EQ(-1.0f, h_A[5]);

  ASSERT_FLOAT_EQ(1.0f, h_A[6]);
  ASSERT_FLOAT_EQ(2.0f, h_A[7]);
  ASSERT_FLOAT_EQ(3.0f, h_A[8]);
  ASSERT_FLOAT_EQ(4.0f, h_A[9]);
  ASSERT_FLOAT_EQ(5.0f, h_A[10]);
  ASSERT_FLOAT_EQ(-1.0f, h_A[11]);

  ASSERT_FLOAT_EQ(2.0f, h_A[12]);
  ASSERT_FLOAT_EQ(3.0f, h_A[13]);
  ASSERT_FLOAT_EQ(4.0f, h_A[14]);
  ASSERT_FLOAT_EQ(5.0f, h_A[15]);
  ASSERT_FLOAT_EQ(6.0f, h_A[16]);
  ASSERT_FLOAT_EQ(-1.0f, h_A[17]);
}