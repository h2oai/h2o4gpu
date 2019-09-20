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

TEST(ARIMA, applyAR) {
  const int length = 10;
  thrust::device_vector<float> ts_data(length);
  for (auto i = 0; i < length; ++i) ts_data[i] = float(i);

  const int p = 2;
  thrust::device_vector<float> phi(p);
  phi[0] = 1.0;
  phi[1] = 0.5;

  thrust::device_vector<float> res(length * p, 0);

  h2o4gpu::ARIMAModel<float>::ApplyAR(thrust::raw_pointer_cast(res.data()),
                                      thrust::raw_pointer_cast(ts_data.data()),
                                      thrust::raw_pointer_cast(phi.data()), p,
                                      length);

  thrust::host_vector<float> h_res = res;

  ASSERT_FLOAT_EQ(-2, h_res[0]);
  ASSERT_FLOAT_EQ(-2.5, h_res[1]);
  ASSERT_FLOAT_EQ(-3, h_res[2]);
  ASSERT_FLOAT_EQ(-3.5, h_res[3]);
  ASSERT_FLOAT_EQ(-4, h_res[4]);
  ASSERT_FLOAT_EQ(-4.5, h_res[5]);
  ASSERT_FLOAT_EQ(-5, h_res[6]);
  ASSERT_FLOAT_EQ(-5.5, h_res[7]);
  ASSERT_FLOAT_EQ(0, h_res[8]);
  ASSERT_FLOAT_EQ(0, h_res[9]);
}

TEST(ARIMA, d_0_p_1_q_0) {
  const int length = 10;
  thrust::device_vector<float> ts_data(length);
  for (auto i = 0; i < length; ++i) ts_data[i] = float(i / 2);

  h2o4gpu::ARIMAModel<float> model(2, 0, 0, length);

  model.Fit(thrust::raw_pointer_cast(ts_data.data()));
}