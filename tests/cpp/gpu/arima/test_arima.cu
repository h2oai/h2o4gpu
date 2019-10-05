#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../gpu/arima/arima.h"
#include "cuda_utils2.h"

TEST(ARIMA, differencing) {
  const int length = 10;
  thrust::device_vector<float> data(length);
  for (auto i = 0; i < length; ++i) data[i] = float(i / 2);
  thrust::device_vector<float> differenced_data(length);

  h2o4gpu::ARIMAModel<float>::Difference(
      thrust::raw_pointer_cast(differenced_data.data()),
      thrust::raw_pointer_cast(data.data()), length);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_differenced_data = differenced_data;

  ASSERT_FLOAT_EQ(0, h_differenced_data[0]);
  ASSERT_FLOAT_EQ(-1, h_differenced_data[1]);
  ASSERT_FLOAT_EQ(0, h_differenced_data[2]);
  ASSERT_FLOAT_EQ(-1, h_differenced_data[3]);
  ASSERT_FLOAT_EQ(0, h_differenced_data[4]);
  ASSERT_FLOAT_EQ(-1, h_differenced_data[5]);
  ASSERT_FLOAT_EQ(0, h_differenced_data[6]);
  ASSERT_FLOAT_EQ(-1, h_differenced_data[7]);
  ASSERT_FLOAT_EQ(0, h_differenced_data[8]);
  ASSERT_TRUE(std::isnan(h_differenced_data[9]));
}

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

TEST(ARIMA, double_ts_data_to_matrix1) {
  const int length = 7;
  const int a_depth = 2;
  const int b_depth = 3;
  const int lda = 10;
  thrust::device_vector<float> ts_a(length);
  for (auto i = 0; i < length; ++i) ts_a[i] = float(i);

  thrust::device_vector<float> ts_b(length);
  for (auto i = 0; i < length; ++i) ts_b[i] = float(i + 1000);

  thrust::device_vector<float> A((a_depth + b_depth) * lda, NAN);

  h2o4gpu::ARIMAModel<float>::AsMatrix(thrust::raw_pointer_cast(ts_a.data()),
                                       thrust::raw_pointer_cast(ts_b.data()),
                                       thrust::raw_pointer_cast(A.data()),
                                       a_depth, b_depth, lda, length);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_A = A;

  ASSERT_FLOAT_EQ(0.000000, h_A[0]);
  ASSERT_FLOAT_EQ(1.000000, h_A[1]);
  ASSERT_FLOAT_EQ(2.000000, h_A[2]);
  ASSERT_FLOAT_EQ(3.000000, h_A[3]);
  ASSERT_FLOAT_EQ(4.000000, h_A[4]);
  ASSERT_FLOAT_EQ(5.000000, h_A[5]);
  ASSERT_TRUE(std::isnan(h_A[6]));
  ASSERT_TRUE(std::isnan(h_A[7]));
  ASSERT_TRUE(std::isnan(h_A[8]));
  ASSERT_TRUE(std::isnan(h_A[9]));
  ASSERT_FLOAT_EQ(1.000000, h_A[10]);
  ASSERT_FLOAT_EQ(2.000000, h_A[11]);
  ASSERT_FLOAT_EQ(3.000000, h_A[12]);
  ASSERT_FLOAT_EQ(4.000000, h_A[13]);
  ASSERT_FLOAT_EQ(5.000000, h_A[14]);
  ASSERT_FLOAT_EQ(6.000000, h_A[15]);
  ASSERT_TRUE(std::isnan(h_A[16]));
  ASSERT_TRUE(std::isnan(h_A[17]));
  ASSERT_TRUE(std::isnan(h_A[18]));
  ASSERT_TRUE(std::isnan(h_A[19]));
  ASSERT_FLOAT_EQ(1000.000000, h_A[20]);
  ASSERT_FLOAT_EQ(1001.000000, h_A[21]);
  ASSERT_FLOAT_EQ(1002.000000, h_A[22]);
  ASSERT_FLOAT_EQ(1003.000000, h_A[23]);
  ASSERT_FLOAT_EQ(1004.000000, h_A[24]);
  ASSERT_TRUE(std::isnan(h_A[25]));
  ASSERT_TRUE(std::isnan(h_A[26]));
  ASSERT_TRUE(std::isnan(h_A[27]));
  ASSERT_TRUE(std::isnan(h_A[28]));
  ASSERT_TRUE(std::isnan(h_A[29]));
  ASSERT_FLOAT_EQ(1001.000000, h_A[30]);
  ASSERT_FLOAT_EQ(1002.000000, h_A[31]);
  ASSERT_FLOAT_EQ(1003.000000, h_A[32]);
  ASSERT_FLOAT_EQ(1004.000000, h_A[33]);
  ASSERT_FLOAT_EQ(1005.000000, h_A[34]);
  ASSERT_TRUE(std::isnan(h_A[35]));
  ASSERT_TRUE(std::isnan(h_A[36]));
  ASSERT_TRUE(std::isnan(h_A[37]));
  ASSERT_TRUE(std::isnan(h_A[38]));
  ASSERT_TRUE(std::isnan(h_A[39]));
  ASSERT_FLOAT_EQ(1002.000000, h_A[40]);
  ASSERT_FLOAT_EQ(1003.000000, h_A[41]);
  ASSERT_FLOAT_EQ(1004.000000, h_A[42]);
  ASSERT_FLOAT_EQ(1005.000000, h_A[43]);
  ASSERT_FLOAT_EQ(1006.000000, h_A[44]);
  ASSERT_TRUE(std::isnan(h_A[45]));
  ASSERT_TRUE(std::isnan(h_A[46]));
  ASSERT_TRUE(std::isnan(h_A[47]));
  ASSERT_TRUE(std::isnan(h_A[48]));
  ASSERT_TRUE(std::isnan(h_A[49]));
}

TEST(ARIMA, double_ts_data_to_matrix2) {
  const int length = 7;
  const int a_depth = 2;
  const int b_depth = 3;
  const int lda = 5;
  thrust::device_vector<float> ts_a(length);
  for (auto i = 0; i < length; ++i) ts_a[i] = float(i);

  thrust::device_vector<float> ts_b(length);
  for (auto i = 0; i < length; ++i) ts_b[i] = float(i + 1000);

  thrust::device_vector<float> A((a_depth + b_depth) * lda, NAN);

  h2o4gpu::ARIMAModel<float>::AsMatrix(thrust::raw_pointer_cast(ts_a.data()),
                                       thrust::raw_pointer_cast(ts_b.data()),
                                       thrust::raw_pointer_cast(A.data()),
                                       a_depth, b_depth, lda, length);
  OK(cudaDeviceSynchronize());
  thrust::host_vector<float> h_A = A;

  ASSERT_FLOAT_EQ(0.000000, h_A[0]);
  ASSERT_FLOAT_EQ(1.000000, h_A[1]);
  ASSERT_FLOAT_EQ(2.000000, h_A[2]);
  ASSERT_FLOAT_EQ(3.000000, h_A[3]);
  ASSERT_FLOAT_EQ(4.000000, h_A[4]);
  ASSERT_FLOAT_EQ(1.000000, h_A[5]);
  ASSERT_FLOAT_EQ(2.000000, h_A[6]);
  ASSERT_FLOAT_EQ(3.000000, h_A[7]);
  ASSERT_FLOAT_EQ(4.000000, h_A[8]);
  ASSERT_FLOAT_EQ(5.000000, h_A[9]);
  ASSERT_FLOAT_EQ(1000.000000, h_A[10]);
  ASSERT_FLOAT_EQ(1001.000000, h_A[11]);
  ASSERT_FLOAT_EQ(1002.000000, h_A[12]);
  ASSERT_FLOAT_EQ(1003.000000, h_A[13]);
  ASSERT_FLOAT_EQ(1004.000000, h_A[14]);
  ASSERT_FLOAT_EQ(1001.000000, h_A[15]);
  ASSERT_FLOAT_EQ(1002.000000, h_A[16]);
  ASSERT_FLOAT_EQ(1003.000000, h_A[17]);
  ASSERT_FLOAT_EQ(1004.000000, h_A[18]);
  ASSERT_FLOAT_EQ(1005.000000, h_A[19]);
  ASSERT_FLOAT_EQ(1002.000000, h_A[20]);
  ASSERT_FLOAT_EQ(1003.000000, h_A[21]);
  ASSERT_FLOAT_EQ(1004.000000, h_A[22]);
  ASSERT_FLOAT_EQ(1005.000000, h_A[23]);
  ASSERT_FLOAT_EQ(1006.000000, h_A[24]);
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

  h2o4gpu::ARIMAModel<float>::Apply(thrust::raw_pointer_cast(res.data()),
                                    thrust::raw_pointer_cast(ts_data.data()),
                                    thrust::raw_pointer_cast(phi.data()), p,
                                    nullptr, nullptr, 0, length);

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

TEST(ARIMA, applyMA) {
  const int length = 10;
  thrust::device_vector<float> last_residual(length);
  thrust::device_vector<float> ts_data(length, 0);
  for (auto i = 0; i < length; ++i) last_residual[i] = float(i % 3);

  const int q = 3;
  thrust::device_vector<float> theta(q);
  theta[0] = 1.0;
  theta[1] = -0.5;
  theta[2] = 0.1;

  thrust::device_vector<float> res(length, 0);

  h2o4gpu::ARIMAModel<float>::Apply(
      thrust::raw_pointer_cast(res.data()),
      thrust::raw_pointer_cast(ts_data.data()), nullptr, 0,
      thrust::raw_pointer_cast(last_residual.data()),
      thrust::raw_pointer_cast(theta.data()), q, length);

  OK(cudaGetLastError());
  OK(cudaDeviceSynchronize());

  thrust::host_vector<float> h_res = res;

  ASSERT_FLOAT_EQ(0.0, h_res[0]);
  ASSERT_FLOAT_EQ(-2.1, h_res[1]);
  ASSERT_FLOAT_EQ(0.3, h_res[2]);
  ASSERT_FLOAT_EQ(0.0, h_res[3]);
  ASSERT_FLOAT_EQ(-2.1, h_res[4]);
  ASSERT_FLOAT_EQ(0.3, h_res[5]);
  ASSERT_FLOAT_EQ(0.0, h_res[6]);
  ASSERT_FLOAT_EQ(0.0, h_res[7]);
  ASSERT_FLOAT_EQ(0, h_res[8]);
  ASSERT_FLOAT_EQ(0, h_res[9]);
}

TEST(ARIMA, applyARMA) {
  const int length = 10;
  thrust::device_vector<float> last_residual(length);
  thrust::device_vector<float> ts_data(length, 0);
  for (auto i = 0; i < length; ++i) {
    ts_data[i] = float(i % 4);
    last_residual[i] = float(i % 3);
  }

  const int p = 2;
  thrust::device_vector<float> phi(p);
  phi[0] = 0.8;
  phi[1] = -0.1;

  const int q = 3;
  thrust::device_vector<float> theta(q);
  theta[0] = 1.0;
  theta[1] = -0.5;
  theta[2] = 0.1;

  thrust::device_vector<float> res(length, 0);

  h2o4gpu::ARIMAModel<float>::Apply(
      thrust::raw_pointer_cast(res.data()),
      thrust::raw_pointer_cast(ts_data.data()),
      thrust::raw_pointer_cast(phi.data()), p,
      thrust::raw_pointer_cast(last_residual.data()),
      thrust::raw_pointer_cast(theta.data()), q, length);

  OK(cudaGetLastError());
  OK(cudaDeviceSynchronize());

  thrust::host_vector<float> h_res = res;

  ASSERT_FLOAT_EQ(-0.6, h_res[0]);
  ASSERT_FLOAT_EQ(-2.4, h_res[1]);
  ASSERT_NEAR(-0.1, h_res[2], 1e-6);
  ASSERT_FLOAT_EQ(3.1, h_res[3]);
  ASSERT_FLOAT_EQ(-2.7, h_res[4]);
  ASSERT_NEAR(0.0, h_res[5], 1e-7);
  ASSERT_FLOAT_EQ(-0.4, h_res[6]);
  ASSERT_NEAR(0.0, h_res[7], 1e-7);
  ASSERT_FLOAT_EQ(0, h_res[8]);
  ASSERT_FLOAT_EQ(0, h_res[9]);
}

TEST(ARIMA, d_0_p_2_q_0) {
  const int length = 10;
  thrust::device_vector<float> ts_data(length);
  for (auto i = 0; i < length; ++i) ts_data[i] = float(i % 3);

  h2o4gpu::ARIMAModel<float> model(2, 0, 0, length);

  model.Fit(thrust::raw_pointer_cast(ts_data.data()));

  ASSERT_FLOAT_EQ(0.34482756, model.Phi()[0]);
  ASSERT_FLOAT_EQ(0.13793102, model.Phi()[1]);
}

TEST(ARIMA, d_0_p_0_q_2_iter_1) {
  const int length = 10;
  thrust::device_vector<float> ts_data(length);
  for (auto i = 0; i < length; ++i) ts_data[i] = float(i % 3);

  h2o4gpu::ARIMAModel<float> model(0, 0, 2, length);

  model.Fit(thrust::raw_pointer_cast(ts_data.data()));

  ASSERT_FLOAT_EQ(0.34482756f, model.Theta()[0]);
  ASSERT_FLOAT_EQ(0.13793102f, model.Theta()[1]);
}

TEST(ARIMA, d_0_p_2_q_2_iter_1) {
  const int length = 7;
  thrust::host_vector<float> h_ts_data(length);
  for (auto i = 0; i < length; ++i)
    h_ts_data[i] = float(i % 5) + 0.1 * float(i % 7 + 1);

  thrust::host_vector<float> ts_data = h_ts_data;

  h2o4gpu::ARIMAModel<float> model(2, 0, 2, length);

  model.Fit(thrust::raw_pointer_cast(ts_data.data()));

  ASSERT_FLOAT_EQ(-2.9589546f, model.Phi()[0]);
  ASSERT_FLOAT_EQ(2.8828485f, model.Phi()[1]);
  ASSERT_FLOAT_EQ(3.9598641f, model.Theta()[0]);
  ASSERT_FLOAT_EQ(-0.61601555f, model.Theta()[1]);
}

TEST(ARIMA, d_0_p_2_q_2_iter_2) {
  const int length = 7;
  thrust::host_vector<float> h_ts_data(length);
  for (auto i = 0; i < length; ++i)
    h_ts_data[i] = float(i % 5) + 0.1 * float(i % 7 + 1);

  thrust::host_vector<float> ts_data = h_ts_data;

  h2o4gpu::ARIMAModel<float> model(2, 0, 2, length);

  model.Fit(thrust::raw_pointer_cast(ts_data.data()), 2);

  ASSERT_FLOAT_EQ(-2.9589546f, model.Phi()[0]);
  ASSERT_FLOAT_EQ(2.8828485f, model.Phi()[1]);
  ASSERT_FLOAT_EQ(3.9598641f, model.Theta()[0]);
  ASSERT_FLOAT_EQ(-0.61601555f, model.Theta()[1]);
}

TEST(ARIMA, d_1_p_1_q_1_iter_1) {
  const int length = 10;
  thrust::device_vector<float> ts_data(length);
  for (auto i = 0; i < length; ++i) ts_data[i] = float(i + i % 3);

  h2o4gpu::ARIMAModel<float> model(1, 1, 1, length);

  model.Fit(thrust::raw_pointer_cast(ts_data.data()));

  ASSERT_FLOAT_EQ(-1.0369391f, model.Phi()[0]);
  ASSERT_FLOAT_EQ(1.1154615f, model.Theta()[0]);
}