#include "gtest/gtest.h"

#include "../gpu/factorization/als.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

TEST(Factorization, get_hermitianT10_eye) {
  // most simple test case, all sizes 20
  constexpr float lambda = 0.0;
  constexpr int f = 20;
  constexpr int m = 20;
  constexpr int n = 20;
  constexpr int batch_size = m;
  int block_dim = f / ALS_T10 * (f / ALS_T10 + 1) / 2;
  if (block_dim < f / 2)
    block_dim = f / 2;

  thrust::device_vector<float> tt_device(f * f * batch_size, 0.0);

  // sparse eye matrix
  int csrRowIndex_host[m + 1] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

  int csrColIndex_host[20] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  thrust::device_vector<int> csrRowIndex_device(csrRowIndex_host,
                                                csrRowIndex_host + m + 1);
  thrust::device_vector<int> csrColIndex_device(
      csrColIndex_host, csrColIndex_host + sizeof(csrColIndex_host) /
                                               sizeof(csrColIndex_host[0]));

  thrust::device_vector<float> thetaT_device(f * m);
  for (int i = 0; i < f * m; ++i) {
    thetaT_device[i] = float(i * 0.1);
  }

  get_hermitianT10<<<batch_size, block_dim,
                     SCAN_BATCH * f / 2 * sizeof(float2)>>>(
      0, thrust::raw_pointer_cast(tt_device.data()),
      thrust::raw_pointer_cast(csrRowIndex_device.data()),
      thrust::raw_pointer_cast(csrColIndex_device.data()), lambda, m, f,
      thrust::raw_pointer_cast(thetaT_device.data()));

  OK(cudaDeviceSynchronize());

  for (int v = 0; v < n; ++v) {
    thrust::host_vector<float> thetaT_theta =
        thrust::host_vector<float>(f * f, 0.0);
    for (int row = 0; row < f; ++row) {
      for (int col = 0; col < f; ++col) {
        thetaT_theta[row * f + col] =
            thetaT_device[row + v * f] * thetaT_device[col + v * f];
        ASSERT_FLOAT_EQ(tt_device[v * f * f + row * f + col],
                        thetaT_theta[row * f + col])
            << v << ", " << row << ", " << col;
      }
    }
  }
}

TEST(Factorization, get_hermitianT10_2columns) {
  // most simple test case, all sizes 20
  constexpr float lambda = 0.0;
  constexpr int f = 20;
  constexpr int m = 20;
  constexpr int n = 20;
  constexpr int batch_size = m;
  int block_dim = f / ALS_T10 * (f / ALS_T10 + 1) / 2;
  if (block_dim < f / 2)
    block_dim = f / 2;

  thrust::device_vector<float> tt_device(f * f * batch_size, 0.0);

  int csrRowIndex_host[m + 1] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
                                 22, 24, 26, 28, 30, 32, 34, 36, 38, 40};

  // v , (v + 1) % m
  int csrColIndex_host[40] = {0,  1,  1,  2,  2,  3,  3,  4,  4,  5,
                              5,  6,  6,  7,  7,  8,  8,  9,  9,  10,
                              10, 11, 11, 12, 12, 13, 13, 14, 14, 15,
                              15, 16, 16, 17, 17, 18, 18, 19, 0,  19};

  thrust::device_vector<int> csrRowIndex_device(csrRowIndex_host,
                                                csrRowIndex_host + m + 1);
  thrust::device_vector<int> csrColIndex_device(
      csrColIndex_host, csrColIndex_host + sizeof(csrColIndex_host) /
                                               sizeof(csrColIndex_host[0]));

  thrust::device_vector<float> thetaT_device(f * m);
  for (int i = 0; i < f * m; ++i) {
    thetaT_device[i] = float(i * 0.1);
  }

  get_hermitianT10<<<batch_size, block_dim,
                     SCAN_BATCH * f / 2 * sizeof(float2)>>>(
      0, thrust::raw_pointer_cast(tt_device.data()),
      thrust::raw_pointer_cast(csrRowIndex_device.data()),
      thrust::raw_pointer_cast(csrColIndex_device.data()), lambda, m, f,
      thrust::raw_pointer_cast(thetaT_device.data()));

  OK(cudaDeviceSynchronize());

  for (int v = 0; v < n; ++v) {
    thrust::host_vector<float> thetaT_theta =
        thrust::host_vector<float>(f * f, 0.0);
    for (int row = 0; row < f; ++row) {
      for (int col = 0; col < f; ++col) {
        thetaT_theta[row * f + col] +=
            thetaT_device[row + v * f] * thetaT_device[col + v * f];
        int v1 = (v + 1) % n;
        thetaT_theta[row * f + col] +=
            thetaT_device[row + v1 * f] * thetaT_device[col + v1 * f];

        ASSERT_FLOAT_EQ(tt_device[v * f * f + row * f + col],
                        thetaT_theta[row * f + col])
            << v << ", " << row << ", " << col;
      }
    }
  }
}

TEST(Factorization, get_hermitianT10_eye_2_batches) {
  constexpr int X_BATCH = 2;
  // most simple test case, all sizes 20
  constexpr float lambda = 0.0;
  constexpr int f = 20;
  constexpr int m = 20;
  constexpr int n = 20;
  int block_dim = f / ALS_T10 * (f / ALS_T10 + 1) / 2;
  if (block_dim < f / 2)
    block_dim = f / 2;

  // sparse eye matrix
  int csrRowIndex_host[m + 1] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

  int csrColIndex_host[20] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  thrust::device_vector<int> csrRowIndex_device(csrRowIndex_host,
                                                csrRowIndex_host + m + 1);
  thrust::device_vector<int> csrColIndex_device(
      csrColIndex_host, csrColIndex_host + sizeof(csrColIndex_host) /
                                               sizeof(csrColIndex_host[0]));

  thrust::device_vector<float> thetaT_device(f * m);
  for (int i = 0; i < f * m; ++i) {
    thetaT_device[i] = float(i * 0.1);
  }

  for (int batch_id = 0; batch_id < X_BATCH; batch_id++) {
    int batch_size = 0;
    if (batch_id != X_BATCH - 1)
      batch_size = m / X_BATCH;
    else
      batch_size = m - batch_id * (m / X_BATCH);

    int batch_offset = batch_id * (m / X_BATCH);

    thrust::device_vector<float> tt_device(f * f * batch_size, 0.0);

    get_hermitianT10<<<batch_size, block_dim,
                       SCAN_BATCH * f / 2 * sizeof(float2)>>>(
        batch_offset, thrust::raw_pointer_cast(tt_device.data()),
        thrust::raw_pointer_cast(csrRowIndex_device.data()),
        thrust::raw_pointer_cast(csrColIndex_device.data()), lambda, m, f,
        thrust::raw_pointer_cast(thetaT_device.data()));

    OK(cudaDeviceSynchronize());

    for (int v = 0; v < batch_size; ++v) {
      thrust::host_vector<float> thetaT_theta =
          thrust::host_vector<float>(f * f, 0.0);
      for (int row = 0; row < f; ++row) {
        for (int col = 0; col < f; ++col) {
          thetaT_theta[row * f + col] =
              thetaT_device[row + (v + batch_offset) * f] *
              thetaT_device[col + (v + batch_offset) * f];

          ASSERT_FLOAT_EQ(tt_device[v * f * f + row * f + col],
                          thetaT_theta[row * f + col])
              << v << ", " << row << ", " << col;
        }
      }
    }
  }
}