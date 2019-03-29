#include "gtest/gtest.h"

#include "../gpu/factorization/als.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// TODO: hermitianT10 had a race condition bug that was not detected by such
// simple tests It was reproducable when F >=70 on test data. Probably when
// a sparse matrix is inbalanced.
// TODO: Use BLAS instead of naive matrix multiplication

void Test_hermitianT10_eye(int f, int m, float lambda) {
  int batch_size = m;
  int block_dim = f / ALS_T10 * (f / ALS_T10 + 1) / 2;
  if (block_dim < f / 2)
    block_dim = f / 2;

  thrust::device_vector<float> tt_device(f * f * batch_size, 0.0);

  // sparse eye matrix
  int csrRowIndex_host[m + 1];
  int csrColIndex_host[m];

  for (int i = 0; i < m + 1; ++i)
    csrRowIndex_host[i] = i;

  for (int i = 0; i < m; ++i)
    csrColIndex_host[i] = i;

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

  for (int v = 0; v < m; ++v) {
    thrust::host_vector<float> thetaT_theta =
        thrust::host_vector<float>(f * f, 0.0);
    for (int row = 0; row < f; ++row) {
      for (int col = 0; col < f; ++col) {
        thetaT_theta[row * f + col] =
            thetaT_device[row + v * f] * thetaT_device[col + v * f];
        if (row == col)
          thetaT_theta[row * f + col] += lambda;
        ASSERT_FLOAT_EQ(tt_device[v * f * f + row * f + col],
                        thetaT_theta[row * f + col])
            << v << ", " << row << ", " << col;
      }
    }
  }
}

void Test_hermitianT10_2columns(int f, int m, float lambda) {
  // most simple test case, all sizes 20
  int offset = 10;
  int batch_size = m;
  int block_dim = f / ALS_T10 * (f / ALS_T10 + 1) / 2;
  if (block_dim < f / 2)
    block_dim = f / 2;

  thrust::device_vector<float> tt_device(f * f * batch_size, 0.0);

  int csrRowIndex_host[m + 1];

  // v , (v + 1) % m
  int csrColIndex_host[2 * m];

  for (int i = 0; i < m + 1; ++i)
    csrRowIndex_host[i] = 2 * i;
  for (int i = 0; i < 2 * m; ++i)
    if (i % 2 == 0)
      csrColIndex_host[i] = i / 2;
    else
      csrColIndex_host[i] = ((i / 2) + offset) % m;
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

  for (int v = 0; v < m; ++v) {
    thrust::host_vector<float> thetaT_theta =
        thrust::host_vector<float>(f * f, 0.0);
    for (int row = 0; row < f; ++row) {
      for (int col = 0; col < f; ++col) {
        thetaT_theta[row * f + col] +=
            thetaT_device[row + v * f] * thetaT_device[col + v * f];
        int v1 = (v + offset) % m;
        thetaT_theta[row * f + col] +=
            thetaT_device[row + v1 * f] * thetaT_device[col + v1 * f];
        if (row == col)
          thetaT_theta[row * f + col] += 2 * lambda;

        ASSERT_FLOAT_EQ(tt_device[v * f * f + row * f + col],
                        thetaT_theta[row * f + col])
            << v << ", " << row << ", " << col;
      }
    }
  }
}

// Those tests are really slow
// TEST(Factorization, get_hermitianT10_eye_70) {
//   Test_hermitianT10_eye(70, 150, 1.0);
// }

// TEST(Factorization, get_hermitianT10_eye_60) {
//   Test_hermitianT10_eye(60, 150, 1.0);
// }

// TEST(Factorization, get_hermitianT10_eye_50) {
//   Test_hermitianT10_eye(50, 150, 1.0);
// }

// TEST(Factorization, get_hermitianT10_eye_40) {
//   Test_hermitianT10_eye(40, 150, 1.0);
// }

// TEST(Factorization, get_hermitianT10_eye_30) {
//   Test_hermitianT10_eye(30, 100, 1.0);
// }

TEST(Factorization, get_hermitianT10_eye_20) {
  Test_hermitianT10_eye(20, 100, 1.0);
}

TEST(Factorization, get_hermitianT10_2columns_20) {
  Test_hermitianT10_2columns(20, 200, 1.0);
}
// TEST(Factorization, get_hermitianT10_2columns_30) {
//   Test_hermitianT10_2columns(30, 200, 1.0);
// }
// TEST(Factorization, get_hermitianT10_2columns_40) {
//   Test_hermitianT10_2columns(40, 200, 1.0);
// }
// Those tests are really slow
// TEST(Factorization, get_hermitianT10_2columns_50) {
//   Test_hermitianT10_2columns(50, 200, 1.0);
// }
// TEST(Factorization, get_hermitianT10_2columns_60) {
//   Test_hermitianT10_2columns(60, 200, 1.0);
// }
// TEST(Factorization, get_hermitianT10_2columns_70) {
//   Test_hermitianT10_2columns(70, 200, 1.0);
// }
