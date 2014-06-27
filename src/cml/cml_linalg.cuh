#ifndef CML_LINALG_CUH_
#define CML_LINALG_CUH_

#include <cublas_v2.h>

#include "cml_blas.cuh"
#include "cml_math.cuh"
#include "cml_matrix.cuh"
#include "cml_utils.cuh"
#include "cml_vector.cuh"

namespace cml {

namespace {

// Cholesky Helpers.
template <typename T>
__global__ void __block_chol(T *A, uint iter, uint tda) {
  uint col = threadIdx.x;
  uint row = threadIdx.y;
  uint mat_dim = blockDim.x;

  uint global_col = iter * kTileSize + col;
  uint global_row = iter * kTileSize + row;

  const uint kSmTda = kTileSize + 1u;

  __shared__ T L[kSmTda * kTileSize];
  L[row + col * kSmTda] = A[global_row + global_col * tda];
  __syncthreads();

  for (uint i = 0; i < mat_dim; ++i) {
    T rl11 = math_rsqrt(L[i + i * kSmTda]);
    __syncthreads();
    if (row >= i && col == 0)
      L[row + i * kSmTda] *= rl11;
    __syncthreads();
    if (row >= col && col > i)
      L[row + col * kSmTda] -= L[col + i * kSmTda] * L[row + i * kSmTda];
    __syncthreads();
  }

  if (row >= col)
    A[global_row + global_col * tda] = L[row + col * kSmTda];
}

template <typename T>
__global__ void __block_trsv(T *A, uint iter, uint n, uint tda) {
  uint tile_idx = blockIdx.x;
  uint row = threadIdx.x;

  const uint kSmTda = kTileSize + 1u;
  __shared__ T L[kSmTda * kTileSize];
  __shared__ T A12[kSmTda * kTileSize];

  uint global_col = iter * kTileSize;
  uint global_row = iter * kTileSize + row;

  // Load A -> L column-wise.
  for (uint i = 0; i < kTileSize; ++i)
    L[row + i * kSmTda] = A[global_row + (global_col + i) * tda];

  global_row = row + (iter + tile_idx + 1u) * kTileSize;

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      A12[row + i * kSmTda] = A[global_row + (global_col + i) * tda];
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i) {
      for (uint j = 0; j < i; ++j)
        A12[row + i * kSmTda] -= A12[row + j * kSmTda] * L[i + j * kSmTda];
      A12[row + i * kSmTda] /= L[i + i * kSmTda];
    }
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      A[global_row + (global_col + i) * tda] = A12[row + i * kSmTda];
  }
}

}  // namespace

// Cholesky.
template <typename T>
cublasStatus_t linalg_cholesky_decomp(cublasHandle_t handle,
                                      matrix<T> *A) {
  cudaStream_t stm;
  cublasStatus_t err = cublasGetStream(handle, &stm);

  uint num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

  for (uint i = 0; i < num_tiles; ++i) {
    if (err != CUBLAS_STATUS_SUCCESS)
      break;

    uint block_dim_1d = std::min<uint>(kTileSize, A->size1 - i * kTileSize);
    dim3 block_dim(block_dim_1d, block_dim_1d);
    __block_chol<<<1, block_dim, 0, stm>>>(A->data, i, A->tda);

    if (i < num_tiles - 1u) {
      uint grid_dim = num_tiles - i - 1u;
      __block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i, A->size1,
                                                    A->tda);

      matrix<T> L12 = matrix_submatrix(A, (i + 1) * kTileSize, i * kTileSize,
          A->size1 - (i + 1) * kTileSize, kTileSize);
      matrix<T> L22 = matrix_submatrix(A, (i + 1) * kTileSize,
          (i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
          A->size1 - (i + 1) * kTileSize);
      err = blas_syrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
         static_cast<T>(-1), &L12, static_cast<T>(1), &L22);
    }
  }
  CublasCheckError(err);
  return err;
}

template <typename T>
cublasStatus_t linalg_cholesky_svx(cublasHandle_t handle,
                                   const matrix<T> *L, vector<T> *x) {
  cublasStatus_t err = blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, L, x);
  CublasCheckError(err);

  err = blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      CUBLAS_DIAG_NON_UNIT, L, x);
  CublasCheckError(err);

  return err;
}

}  // namespace cml

#endif  // CML_LINALG_CUH_

