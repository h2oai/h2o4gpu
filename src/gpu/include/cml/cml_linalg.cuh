#ifndef CML_LINALG_CUH_
#define CML_LINALG_CUH_

#include <cublas_v2.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_utils.cuh"
#include "cml/cml_vector.cuh"

namespace {
__device__ double math_sqrt(double x) {
  return sqrt(x);
}

__device__ float math_sqrt(float x) {
  return sqrtf(x);
}

__device__ double math_rsqrt(double x) {
  return rsqrt(x);
}

__device__ float math_rsqrt(float x) {
  return rsqrtf(x);
}
}  // namespace

namespace cml {

namespace {

template <typename T, CBLAS_ORDER O>
__device__ inline T& Get(T *A, uint i, uint j, uint tda) {
  if (O == CblasColMajor)
    return A[i + j * tda];
  else
    return A[i * tda + j];
}

// Cholesky Helpers.
template <typename T, CBLAS_ORDER O>
__global__ void __block_chol(T *A, uint iter, uint tda) {
  uint col = threadIdx.x;
  uint row = threadIdx.y;
  uint mat_dim = blockDim.x;

  uint global_col = iter * kTileSize + col;
  uint global_row = iter * kTileSize + row;

  const uint kSmTda = kTileSize + 1u;

  __shared__ T L[kSmTda * kTileSize];
  Get<T, O>(L, row, col, kSmTda) = Get<T, O>(A, global_row, global_col, tda);
  __syncthreads();

  for (uint i = 0; i < mat_dim; ++i) {
    T rl11 = math_rsqrt(Get<T, O>(L, i, i, kSmTda));
    __syncthreads();
    if (row >= i && col == 0)
      Get<T, O>(L, row, i, kSmTda) *= rl11;
    __syncthreads();
    if (row >= col && col > i)
      Get<T, O>(L, row, col, kSmTda) -=
          Get<T, O>(L, col, i, kSmTda) * Get<T, O>(L, row, i, kSmTda);
    __syncthreads();
  }

  if (row >= col)
    Get<T, O>(A, global_row, global_col, tda) = Get<T, O>(L, row, col, kSmTda);
}

template <typename T, CBLAS_ORDER O>
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
    Get<T, O>(L, row, i, kSmTda) =
        Get<T, O>(A, global_row, global_col + i, tda);

  global_row = row + (iter + tile_idx + 1u) * kTileSize;

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      Get<T, O>(A12, row, i, kSmTda) = 
          Get<T, O>(A, global_row, global_col + i, tda);
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i) {
      for (uint j = 0; j < i; ++j)
        Get<T, O>(A12, row, i, kSmTda) -=
            Get<T, O>(A12, row, j, kSmTda) * Get<T, O>(L, i, j, kSmTda);
      Get<T, O>(A12, row, i, kSmTda) /= Get<T, O>(L, i, i, kSmTda);
    }
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      Get<T, O>(A, global_row, global_col + i, tda) =
          Get<T, O>(A12, row, i, kSmTda);
  }
}

}  // namespace

/// Cholesky decomposition
template <typename T, CBLAS_ORDER O>
cublasStatus_t linalg_cholesky_decomp(cublasHandle_t handle,
                                      matrix<T, O> *A) {
  cudaStream_t stm;
  cublasStatus_t err = cublasGetStream(handle, &stm);

  uint num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

  for (uint i = 0; i < num_tiles; ++i) {
    if (err != CUBLAS_STATUS_SUCCESS)
      break;

    uint block_dim_1d = std::min<uint>(kTileSize, A->size1 - i * kTileSize);
    dim3 block_dim(block_dim_1d, block_dim_1d);
    __block_chol<T, O><<<1, block_dim, 0, stm>>>(A->data, i, A->tda);

    if (i < num_tiles - 1u) {
      uint grid_dim = num_tiles - i - 1u;
      __block_trsv<T, O><<<grid_dim, kTileSize, 0, stm>>>(A->data, i, A->size1,
                                                    A->tda);

      matrix<T, O> L12 = matrix_submatrix(A, (i + 1) * kTileSize, i * kTileSize,
          A->size1 - (i + 1) * kTileSize, kTileSize);
      matrix<T, O> L22 = matrix_submatrix(A, (i + 1) * kTileSize,
          (i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
          A->size1 - (i + 1) * kTileSize);
      err = blas_syrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
         static_cast<T>(-1), &L12, static_cast<T>(1), &L22);
    }
  }
  CublasCheckError(err);
  return err;
}

/// Solves L*x = b and L'*x = b for x
template <typename T, CBLAS_ORDER O>
cublasStatus_t linalg_cholesky_svx(cublasHandle_t handle,
                                   const matrix<T, O> *L, vector<T> *x) {
  cublasStatus_t err;
  // Solves Ly=b where b is assigned as x, and output y is x.
  err = blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, //The non-transpose operation
      CUBLAS_DIAG_NON_UNIT, L, x);
  CublasCheckError(err);

  // Solve L^Tx = y (inputted as x) for x and store solution in x-named C variable
  err = blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, //The transpose operation
      CUBLAS_DIAG_NON_UNIT, L, x);
  CublasCheckError(err);

  return err;
}

}  // namespace cml

#endif  // CML_LINALG_CUH_

