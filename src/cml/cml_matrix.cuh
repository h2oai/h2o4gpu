#ifndef CML_MATRIX_CUH_
#define CML_MATRIX_CUH_

#include <algorithm>
#include <cstdio>

#include "cblas_def.h"
#include "cml_defs.cuh"
#include "cml_utils.cuh"
#include "cml_vector.cuh"

// Cuda Matrix Library
namespace cml {

// Matrix Class
template <typename T, CBLAS_ORDER O>
struct matrix {
  size_t size1, size2, tda;
  T* data;
};

// Helper Methods
namespace {

template <typename T, CBLAS_ORDER O>
__global__ void __set_matrix(T *data, T val, size_t tda, size_t size1,
                             size_t size2) {
  uint tid_row = blockIdx.x * blockDim.x + threadIdx.x;
  uint tid_col = blockIdx.y * blockDim.y + threadIdx.y;
  if (O == CblasRowMajor)
    for (uint i = tid_row; i < size1; i += gridDim.x * blockDim.x)
      for (uint j = tid_col; j < size2; j += gridDim.y * blockDim.y)
        data[i * tda  + j] = val;
  else
    for (uint j = tid_col; j < size2; j += gridDim.y * blockDim.y)
      for (uint i = tid_row; i < size1; i += gridDim.x * blockDim.x)
        data[i + j * tda] = val;
}

template <typename T, CBLAS_ORDER O>
void _set_matrix(matrix<T, O> *A, T val) {
  uint grid_dimx = calc_grid_dim(A->size1, kBlockSize);
  uint grid_dimy = calc_grid_dim(A->size2, kBlockSize);
  dim3 grid_dim(grid_dimx, grid_dimy, 1u);
  dim3 block_dim(kBlockSize, kBlockSize, 1u);
  __set_matrix<T, O><<<grid_dim, block_dim>>>(A->data, val, A->tda, A->size1,
                                              A->size2);
}

template <typename T, CBLAS_ORDER O>
__global__ void __matrix_add_constant_diag(T *data, T val, size_t tda) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i * tda + i] += val;
}

}  // namespace

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_alloc(size_t m, size_t n) {
  matrix<T, O> mat;
  mat.size1 = m;
  mat.size2 = n;
  if (O == CblasRowMajor)
    mat.tda = n;
  else
    mat.tda = m;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&mat.data),
      m * n * sizeof(T));
  CudaCheckError(err);
  if (err != cudaSuccess)
    mat.data = 0;
  return mat;
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_calloc(size_t m, size_t n) {
  matrix<T, O> mat = matrix_alloc<T, O>(m, n);
  if (mat.data != 0)
    _set_matrix(&mat, static_cast<T>(0));
  return mat;
}

template <typename T, CBLAS_ORDER O>
void matrix_free(matrix<T, O> *A) {
  cudaError_t err = cudaFree(A->data);
  CudaCheckError(err);
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_submatrix(matrix<T, O> *A, size_t i, size_t j, size_t n1,
                              size_t n2) {
  matrix<T, O> submat;
  submat.size1 = n1;
  submat.size2 = n2;
  if (O == CblasRowMajor)
    submat.data = A->data + i * A->tda + j;
  else
    submat.data = A->data + i + j * A->tda;
  submat.tda = A->tda;
  return submat;
}

template <typename T, CBLAS_ORDER O>
vector<T> matrix_row(matrix<T, O> *A, size_t i) {
  vector<T> v;
  v.size = A->size2;
  if (O == CblasRowMajor) {
    v.data = A->data + i * A->tda;
    v.stride = static_cast<size_t>(1);
  } else {
    v.data = A->data + i;
    v.stride = A->tda;
  }
  return v;
}

template <typename T, CBLAS_ORDER O>
vector<T> matrix_column(matrix<T, O> *A, size_t j) {
  vector<T> v;
  v.size = A->size1;
  if (O == CblasRowMajor) {
    v.data = A->data + j;
    v.stride = A->tda;
  } else {
    v.data = A->data + j * A->tda;
    v.stride = static_cast<size_t>(1);
  }
  return v;
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_view_array(T *base, size_t n1, size_t n2) {
  matrix<T, O> mat;
  mat.size1 = n1;
  mat.size2 = n2;
  if (O == CblasRowMajor)
    mat.tda = n2;
  else
    mat.tda = n1;
  mat.data = base;
  return mat;
}

template <typename T, CBLAS_ORDER O>
void matrix_memcpy(matrix<T, O> *A, const matrix<T, O> *B) {
  cudaError_t err;
  if ((O == CblasRowMajor && A->tda == A->size2 && B->tda == B->size2) ||
      (O == CblasColMajor && A->tda == A->size1 && B->tda == B->size1))
    err = cudaMemcpy(A->data, B->data, A->size1 * A->size2 * sizeof(T),
        cudaMemcpyDefault);
  else if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      err = cudaMemcpy(A->data + i * A->tda, B->data + i * B->tda,
          A->size2 * sizeof(T), cudaMemcpyDefault);
  else 
    for (unsigned int j = 0; j < A->size2; ++j) 
      err = cudaMemcpy(A->data + j * A->tda, B->data + j * B->tda,
          A->size1 * sizeof(T), cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T, CBLAS_ORDER O>
void matrix_memcpy(matrix<T, O> *A, const T *B) {
  cudaError_t err;
  if ((O == CblasRowMajor && A->tda == A->size2) ||
      (O == CblasColMajor && A->tda == A->size1))
    err = cudaMemcpy(A->data, B, A->size1 * A->size2 * sizeof(T),
        cudaMemcpyDefault);
  else if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      err = cudaMemcpy(A->data + i * A->tda, B + i * A->size2,
          A->size2 * sizeof(T), cudaMemcpyDefault);
  else 
    for (unsigned int j = 0; j < A->size2; ++j) 
      err = cudaMemcpy(A->data + j * A->tda, B + j * A->size1,
          A->size1 * sizeof(T), cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T, CBLAS_ORDER O>
void matrix_memcpy(T *A, const matrix<T, O> *B) {
  cudaError_t err;
  if ((O == CblasRowMajor && B->tda == B->size2) ||
      (O == CblasColMajor && B->tda == B->size1)) {
    err = cudaMemcpy(A, B->data, B->size1 * B->size2 * sizeof(T),
        cudaMemcpyDefault);
  } else if (O == CblasRowMajor) {
    for (unsigned int i = 0; i < B->size1; ++i)
      err = cudaMemcpy(A + i * B->size2, B->data + i * B->tda,
          B->size2 * sizeof(T), cudaMemcpyDefault);
  } else {
    for (unsigned int j = 0; j < B->size2; ++j) 
      err = cudaMemcpy(A + j * B->size1, B->data + j * B->tda,
          B->size1 * sizeof(T), cudaMemcpyDefault);
  }
  CudaCheckError(err);
}

template <typename T, CBLAS_ORDER O>
void matrix_print(const matrix<T, O> *A) {
  T* A_;
  if (O == CblasRowMajor)
    A_ = new T[A->tda * A->size2];
  else
    A_ = new T[A->tda * A->size1];
  matrix_memcpy(A_, A);
  for (unsigned int i = 0; i < A->size1; ++i) {
    for (unsigned int j = 0; j < A->size2; ++j)
      if (O == CblasRowMajor)
        printf("%e ", A_[i * A->tda + j]);
      else
        printf("%e ", A_[i + j * A->tda]);
    printf("\n");
  }
  printf("\n");
  delete [] A_;
}

template <typename T, CBLAS_ORDER O>
vector<T> matrix_diagonal(matrix<T, O> *A) {
  vector<T> v;
  v.data = A->data;
  v.stride = A->tda + 1;
  v.size = std::min(A->size1, A->size2);
  return v;
}

template <typename T, CBLAS_ORDER O>
void matrix_scale(matrix<T, O> *A, T x) {
  if ((O == CblasRowMajor && A->tda == A->size2) ||
      (O == CblasColMajor && A->tda == A->size1))
    thrust::transform(thrust::device_pointer_cast(A->data),
        thrust::device_pointer_cast(A->data + A->size2 * A->size1),
        thrust::constant_iterator<T>(x), thrust::device_pointer_cast(A->data),
        thrust::multiplies<T>());
  else if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      thrust::transform(thrust::device_pointer_cast(A->data + i * A->size2),
          thrust::device_pointer_cast(A->data + (i + 1) * A->size2),
          thrust::constant_iterator<T>(x),
          thrust::device_pointer_cast(A->data + i * A->size2),
          thrust::multiplies<T>());
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      thrust::transform(thrust::device_pointer_cast(A->data + j * A->size1),
          thrust::device_pointer_cast(A->data + (j + 1) * A->size1),
          thrust::constant_iterator<T>(x),
          thrust::device_pointer_cast(A->data + j * A->size1),
          thrust::multiplies<T>());
}

}  // namespace cml

#endif  // CML_MATRIX_CUH_

