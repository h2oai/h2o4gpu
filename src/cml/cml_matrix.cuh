#ifndef CML_MATRIX_CUH_
#define CML_MATRIX_CUH_

#include <algorithm>
#include <cstdio>

#include "cml_defs.cuh"
#include "cml_utils.cuh"
#include "cml_vector.cuh"

// Cuda Matrix Library
namespace cml {

// Matrix Class
template <typename T>
struct matrix {
  size_t size1, size2, tda;
  T* data;
};

// Helper Methods
namespace {

template <typename T>
__global__ void __set_matrix(T *data, T val, size_t tda, size_t size1,
                             size_t size2) {
  uint tid_row = blockIdx.x * blockDim.x + threadIdx.x;
  uint tid_col = blockIdx.y * blockDim.y + threadIdx.y;
  for (uint j = tid_col; j < size2; j += gridDim.y * blockDim.y)
    for (uint i = tid_row; i < size1; i += gridDim.x * blockDim.x)
      data[i  + j * tda] = val;
}

template <typename T>
void _set_matrix(matrix<T> *A, T val) {
  uint grid_dimx = calc_grid_dim(A->size1, kBlockSize);
  uint grid_dimy = calc_grid_dim(A->size2, kBlockSize);
  dim3 grid_dim(grid_dimx, grid_dimy, 1u);
  dim3 block_dim(kBlockSize, kBlockSize, 1u);
  __set_matrix<<<grid_dim, block_dim>>>(A->data, val, A->tda, A->size1,
                                        A->size2);
}

template <typename T>
__global__ void __matrix_add_constant_diag(T *data, T val, size_t tda) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i * tda + i] += val;
}

}  // namespace

template <typename T>
matrix<T> matrix_alloc(size_t m, size_t n) {
  matrix<T> mat;
  mat.size1 = m;
  mat.size2 = n;
  mat.tda = m;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&mat.data),
      m * n * sizeof(T));
  CudaCheckError(err);
  return mat;
}

template <typename T>
matrix<T> matrix_calloc(size_t m, size_t n) {
  matrix<T> mat = matrix_alloc<T>(m, n);
  _set_matrix(&mat, static_cast<T>(0));
  return mat;
}

template<typename T>
void matrix_free(matrix<T> *A) {
  cudaError_t err = cudaFree(A->data);
  CudaCheckError(err);
}

template <typename T>
matrix<T> matrix_submatrix(matrix<T> *A, size_t i, size_t j, size_t n1,
                           size_t n2) {
  matrix<T> submat;
  submat.size1 = n1;
  submat.size2 = n2;
  submat.data = A->data + j * A->tda + i;
  submat.tda = A->tda;
  return submat;
}

template <typename T>
vector<T> matrix_row(matrix<T> *A, size_t i) {
  vector<T> v;
  v.size = A->size2;
  v.data = A->data + i;
  v.stride = A->tda;
  return v;
}

template <typename T>
vector<T> matrix_column(matrix<T> *A, size_t j) {
  vector<T> v;
  v.size = A->size1;
  v.data = A->data + A->tda * j;
  v.stride = static_cast<size_t>(1);
  return v;
}

// TODO: Take tda into account properly
template <typename T>
void matrix_memcpy(matrix<T> *A, const matrix<T> *B) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(A->data),
      reinterpret_cast<const void*>(B->data), A->tda * A->size2 * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void matrix_memcpy(matrix<T> *A, const T *B) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(A->data),
      reinterpret_cast<const void*>(B), A->tda * A->size2 * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void matrix_memcpy(T *A, const matrix<T> *B) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(A),
      reinterpret_cast<const void*>(B->data), B->tda * B->size2 * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void matrix_print(const matrix<T> &A) {
  T* A_ = new T[A.tda * A.size2];
  matrix_memcpy(A_, &A);
  for (unsigned int i = 0; i < A.size1; ++i) {
    for (unsigned int j = 0; j < A.size2; ++j)
      printf("%e ", A_[i + j * A.tda]);
    printf("\n");
  }
  printf("\n");
  delete [] A_;
}

template <typename T>
vector<T> matrix_diagonal(matrix<T> *A) {
  vector<T> v;
  v.data = A->data;
  v.stride = A->tda + 1;
  v.size = std::min(A->size1, A->size2);
  return v;
}

template <typename T>
void matrix_scale(matrix<T> *A, T x) {
  thrust::transform(thrust::device_pointer_cast(A->data),
      thrust::device_pointer_cast(A->data + A->size2 * A->tda),
      thrust::constant_iterator<T>(x), thrust::device_pointer_cast(A->data),
      thrust::multiplies<T>());
}

}  // namespace cml

#endif  // CML_MATRIX_CUH_

