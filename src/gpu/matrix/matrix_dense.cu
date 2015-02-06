#include <assert.h>
#include <cublas_v2.h>

#include "cml/cml_matrix.cuh"
#include "cml/cml_blas.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "util.cuh"


namespace {

template<typename T>
struct GpuData {
  const T *orig_data;
  cublasHandle_t handle;
  GpuData(const T *orig_data) : orig_data(orig_data) {
    cublasCreate(&handle);
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(handle);
    DEBUG_CUDA_CHECK_ERR();
  }
};

cublasOperation_t OpToCublasOp(char trans) {
  assert(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
}

}  // namespace

namespace pogs {

template <typename T>
MatrixDense<T>::MatrixDense(char ord, size_t m, size_t n, const T *data)
    : Matrix<T>(m, n) {
  assert(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  // Set GPU specific _info.
  GpuData<T> *info = new GpuData<T>(data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDense<T>::~MatrixDense() {
  // TODO: Check memory implications of this
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  delete info;
}

template <typename T>
int MatrixDense<T>::Init() {
  DEBUG_ASSERT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  // Copy Matrix to GPU.
  cudaMalloc(&_data, this->_m * this->_n * sizeof(T));
  cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),
      cudaMemcpyHostToDevice);
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixDense<T>::Free() {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  if (this->_data) {
    cudaFree(this->_data);
    this->_data = 0;
    DEBUG_CUDA_CHECK_ERR();
  }

  return 0;
}

template <typename T>
int MatrixDense<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, this->_n);
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, this->_m);

  if (_ord == ROW) {
    cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>(_data, this->_m, this->_n);
    cml::blas_gemv(info->handle, OpToCublasOp(trans), alpha, &A, &x_vec, beta,
        &y_vec);
  } else {
    cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>(_data, this->_m, this->_n);
    cml::blas_gemv(info->handle, OpToCublasOp(trans), alpha, &A, &x_vec, beta,
        &y_vec);
  }
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixDense<T>::Equil(T *d, T *e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  cml::vector<T> d_vec = cml::vector_view_array<T>(d, this->_m);
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, this->_n);

  // TODO: implement proper equilibration
  cml::vector_set_all(&d_vec, static_cast<T>(1));
  cml::vector_set_all(&e_vec, static_cast<T>(1));

  return 0;
}

template class MatrixDense<double>;
template class MatrixDense<float>;

}  // _namespace pogs

