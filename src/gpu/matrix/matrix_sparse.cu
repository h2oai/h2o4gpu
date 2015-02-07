#include <assert.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <memory>

#include "cml/cml_spblas.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_vector.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_sparse.h"
#include "util.cuh"

namespace pogs {

namespace {

template <typename T>
struct GpuData {
  const T *orig_data;
  const POGS_INT *orig_ptr;
  const POGS_INT *orig_ind;
  cublasHandle_t d_hdl;
  cusparseHandle_t s_hdl;
  cusparseMatDescr_t descr;
  GpuData(const T *data, const POGS_INT *ptr, const POGS_INT *ind)
      : orig_data(data), orig_ptr(ptr), orig_ind(ind) {
    cublasCreate(&d_hdl);
    cusparseCreate(&s_hdl);
    cusparseCreateMatDescr(&descr);
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(d_hdl);
    cusparseDestroy(s_hdl);
    cusparseDestroyMatDescr(descr);
    DEBUG_CUDA_CHECK_ERR();
  }
};

cusparseOperation_t OpToCusparseOp(char trans) {
  assert(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N'
      ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
}

}  // namespace

template <typename T>
MatrixSparse<T>::MatrixSparse(char ord, POGS_INT m, POGS_INT n, POGS_INT nnz,
                              const T *data, const POGS_INT *ptr,
                              const POGS_INT *ind)
    : Matrix<T>(m, n), _data(0), _ptr(0), _ind(0), _nnz(nnz) {
  assert(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  // Set GPU specific data.
  GpuData<T> *info = new GpuData<T>(data, ptr, ind);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixSparse<T>::MatrixSparse(const MatrixSparse<T>& A)
    : Matrix<T>(A._m, A._n), _data(0), _ptr(0), _ind(0), _nnz(A._nnz), 
      _ord(A._ord) {

  GpuData<T> *info_A = reinterpret_cast<GpuData<T>*>(A._info);
  GpuData<T> *info = new GpuData<T>(info_A->orig_data, info_A->orig_ptr,
      info_A->orig_ind);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixSparse<T>::~MatrixSparse() {
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;

  if (this->_done_init) {
    if (_data) {
      cudaFree(_data);
      _data = 0;
      DEBUG_CUDA_CHECK_ERR();
    }

    if (_ptr) {
      cudaFree(_ptr);
      _ptr = 0;
      DEBUG_CUDA_CHECK_ERR();
    }

    if (_ind) {
      cudaFree(_ind);
      _ind = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
  }
}

template <typename T>
int MatrixSparse<T>::Init() {
  DEBUG_ASSERT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  // Allocate sparse matrix on gpu.
  cudaMalloc(&_data, 2 * _nnz * sizeof(T));
  cudaMalloc(&_ind, 2 * _nnz * sizeof(POGS_INT));
  cudaMalloc(&_ptr, (this->_m + this->_n + 2) * sizeof(POGS_INT));
  DEBUG_CUDA_CHECK_ERR();

  if (_ord == ROW) {
    cml::spmat<T, POGS_INT, CblasRowMajor> A(_data, _ptr, _ind, this->_m,
        this->_n, _nnz);
    cml::spmat_memcpy(info->s_hdl, &A, _data, _ind, _ptr);
  } else {
    cml::spmat<T, POGS_INT, CblasColMajor> A(_data, _ptr, _ind, this->_m,
        this->_n, _nnz);
    cml::spmat_memcpy(info->s_hdl, &A, _data, _ind, _ptr);
  }
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixSparse<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, this->_n);
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, this->_m);

  if (_ord == ROW) {
    cml::spmat<T, POGS_INT, CblasRowMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    cml::spblas_gemv(info->s_hdl, OpToCusparseOp(trans), info->descr, alpha,
        &A, &x_vec, beta, &y_vec);
  } else {
    cml::spmat<T, POGS_INT, CblasColMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    cml::spblas_gemv(info->s_hdl, OpToCusparseOp(trans), info->descr, alpha,
        &A, &x_vec, beta, &y_vec);
  }
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixSparse<T>::Equil(T *d, T *e) {
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

template class MatrixSparse<double>;
template class MatrixSparse<float>;

}  // namespace pogs

