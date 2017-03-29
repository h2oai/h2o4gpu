#include <cublas_v2.h>

#include <algorithm>
#include <limits>

#include "cml/cml_blas.cuh"
#include "cml/cml_linalg.cuh"
#include "cml/cml_matrix.cuh"
#include "matrix/matrix_dense.h"
#include "projector/projector_direct.h"
#include "projector_helper.cuh"
#include "util.h"

#include "pogs.h"

namespace pogs {

namespace {

template<typename T>
struct GpuData {
  T *AA, *L, s;
  cublasHandle_t handle;
  GpuData() : AA(0), L(0), s(static_cast<T>(-1.)) {
    cublasCreate(&handle);
    CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(handle);
    CUDA_CHECK_ERR();
  }
};

}  // namespace

template <typename T, typename M>
ProjectorDirect<T, M>::ProjectorDirect(const M& A)
    : _A(A) {
  // Set GPU specific this->_info.
  GpuData<T> *info = new GpuData<T>();
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T, typename M>
ProjectorDirect<T, M>::~ProjectorDirect() {
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  if (info->AA) {
    cudaFree(info->AA);
    info->AA = 0;
    CUDA_CHECK_ERR();
  }

  if (info->L) {
    cudaFree(info->L);
    info->L = 0;
    CUDA_CHECK_ERR();
  }
  
  delete info;
  this->_info = 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Init() {
  if (this->_done_init)
    return 1;
  this->_done_init = true;
  ASSERT(_A.IsInit());

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  PUSH_RANGE("AA0",1);
  cudaMalloc(&(info->AA), min_dim * min_dim * sizeof(T));
  cudaMalloc(&(info->L), min_dim * min_dim * sizeof(T));
  cudaMemset(info->AA, 0, min_dim * min_dim * sizeof(T));
  cudaMemset(info->L, 0, min_dim * min_dim * sizeof(T));
  CUDA_CHECK_ERR();
  POP_RANGE("AA0",1);

  cublasOperation_t op_type = _A.Rows() > _A.Cols()
      ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Compute AA
  if (_A.Order() == MatrixDense<T>::ROW) {
    PUSH_RANGE("AA1",1);
    const cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    POP_RANGE("AA1",1);
    PUSH_RANGE("AA2",1);
    cml::matrix<T, CblasRowMajor> AA = cml::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    POP_RANGE("AA2",1);
    PUSH_RANGE("AA3",1);
    cml::blas_syrk(info->handle, CUBLAS_FILL_MODE_LOWER, op_type,
        static_cast<T>(1.), &A, static_cast<T>(0.), &AA);
    POP_RANGE("AA3",1);
  } else {
    PUSH_RANGE("AA1",1);
    const cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    POP_RANGE("AA1",1);
    PUSH_RANGE("AA2",1);
    cml::matrix<T, CblasColMajor> AA = cml::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    POP_RANGE("AA2",1);
    PUSH_RANGE("AA3",1);
    cml::blas_syrk(info->handle, CUBLAS_FILL_MODE_LOWER, op_type,
        static_cast<T>(1.), &A, static_cast<T>(0.), &AA);
    POP_RANGE("AA3",1);
  }
  CUDA_CHECK_ERR();

  return 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Project(const T *x0, const T *y0, T s, T *x, T *y,
                                   T tol) {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init || s < static_cast<T>(0.))
    return 1;

  // Get Cublas handle
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  PUSH_RANGE("P0",1);
  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  // Set up views for raw vectors.
  cml::vector<T> y_vec = cml::vector_view_array(y, _A.Rows());
  const cml::vector<T> y0_vec = cml::vector_view_array(y0, _A.Rows());
  cml::vector<T> x_vec = cml::vector_view_array(x, _A.Cols());
  const cml::vector<T> x0_vec = cml::vector_view_array(x0, _A.Cols());

  // Set (x, y) = (x0, y0).
  cml::vector_memcpy(&x_vec, &x0_vec);
  cml::vector_memcpy(&y_vec, &y0_vec);
  CUDA_CHECK_ERR();
  POP_RANGE("P0",1);

  if (_A.Order() == MatrixDense<T>::ROW) {
    PUSH_RANGE("P1",1);
    const cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    POP_RANGE("P1",1);
    PUSH_RANGE("P2",1);
    cml::matrix<T, CblasRowMajor> AA = cml::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    POP_RANGE("P2",1);
    PUSH_RANGE("P3",1);
    cml::matrix<T, CblasRowMajor> L = cml::matrix_view_array<T, CblasRowMajor>
        (info->L, min_dim, min_dim);
    CUDA_CHECK_ERR();
    POP_RANGE("P3",1);

    if (s != info->s) {
      PUSH_RANGE("P4a",1);
      cml::matrix_memcpy(&L, &AA);
      cml::vector<T> diagL = cml::matrix_diagonal(&L);
      cml::vector_add_constant(&diagL, s);
      wrapcudaDeviceSynchronize(); // not needed as next call is cuda call that will occur sequentially on device
      CUDA_CHECK_ERR();
      POP_RANGE("P4a",1);

      PUSH_RANGE("P4b",1);
      cml::linalg_cholesky_decomp(hdl, &L);
      wrapcudaDeviceSynchronize(); // not needed as next call is cuda call that will occur sequentially on device
      CUDA_CHECK_ERR();
      POP_RANGE("P4b",1);
    }
    if (_A.Rows() > _A.Cols()) {
      PUSH_RANGE("P5a",1);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P5a",1);
      PUSH_RANGE("P5b",1);
      cml::linalg_cholesky_svx(hdl, &L, &x_vec);
      POP_RANGE("P5b",1);
      PUSH_RANGE("P5c",1);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
      POP_RANGE("P5c",1);
    } else {
      PUSH_RANGE("P5a",1);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      POP_RANGE("P5a",1);
      PUSH_RANGE("P5b",1);
      cml::linalg_cholesky_svx(hdl, &L, &y_vec);
      POP_RANGE("P5b",1);
      PUSH_RANGE("P5c",1);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P5c",1);
      PUSH_RANGE("P5d",1);
      cml::blas_axpy(hdl, static_cast<T>(1.), &y0_vec, &y_vec);
      POP_RANGE("P5d",1);
    }
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  } else {
    PUSH_RANGE("P6a",1);
    const cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    POP_RANGE("P6a",1);
    PUSH_RANGE("P6b",1);
    cml::matrix<T, CblasColMajor> AA = cml::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    POP_RANGE("P6b",1);
    PUSH_RANGE("P6c",1);
    cml::matrix<T, CblasColMajor> L = cml::matrix_view_array<T, CblasColMajor>
        (info->L, min_dim, min_dim);
    CUDA_CHECK_ERR();
    POP_RANGE("P5c",1);

    if (s != info->s) {
      PUSH_RANGE("P7a",1);
      cml::matrix_memcpy(&L, &AA);
      cml::vector<T> diagL = cml::matrix_diagonal(&L);
      cml::vector_add_constant(&diagL, s);
      wrapcudaDeviceSynchronize();
      CUDA_CHECK_ERR();
      POP_RANGE("P7a",1);
      PUSH_RANGE("P7b",1);
      cml::linalg_cholesky_decomp(hdl, &L);
      wrapcudaDeviceSynchronize();
      CUDA_CHECK_ERR();
      POP_RANGE("P7b",1);
    }
    if (_A.Rows() > _A.Cols()) {
      PUSH_RANGE("P8a",1);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P8a",1);
      PUSH_RANGE("P8b",1);
      cml::linalg_cholesky_svx(hdl, &L, &x_vec);
      POP_RANGE("P8b",1);
      PUSH_RANGE("P8c",1);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
      POP_RANGE("P8c",1);
    } else {
      PUSH_RANGE("P8a",1);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      POP_RANGE("P8a",1);
      PUSH_RANGE("P8b",1);
      cml::linalg_cholesky_svx(hdl, &L, &y_vec);
      POP_RANGE("P8b",1);
      PUSH_RANGE("P8c",1);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P8c",1);
      PUSH_RANGE("P8d",1);
      cml::blas_axpy(hdl, static_cast<T>(1.), &y0_vec, &y_vec);
      POP_RANGE("P8d",1);
    }
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

#ifdef DEBUG
  // Verify that projection was successful.
  CheckProjection(&_A, x0, y0, x, y, s,
      static_cast<T>(1e3) * std::numeric_limits<T>::epsilon());
#endif
  cudaDeviceSynchronize(); // added
  
  info->s = s;
  return 0;
}

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class ProjectorDirect<double, MatrixDense<double> >;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class ProjectorDirect<float, MatrixDense<float> >;
#endif

}  // namespace pogs

