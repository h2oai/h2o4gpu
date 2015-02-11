#include <cublas_v2.h>

#include <algorithm>
#include <limits>

#include "cml/cml_blas.cuh"
#include "cml/cml_linalg.cuh"
#include "cml/cml_matrix.cuh"
#include "matrix/matrix_dense.h"
#include "projector/projector_direct.h"
#include "projector_helper.cuh"
#include "util.cuh"

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
  _info = 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Init() {
  if (this->_done_init)
    return 1;
  this->_done_init = true;
  ASSERT(_A.IsInit());

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  cudaMalloc(&(info->AA), min_dim * min_dim * sizeof(T));
  cudaMalloc(&(info->L), min_dim * min_dim * sizeof(T));
  cudaMemset(info->AA, 0, min_dim * min_dim * sizeof(T));
  cudaMemset(info->L, 0, min_dim * min_dim * sizeof(T));
  CUDA_CHECK_ERR();

  cublasOperation_t op_type = _A.Rows() >= _A.Cols()
      ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Compute AA
  if (_A.Order() == MatrixDense<T>::ROW) {
    const cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasRowMajor> AA = cml::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    cml::blas_syrk(info->handle, CUBLAS_FILL_MODE_LOWER, op_type,
        static_cast<T>(1.), &A, static_cast<T>(0.), &AA);
  } else {
    const cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasColMajor> AA = cml::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    cml::blas_syrk(info->handle, CUBLAS_FILL_MODE_LOWER, op_type,
        static_cast<T>(1.), &A, static_cast<T>(0.), &AA);
  }
  CUDA_CHECK_ERR();

  return 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Project(const T *x0, const T *y0, T s, T *x, T *y) {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init || s < static_cast<T>(0.))
    return 1;

  // Get Cublas handle
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

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

  if (_A.Order() == MatrixDense<T>::ROW) {
    const cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasRowMajor> AA = cml::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    cml::matrix<T, CblasRowMajor> L = cml::matrix_view_array<T, CblasRowMajor>
        (info->L, min_dim, min_dim);
    CUDA_CHECK_ERR();

    if (s != info->s) {
      cml::matrix_memcpy(&L, &AA);
      cml::vector<T> diagL = cml::matrix_diagonal(&L);
      cml::vector_add_constant(&diagL, s);
      cml::linalg_cholesky_decomp(hdl, &L);
    }
    if (_A.Rows() >= _A.Cols()) {
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      cml::linalg_cholesky_svx(hdl, &L, &x_vec);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
    } else {
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      cml::linalg_cholesky_svx(hdl, &L, &y_vec);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      cml::blas_axpy(hdl, static_cast<T>(1.), &y0_vec, &y_vec);
    }
    CUDA_CHECK_ERR();
  } else {
    const cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasColMajor> AA = cml::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    cml::matrix<T, CblasColMajor> L = cml::matrix_view_array<T, CblasColMajor>
        (info->L, min_dim, min_dim);
    CUDA_CHECK_ERR();

    if (s != info->s) {
      cml::matrix_memcpy(&L, &AA);
      cml::vector<T> diagL = cml::matrix_diagonal(&L);
      cml::vector_add_constant(&diagL, s);
      cudaDeviceSynchronize();
      cml::linalg_cholesky_decomp(hdl, &L);
      CUDA_CHECK_ERR();
    }
    if (_A.Rows() >= _A.Cols()) {
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      cml::linalg_cholesky_svx(hdl, &L, &x_vec);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
    } else {
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      cml::linalg_cholesky_svx(hdl, &L, &y_vec);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      cml::blas_axpy(hdl, static_cast<T>(1.), &y0_vec, &y_vec);
    }
    CUDA_CHECK_ERR();
  }

#ifdef DEBUG
  // Verify that projection was successful.
  CheckProjection(&_A, x, y, s);
#endif

  info->s = s;
  return 0;
}

template class ProjectorDirect<double, MatrixDense<double> >;
template class ProjectorDirect<float, MatrixDense<float> >;

}  // namespace pogs

