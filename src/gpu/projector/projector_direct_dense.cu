/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
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
#include "timer.h"

#include "h2o4gpuglm.h"

#include "../include/cuda_utils.h"

namespace h2o4gpu {

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
ProjectorDirect<T, M>::ProjectorDirect(int wDev, const M& A)
  : _wDev(wDev), _A(A) {

  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  DEBUG_FPRINTF(stderr,"Rows=%d Cols=%d done_init=%d\n",(int)_A.Rows(),(int)_A.Cols(),_A.IsInit());

  // Set GPU specific this->_info.
  PUSH_RANGE("PDnew",PDnew,1);
  GpuData<T> *info = new GpuData<T>();
  this->_info = reinterpret_cast<void*>(info);
  POP_RANGE("PDnew",PDnew,1);
}

template <typename T, typename M>
ProjectorDirect<T, M>::ProjectorDirect(const M& A)
  : _wDev(A._wDev), _A(A) {

  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  DEBUG_FPRINTF(stderr,"Rows=%d Cols=%d done_init=%d\n",(int)_A.Rows(),(int)_A.Cols(),_A.IsInit());

  // Set GPU specific this->_info.
  PUSH_RANGE("PDnew",PDnew,1);
  GpuData<T> *info = new GpuData<T>();
  this->_info = reinterpret_cast<void*>(info);
  POP_RANGE("PDnew",PDnew,1);
}


template <typename T, typename M>
ProjectorDirect<T, M>::~ProjectorDirect() {


  if(0){ // FIXME: segfaults sometimes
	  checkwDev(_wDev);
	  CUDACHECK(cudaSetDevice(_wDev));

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

}

template <typename T, typename M>
int ProjectorDirect<T, M>::Init() {
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  CUDACHECK(cudaSetDevice(_wDev));
  ASSERT(_A.IsInit());


  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  PUSH_RANGE("AAalloc",AAalloc,1);
  cudaMalloc(&(info->AA), min_dim * min_dim * sizeof(T));
  cudaMalloc(&(info->L), min_dim * min_dim * sizeof(T));
  cudaMemset(info->AA, 0, min_dim * min_dim * sizeof(T));
  cudaMemset(info->L, 0, min_dim * min_dim * sizeof(T));
  DEBUG_FPRINTF(stderr,"TEST: r=%d c=%d : %d %d\n",(int)_A.Rows(), (int)_A.Cols(), (int)min_dim,(int)sizeof(T));
  CUDA_CHECK_ERR();
  POP_RANGE("AAalloc",AAalloc,1);

  cublasOperation_t op_type = _A.Rows() > _A.Cols()
      ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Compute AA (i.e. Gramian matrix)
  PUSH_RANGE("AAcompute(gram)",AAcompute,1);
  double t0 = timer<double>();

  if (_A.Order() == MatrixDense<T>::ROW) {
    const cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasRowMajor> AA = cml::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    //C := alpha*A*A' + beta*C
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

  double t1 = timer<double>() - t0;
  DEBUG_FPRINTF(stderr,"Time to compute the Gram: %f\n", t1);
  CUDA_CHECK_ERR();
  POP_RANGE("AAcompute(gram)",AAcompute,1);

  return 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Project(const T *x0, const T *y0, T s, T *x, T *y,
                                   T tol) {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init || s < static_cast<T>(0.))
    return 1;
  CUDACHECK(cudaSetDevice(_wDev));

  // Get Cublas handle
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  PUSH_RANGE("P1alloc",P1alloc,2);
  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  // Set up views for raw vectors.
  cml::vector<T> y_vec = cml::vector_view_array(y, _A.Rows()); // y^{k+1/2} to be updated to y^{k+1}
  const cml::vector<T> y0_vec = cml::vector_view_array(y0, _A.Rows()); // \tilde{y}^{k} input only
  cml::vector<T> x_vec = cml::vector_view_array(x, _A.Cols()); // x^{k+1/2} to be updated to x^{k+1}
  const cml::vector<T> x0_vec = cml::vector_view_array(x0, _A.Cols()); // \tilde{x}^{k} input only

  // Set (x, y) = (x0, y0).
  cml::vector_memcpy(&x_vec, &x0_vec);
  cml::vector_memcpy(&y_vec, &y0_vec);
  CUDA_CHECK_ERR();
  POP_RANGE("P1alloc",P1alloc,2);

  double t0 = timer<double>();
  if (_A.Order() == MatrixDense<T>::ROW) {
    PUSH_RANGE("P1(row)",P1row,2);
    const cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasRowMajor> AA = cml::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    cml::matrix<T, CblasRowMajor> L = cml::matrix_view_array<T, CblasRowMajor>
        (info->L, min_dim, min_dim);
    CUDA_CHECK_ERR();
    POP_RANGE("P1(row)",P1row,2);

    if (s != info->s) {
      PUSH_RANGE("P1r_diagonal",P1r_diagonal,2);
      cml::matrix_memcpy(&L, &AA);
      cml::vector<T> diagL = cml::matrix_diagonal(&L); // vector view of diagonal of L
      cml::vector_add_constant(&diagL, s); // add s=kOne=1 to diagonal of L
      wrapcudaDeviceSynchronize(); // not needed as next call is cuda call that will occur sequentially on device
      CUDA_CHECK_ERR();
      POP_RANGE("P1r_diagonal",P1r_diagonal,2);

      PUSH_RANGE("P1r_cholesky_decomp",P1r_cholesky_decomp,2);
      // L input contains AA + I, L on output has cholesky of input
      cml::linalg_cholesky_decomp(hdl, &L);
      wrapcudaDeviceSynchronize(); // not needed as next call is cuda call that will occur sequentially on device
      CUDA_CHECK_ERR();
      POP_RANGE("P1r_cholesky_decomp",P1r_cholesky_decomp,2);
    }
    if (_A.Rows() > _A.Cols()) {
      PUSH_RANGE("P1r_gemv(r>c)",P1r_gemvrgc,2);
      // 1*A*y + 1*x -> x
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P1r_gemv(r>c)",P1r_gemvrgc,2);
      PUSH_RANGE("P1r_cholesky_svx",P1r_cholesky_svx,2);
      // Solve LL^T x=b for x (where output for x_vec:= x^{k+1} := (A^T A + I)^{-1} (c + A^t d) in h2o4gpu paper)
      cml::linalg_cholesky_svx(hdl, &L, &x_vec);
      POP_RANGE("P1r_cholesky_svx",P1r_cholesky_svx,2);
      PUSH_RANGE("P1r_gemv2",P1r_gemv2,2);
      // 1*A*x + 0*y -> y (y^{k+1} := A x^{k+1} in h2o4gpu paper)
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
      POP_RANGE("P1r_gemv2",P1r_gemv2,2);
    } else {
      PUSH_RANGE("P1r_gemv",P1r_gemv,2);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      POP_RANGE("P1r_gemv",P1r_gemv,2);
      PUSH_RANGE("P1r_cholesky_svx",P1r_cholesky_svx,2);
      cml::linalg_cholesky_svx(hdl, &L, &y_vec);
      POP_RANGE("P1r_cholesky_svx",P1r_cholesky_svx,2);
      PUSH_RANGE("P1r_gemv2",P1r_gemv2,2);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P1r_gemv2",P1r_gemv2,2);
      PUSH_RANGE("P1r_axpy",P1r_axpy,2);
      cml::blas_axpy(hdl, static_cast<T>(1.), &y0_vec, &y_vec);
      POP_RANGE("P1r_axpy",P1r_axpy,2);
    }
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  } else {
    PUSH_RANGE("P1(col)",P1col,2);
    const cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    cml::matrix<T, CblasColMajor> AA = cml::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    cml::matrix<T, CblasColMajor> L = cml::matrix_view_array<T, CblasColMajor>
        (info->L, min_dim, min_dim);
    CUDA_CHECK_ERR();
    POP_RANGE("P1(col)",P1col,2);

    if (s != info->s) {
      PUSH_RANGE("P1c_diagonal",P1c_diagonal,2);
      cml::matrix_memcpy(&L, &AA);
      cml::vector<T> diagL = cml::matrix_diagonal(&L);
      cml::vector_add_constant(&diagL, s);
      wrapcudaDeviceSynchronize();
      CUDA_CHECK_ERR();
      POP_RANGE("P1c_diagonal",P1c_diagonal,2);
      PUSH_RANGE("P1c_cholesky_decomp",P1c_cholesky_decomp,2);
      cml::linalg_cholesky_decomp(hdl, &L);
      wrapcudaDeviceSynchronize();
      CUDA_CHECK_ERR();
      POP_RANGE("P1c_cholesky_decomp",P1c_cholesky_decomp,2);
    }
    if (_A.Rows() > _A.Cols()) {
      PUSH_RANGE("P1c_gemv(r>c)",P1c_gemvrgc,2);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P1c_gemv(r>c)",P1c_gemvrgc,2);
      PUSH_RANGE("P1c_cholesky_svx",P1c_cholesky_svx,2);
      cml::linalg_cholesky_svx(hdl, &L, &x_vec);
      POP_RANGE("P1c_cholesky_svx",P1c_cholesky_svx,2);
      PUSH_RANGE("P1c_gemv2",P1c_gemv2,2);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
      POP_RANGE("P1c_gemv2",P1c_gemv2,2);
    } else {
      PUSH_RANGE("P1c_gemv",P1c_gemv,2);
      cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      POP_RANGE("P1c_gemv",P1c_gemv,2);
      PUSH_RANGE("P1c_cholesky_svx",P1c_cholesky_svx,2);
      cml::linalg_cholesky_svx(hdl, &L, &y_vec);
      POP_RANGE("P1c_cholesky_svx",P1c_cholesky_svx,2);
      PUSH_RANGE("P1c_gemv2",P1c_gemv2,2);
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      POP_RANGE("P1c_gemv2",P1c_gemv2,2);
      PUSH_RANGE("P1c_axpy",P1c_axpy,2);
      cml::blas_axpy(hdl, static_cast<T>(1.), &y0_vec, &y_vec);
      POP_RANGE("P1c_axpy",P1c_axpy,2);
    }
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  PUSH_RANGE("P2",P2,1);
#ifdef DEBUG
  double t1 = timer<double>() - t0;
  printf("Time to compute Cholesky decomp and backward solve: %f\n", t1);

  // Verify that projection was successful.
  CheckProjection(&_A, x0, y0, x, y, s,
      static_cast<T>(1e3) * std::numeric_limits<T>::epsilon());
#endif
  cudaDeviceSynchronize(); // added synch
  POP_RANGE("P2",P2,1);
  
  info->s = s;
  return 0;
}

#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE==1
template class ProjectorDirect<double, MatrixDense<double> >;
#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE==1
template class ProjectorDirect<float, MatrixDense<float> >;
#endif

}  // namespace h2o4gpu

