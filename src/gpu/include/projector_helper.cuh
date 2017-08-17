#pragma once

#include "cml/cml_blas.cuh" 
#include "cml/cml_vector.cuh" 
#include "matrix/matrix.h" 
#include "util.h" 

namespace h2ogpuml {
namespace {

// Check that (x, y) satisfies two conditions
//   1. y = Ax
//   2. x = min. ||Ax - y0||_2^2 + s||x - x0||_2^2
//      <=> A^T(Ax - y0) + s(x - x0) = 0
template <typename T>
void CheckProjection(const Matrix<T> *A, const T *x0, const T *y0,
                     const T *x, const T *y, T s, T tol) {
  cublasHandle_t hdl;
  cublasCreate(&hdl);
  cml::vector<T> x_ = cml::vector_calloc<T>(A->Cols());
  cml::vector<T> y_ = cml::vector_calloc<T>(A->Rows());
  
  // Check residual
  cml::vector_memcpy(&x_, x);
  cml::vector_memcpy(&y_, y);
  A->Mul('n', static_cast<T>(1.), x_.data, static_cast<T>(-1.), y_.data);
  wrapcudaDeviceSynchronize();
  T nrm_r = cml::blas_nrm2(hdl, &y_)  / std::sqrt(A->Rows());
#ifdef DEBUG
  if(nrm_r/tol>1.0){
    fprintf(stderr,"CHECKPROJECTION: nrm_r/tol=%g <1?\n",nrm_r/tol);
  }
#endif
  DEBUG_EXPECT_EQ_EPS(nrm_r, static_cast<T>(0.), tol);

  // Check KKT
  cml::vector_memcpy(&x_, x);
  cml::vector_memcpy(&y_, y0);
  const cml::vector<T> x0_vec = cml::vector_view_array(x0, A->Cols());
  A->Mul('n', static_cast<T>(1.), x_.data, static_cast<T>(-1.), y_.data);
  wrapcudaDeviceSynchronize();
  A->Mul('t', static_cast<T>(1.), y_.data, s, x_.data);
  wrapcudaDeviceSynchronize();
  cml::blas_axpy(hdl, -s, &x0_vec, &x_);
  T nrm_kkt = cml::blas_nrm2(hdl, &x_) / std::sqrt(A->Cols());
#ifdef DEBUG
  if(nrm_kkt/tol>1.0){
    fprintf(stderr,"CHECKPROJECTION: nrm_kkt/tol=%g <1?\n",nrm_kkt/tol);
  }
#endif
  DEBUG_EXPECT_EQ_EPS(nrm_kkt, static_cast<T>(0.), tol);

  cml::vector_free(&x_);
  cml::vector_free(&y_);
  cublasDestroy(hdl);
}

}  // namespace
}  // namespace h2ogpuml
