#ifndef PROJECTOR_HELPER_CUH_
#define PROJECTOR_HELPER_CUH_

#include "cml/cml_blas.cuh" 
#include "cml/cml_vector.cuh" 
#include "matrix/matrix.h" 
#include "util.cuh" 

namespace pogs {
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
  cudaDeviceSynchronize();
  T nrm_r = cml::blas_nrm2(hdl, &y_)  / std::sqrt(A->Rows());
  DEBUG_EXPECT_EQ_EPS(nrm_r, static_cast<T>(0.), tol);

  // Check KKT
  cml::vector_memcpy(&x_, x);
  cml::vector_memcpy(&y_, y0);
  const cml::vector<T> x0_vec = cml::vector_view_array(x0, A->Cols());
  A->Mul('n', static_cast<T>(1.), x_.data, static_cast<T>(-1.), y_.data);
  cudaDeviceSynchronize();
  A->Mul('t', static_cast<T>(1.), y_.data, s, x_.data);
  cudaDeviceSynchronize();
  cml::blas_axpy(hdl, -s, &x0_vec, &x_);
  T nrm_kkt = cml::blas_nrm2(hdl, &x_) / std::sqrt(A->Cols());
  DEBUG_EXPECT_EQ_EPS(nrm_kkt, static_cast<T>(0.), tol);

  cml::vector_free(&x_);
  cml::vector_free(&y_);
  cublasDestroy(hdl);
}

}  // namespace
}  // namespace pogs

#endif  // PROJECTOR_HELPER_CUH_

