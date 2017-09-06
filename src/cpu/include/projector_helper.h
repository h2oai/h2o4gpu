/*!
 * Modifications copyright (C) 2017 H2O.ai
 */
#ifndef PROJECTOR_HELPER_H_
#define PROJECTOR_HELPER_H_

#include "gsl/gsl_blas.h" 
#include "gsl/gsl_vector.h" 
#include "matrix/matrix.h" 
#include "util.h" 

namespace h2o4gpu {
namespace {

// Check that (x, y) satisfies two conditions
//   1. y = Ax
//   2. x = min. ||Ax - y0||_2^2 + s||x - x0||_2^2
//      <=> A^T(Ax - y0) + s(x - x0) = 0
template <typename T>
void CheckProjection(const Matrix<T> *A, const T *x0, const T *y0,
                     const T *x, const T *y, T s, T tol) {
  gsl::vector<T> x_ = gsl::vector_calloc<T>(A->Cols());
  gsl::vector<T> y_ = gsl::vector_calloc<T>(A->Rows());
  
  // Check residual
  gsl::vector_memcpy(&x_, x);
  gsl::vector_memcpy(&y_, y);
  A->Mul('n', static_cast<T>(1.), x_.data, static_cast<T>(-1.), y_.data);
  T nrm_r = gsl::blas_nrm2(&y_)  / std::sqrt(A->Rows());
  DEBUG_EXPECT_EQ_EPS(nrm_r, static_cast<T>(0.), tol);

  // Check KKT
  gsl::vector_memcpy(&x_, x);
  gsl::vector_memcpy(&y_, y0);
  const gsl::vector<T> x0_vec = gsl::vector_view_array(x0, A->Cols());
  A->Mul('n', static_cast<T>(1.), x_.data, static_cast<T>(-1.), y_.data);
  A->Mul('t', static_cast<T>(1.), y_.data, s, x_.data);
  gsl::blas_axpy(-s, &x0_vec, &x_);
  T nrm_kkt = gsl::blas_nrm2(&x_) / std::sqrt(A->Cols());
  DEBUG_EXPECT_EQ_EPS(nrm_kkt, static_cast<T>(0.), tol);

  gsl::vector_free(&x_);
  gsl::vector_free(&y_);
}

}  // namespace
}  // namespace h2o4gpu

#endif  // PROJECTOR_HELPER_H_

