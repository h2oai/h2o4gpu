#ifndef SINKHORN_KNOPP_HPP_
#define SINKHORN_KNOPP_HPP_

#include <cmath>

#include "gsl/gsl_blas.hpp"
#include "gsl/gsl_matrix.hpp"
#include "gsl/gsl_vector.hpp"

// Sinkhorn Knopp algorithm for matrix equilibration.
// The following approx. holds: diag(d) * Ain * e =  1, diag(e) * Ain' * d = 1
// Output matrix is generated as: Aout = diag(d) * Ain * diag(e),
template <typename T>
void SinkhornKnopp(const gsl::matrix<T> *Ain, gsl::matrix<T> *Aout,
                   gsl::vector<T> *d, gsl::vector<T> *e) {
  unsigned int kNumItr = 10;
  const T kOne = static_cast<T>(1);
  const T kZero = static_cast<T>(0);
  gsl::matrix_memcpy(Aout, Ain);
  gsl::vector_set_all(d, kOne);

  // A := |A| -- elementwise
  for (unsigned int i = 0; i < Aout->size1; ++i)
    for (unsigned int j = 0; j < Aout->size2; ++j)
      gsl::matrix_set(Aout, i, j, std::fabs(gsl::matrix_get(Aout, i, j)));

  // e := 1 ./ A' * d; d := 1 ./ A * e; -- k times.
  for (unsigned int k = 0; k < kNumItr; ++k) {
    gsl::blas_gemv(CblasTrans, kOne, Aout, d, kZero, e);
    for (unsigned int i = 0; i < e->size; ++i)
      gsl::vector_set(e, i, kOne / gsl::vector_get(e, i));

    gsl::blas_gemv(CblasNoTrans, kOne, Aout, e, kZero, d);
    for (unsigned int i = 0; i < d->size; ++i)
      gsl::vector_set(d, i, kOne / gsl::vector_get(d, i));

    T nrm_d = gsl::blas_nrm2(d) / std::sqrt(static_cast<T>(d->size));
    T nrm_e = gsl::blas_nrm2(e) / std::sqrt(static_cast<T>(e->size));
    T scale = std::sqrt(nrm_e / nrm_d);
    gsl::blas_scal(scale, d);
    gsl::blas_scal(kOne / scale, e);
  }

  // A := D * A * E
  gsl::matrix_memcpy(Aout, Ain);
  for (unsigned int i = 0; i < Ain->size1; ++i) {
    gsl::vector<T> v = gsl::matrix_row(Aout, i);
    gsl::blas_scal(gsl::vector_get(d, i), &v);
  }
  for (unsigned int j = 0; j < Ain->size2; ++j) {
    gsl::vector<T> v = gsl::matrix_column(Aout, j);
    gsl::blas_scal(gsl::vector_get(e, j), &v);
  }
}

#endif  // SINKHORN_KNOPP_HPP_

