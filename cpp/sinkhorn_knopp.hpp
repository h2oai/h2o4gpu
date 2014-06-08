#ifndef SINKHORN_KNOPP_HPP_
#define SINKHORN_KNOPP_HPP_

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

// Sinkhorn Knopp algorithm for matrix equilibration.
// The following approx. holds: diag(d) * Ain * e =  1, diag(e) * Ain' * d = 1
// Output matrix is generated as: Aout = diag(d) * Ain * diag(e),
void SinkhornKnopp(const gsl_matrix *Ain, gsl_matrix *Aout, gsl_vector *d,
                   gsl_vector *e) {
  gsl_matrix_memcpy(Aout, Ain);
  unsigned int kNumItr = 10;
  gsl_vector_set_all(d, 1.0);

  // A := |A| -- elementwise
  for (unsigned int i = 0; i < Aout->size1; ++i)
    for (unsigned int j = 0; j < Aout->size2; ++j)
      gsl_matrix_set(Aout, i, j, fabs(gsl_matrix_get(Aout, i, j)));

  // e := 1 ./ A' * d; d := 1 ./ A * e; -- k times.
  for (unsigned int k = 0; k < kNumItr; ++k) {
    gsl_blas_dgemv(CblasTrans, 1.0, Aout, d, 0.0, e);
    for (unsigned int i = 0; i < e->size; ++i)
      gsl_vector_set(e, i, 1.0 / gsl_vector_get(e, i));

    gsl_blas_dgemv(CblasNoTrans, 1.0, Aout, e, 0.0, d);
    for (unsigned int i = 0; i < d->size; ++i)
      gsl_vector_set(d, i, 1.0 / gsl_vector_get(d, i));

    double nrm_d = gsl_blas_dnrm2(d) / sqrt(static_cast<double>(d->size));
    double nrm_e = gsl_blas_dnrm2(e) / sqrt(static_cast<double>(e->size));
    double scale = sqrt(nrm_e / nrm_d);
    gsl_blas_dscal(scale, d);
    gsl_blas_dscal(1.0 / scale, e);
  }

  // A := D * A * E
  gsl_matrix_memcpy(Aout, Ain);
  for (unsigned int i = 0; i < Ain->size1; ++i) {
    gsl_vector_view v = gsl_matrix_row(Aout, i);
    gsl_blas_dscal(gsl_vector_get(d, i), &v.vector);
  }
  for (unsigned int j = 0; j < Ain->size2; ++j) {
    gsl_vector_view v = gsl_matrix_column(Aout, j);
    gsl_blas_dscal(gsl_vector_get(e, j), &v.vector);
  }
}

#endif  // SINKHORN_KNOPP_HPP_

