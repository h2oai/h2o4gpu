#ifndef GSL_LINALG_HPP_
#define GSL_LINALG_HPP_

#include <cmath>

#include "gsl_blas.hpp"
#include "gsl_matrix.hpp"
#include "gsl_vector.hpp"

namespace gsl {

template <typename T>
void linalg_cholesky_decomp(matrix<T> *A) {
  const size_t M = A->size1;

  T A_00 = matrix_get(A, 0, 0);

  T L_00 = std::sqrt(A_00);

  matrix_set(A, 0, 0, L_00);

  if (M > 1) {
    T A_10 = matrix_get(A, 1, 0);
    T A_11 = matrix_get(A, 1, 1);

    T L_10 = A_10 / L_00;
    T diag = A_11 - L_10 * L_10;
    T L_11 = std::sqrt(diag);

    matrix_set(A, 1, 0, L_10);
    matrix_set(A, 1, 1, L_11);
  }

  for (unsigned int k = 2; k < M; k++) {
    T A_kk = matrix_get(A, k, k);

    for (unsigned int i = 0; i < k; i++) {
      T sum = 0;

      T A_ki = matrix_get(A, k, i);
      T A_ii = matrix_get(A, i, i);

      vector<T> ci = matrix_row(A, i);
      vector<T> ck = matrix_row(A, k);

      if (i > 0) {
        vector<T> di = vector_subvector(&ci, 0, i);
        vector<T> dk = vector_subvector(&ck, 0, i);

        blas_dot(&di, &dk, &sum);
      }

      A_ki = (A_ki - sum) / A_ii;
      matrix_set(A, k, i, A_ki);
    }

    vector<T> ck = matrix_row(A, k);
    vector<T> dk = vector_subvector(&ck, 0, k);

    T sum = blas_nrm2(&dk);
    T diag = A_kk - sum * sum;

    T L_kk = std::sqrt(diag);

    matrix_set(A, k, k, L_kk);
  }

  for (unsigned int i = 1; i < M; i++) {
    for (unsigned int j = 0; j < i; j++) {
      T A_ij = matrix_get(A, i, j);
      matrix_set(A, j, i, A_ij);
    }
  }
}

template <typename T>
void linalg_cholesky_svx(const matrix<T> *LLT, vector<T> *x) {
  blas_trsv(CblasLower, CblasNoTrans, CblasNonUnit, LLT, x);
  blas_trsv(CblasUpper, CblasNoTrans, CblasNonUnit, LLT, x);
}

}  // namespace gsl

#endif  // GSL_LINALG_HPP_

