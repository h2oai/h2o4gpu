#ifndef GSL_SPBLAS_H_
#define GSL_SPBLAS_H_

#include "gsl_spmat.h"
#include "gsl_vector.h"

namespace gsl {

template <typename T, typename I, CBLAS_ORDER O>
void spblas_gemv(CBLAS_TRANSPOSE_t transA, T alpha, const spmat<T, I, O> *A,
                 const vector<T> *x, T beta, vector<T> *y) {
  T *data;
  I *col_ind;
  I *row_ptr;

  if ((O == CblasRowMajor && transA == CblasNoTrans) ||
      (O == CblasColMajor && transA == CblasTrans)) {
    data = A->val;
    col_ind = A->ind;
    row_ptr = A->ptr;
  } else {
    data = A->val + A->nnz;
    col_ind = A->ind + A->nnz;
    row_ptr = A->ptr + ptr_len(*A);
  }

  I size = transA == CblasNoTrans ? A->m : A->n;

  // TODO: Allow for use of MKL or similar.
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (I i = 0; i < size; ++i) {
    T tmp = static_cast<T>(0);
    for (I j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      tmp += data[j] * x->data[col_ind[j]];
    }
    y->data[i] = alpha * tmp + beta * y->data[i];
  }
}

}

#endif  // GSL_SPBLAS_H_

