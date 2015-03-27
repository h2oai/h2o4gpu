#ifndef GSL_SPMAT_H_
#define GSL_SPMAT_H_

#include <cstdio>
#include <cstring>

#include "gsl/cblas.h"

namespace gsl {

template <typename T, typename I, CBLAS_ORDER O>
struct spmat {
  T *val;
  I *ind, *ptr;
  I m, n, nnz;
  spmat(T *val, I *ind, I *ptr, I m, I n, I nnz) 
      : val(val), ind(ind), ptr(ptr), m(m), n(n), nnz(nnz) { }
  spmat() : val(0), ind(0), m(0), n(0), nnz(0) { };
};

template <typename T, typename I, CBLAS_ORDER O>
I ptr_len(const spmat<T, I, O> &mat) {
  if (O == CblasColMajor)
    return mat.n + 1;
  else
    return mat.m + 1;
}

namespace {

template <typename T, typename I>
void csr2csc(I m, I n, I nnz, const T *a, const I *row_ptr, const I *col_ind,
             T *at, I *row_ind, I *col_ptr) {
  memset(col_ptr, 0, (n + 1) * sizeof(I));

  for (I i = 0; i < nnz; i++)
    col_ptr[col_ind[i] + 1]++;

  for (I i = 0; i < n; i++)
    col_ptr[i + 1] += col_ptr[i];

  for (I i = 0; i < m; i++) {
    for (I j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      I k = col_ind[j];
      I l = col_ptr[k]++;
      row_ind[l] = i;
      at[l] = a[j];
    }
  }

  for (I i = n; i > 0; i--)
    col_ptr[i] = col_ptr[i - 1];

  col_ptr[0] = 0;
}

template <typename T, typename I, CBLAS_ORDER O>
void MatTransp(I m, I n, I nnz, const T *val_n, const I *ptr_n, const I *ind_n,
               T *val_t, I *ind_t, I *ptr_t) {
  if (O == CblasRowMajor) {
    csr2csc(m, n, nnz, val_n, ptr_n, ind_n, val_t, ind_t, ptr_t);
  } else {
    csr2csc(n, m, nnz, val_n, ptr_n, ind_n, val_t, ind_t, ptr_t);
  }
}

}  // namespace

template <typename T, typename I, CBLAS_ORDER O>
spmat<T, I, O> spmat_alloc(I m, I n, I nnz) {
  spmat<T, I, O> mat(0, 0, 0, m, n, nnz);
  mat.val = new T[2 * nnz];
  mat.ind = new I[2 * nnz];
  mat.ptr = new I[m + n + 2];
  return mat;
}

template <typename T, typename I, CBLAS_ORDER O>
void spmat_free(spmat<T, I, O> *A) {
  delete [] A->val;
  delete [] A->ind;
  delete [] A->ptr;
}

template <typename T, typename I, CBLAS_ORDER O>
void spmat_memcpy(spmat<T, I, O> *A,
                  const T *val, const I *ind, const I *ptr) {
  memcpy(A->val, val, A->nnz * sizeof(T));
  memcpy(A->ind, ind, A->nnz * sizeof(I));
  memcpy(A->ptr, ptr, ptr_len(*A) * sizeof(I));
  MatTransp<T, I, O>(A->m, A->n, A->nnz, A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ind + A->nnz, A->ptr + ptr_len(*A));
}

}  // namespace

#endif  // GSL_SPMAT_H_

