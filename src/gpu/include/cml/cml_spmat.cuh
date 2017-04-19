#ifndef CML_SPMAT_CUH_
#define CML_SPMAT_CUH_

#include <cusparse.h>

#include "cml/cblas.h"
#include "cml/cml_defs.cuh"
#include "cml/cml_utils.cuh"
#include <cassert>

namespace cml {

template <typename T, typename I, CBLAS_ORDER O>
struct spmat {
  cusparseMatDescr_t descr;
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

template <CBLAS_ORDER O>
cusparseStatus_t MatTransp(cusparseHandle_t handle, int m, int n, int nnz,
                           const float *val_n, const int *ptr_n,
                           const int *ind_n, float *val_t, int *ind_t,
                           int *ptr_t) {
  cusparseStatus_t err;
  if (O == CblasRowMajor) {
    err = cusparseScsr2csc(handle, m, n, nnz, val_n,
        ptr_n, ind_n, val_t, ind_t, ptr_t, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO);
    int tmp;
    cudaMemcpy(&tmp, ptr_n + m, sizeof(int), cudaMemcpyDeviceToHost);
    printf("__%d %d\n", nnz, tmp);
  } else {
    err = cusparseScsr2csc(handle, n, m, nnz, val_n,
        ptr_n, ind_n, val_t, ind_t, ptr_t, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO);
  }
  CusparseCheckError(err);
  return err;
}

template <CBLAS_ORDER O>
cusparseStatus_t MatTransp(cusparseHandle_t handle, int m, int n, int nnz,
                           const double *val_n, const int *ptr_n,
                           const int *ind_n, double *val_t, int *ind_t,
                           int *ptr_t) {
  cusparseStatus_t err;
  if (O == CblasRowMajor) {
    err = cusparseDcsr2csc(handle, m, n, nnz, val_n,
        ptr_n, ind_n, val_t, ind_t, ptr_t, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO);
  } else {
    err = cusparseDcsr2csc(handle, n, m, nnz, val_n,
        ptr_n, ind_n, val_t, ind_t, ptr_t, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO);
  }
  CusparseCheckError(err);
  return err;
}

}  // namespace

template <typename T, typename I, CBLAS_ORDER O>
spmat<T, I, O> spmat_alloc(int m, int n, int nnz) {
  spmat<T, I, O> mat(0, 0, 0, m, n, nnz);
  cudaError_t err;
  err = cudaMalloc(&mat.val, 2 * nnz * sizeof(T));
  CudaCheckError(err);
  err = cudaMalloc(&mat.ind, 2 * nnz * sizeof(I));
  CudaCheckError(err);
  err = cudaMalloc(&mat.ptr, (m + n + 2) * sizeof(I));
  CudaCheckError(err);
  return mat;
}

template <typename T, typename I, CBLAS_ORDER O>
void spmat_free(spmat<T, I, O> *A) {
  cudaError_t err;
  err = cudaFree(A->val);
  CudaCheckError(err);
  err = cudaFree(A->ind);
  CudaCheckError(err);
  err = cudaFree(A->ptr);
}

template <typename T, typename I, CBLAS_ORDER O>
void spmat_memcpy(cusparseHandle_t handle, spmat<T, I, O> *A,
                  const T *val, const I *ind, const I *ptr) {
  cudaMemcpy(A->val, val, A->nnz * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(A->ind, ind, A->nnz * sizeof(I), cudaMemcpyHostToDevice);
  cudaMemcpy(A->ptr, ptr, ptr_len(*A) * sizeof(I), cudaMemcpyHostToDevice);
  MatTransp<O>(handle, A->m, A->n, A->nnz, A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ind + A->nnz, A->ptr + ptr_len(*A));
}

}  // namespace

#endif  // CML_SPMAT_CUH_

