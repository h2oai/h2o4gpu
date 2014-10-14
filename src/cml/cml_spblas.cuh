#ifndef CML_SPBLAS_CUH_
#define CML_SPBLAS_CUH_

#include "cgls.cuh"
#include "cml_spmat.cuh"
#include "cml_utils.cuh"
#include "cml_vector.cuh"

namespace cml {

template <typename T, typename I, CBLAS_ORDER O>
int spblas_solve(cusparseHandle_t handle_s, cublasHandle_t handle_b,
          cusparseMatDescr_t descr, const spmat<T, I, O> *A, const T shift,
          const vector<T> *b, vector<T> *x, const T tol, const I maxit,
          bool quiet) {
  const cgls::CGLS_ORD kOrd = (O == CblasRowMajor ? cgls::CSR : cgls::CSC);
  return cgls::solve<T, kOrd>(handle_s, handle_b, descr, A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ptr + ptr_len(*A), A->ind + A->nnz, A->m, A->n,
      A->nnz, b->data, x->data, shift, tol, maxit, quiet);
}

template <typename I>
cusparseStatus_t spblas_gemv(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             cusparseMatDescr_t descrA, double alpha,
                             const spmat<double, I, CblasRowMajor> *A,
                             const vector<double> *x, double beta,
                             vector<double> *y) {
  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = cusparseDcsrmv(handle, trans, A->m, A->n, A->nnz, &alpha, descrA,
        A->val, A->ptr, A->ind, x->data, &beta, y->data);
  else
    err = cusparseDcsrmv(handle, trans, A->n, A->m, A->nnz, &alpha, descrA,
        A->val + A->nnz, A->ptr + ptr_len(*A), A->ind + A->nnz, x->data, &beta,
        y->data);
  CusparseCheckError(err);
  return err;
}

template <typename I>
cusparseStatus_t spblas_gemv(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             cusparseMatDescr_t descrA, float alpha,
                             const spmat<float, I, CblasRowMajor> *A,
                             const vector<float> *x, float beta,
                             vector<float> *y) {
  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = cusparseScsrmv(handle, trans, A->m, A->n, A->nnz, &alpha, descrA,
        A->val, A->ptr, A->ind, x->data, &beta, y->data);
  else
    err = cusparseScsrmv(handle, trans, A->n, A->m, A->nnz, &alpha, descrA,
        A->val + A->nnz, A->ptr + ptr_len(*A), A->ind + A->nnz, x->data, &beta,
        y->data);
  CusparseCheckError(err);
  return err;
}

template <typename I>
cusparseStatus_t spblas_gemv(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             cusparseMatDescr_t descrA, float alpha,
                             const spmat<float, I, CblasColMajor> *A,
                             const vector<float> *x, float beta,
                             vector<float> *y) {
  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = cusparseScsrmv(handle, trans, A->n, A->m, A->nnz, &alpha,
        descrA, A->val + A->nnz, A->ptr + ptr_len(*A), A->ind + A->nnz, x->data,
        &beta, y->data);
  else
    err = cusparseScsrmv(handle, trans, A->m, A->n, A->nnz, &alpha,
        descrA, A->val, A->ptr, A->ind, x->data, &beta, y->data);
  CusparseCheckError(err);
  return err;
}

template <typename I>
cusparseStatus_t spblas_gemv(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             cusparseMatDescr_t descrA, double alpha,
                             const spmat<double, I, CblasColMajor> *A,
                             const vector<double> *x, double beta,
                             vector<double> *y) {
  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = cusparseDcsrmv(handle, trans, A->n, A->m, A->nnz, &alpha,
        descrA, A->val + A->nnz, A->ptr + ptr_len(*A), A->ind + A->nnz, x->data,
        &beta, y->data);
  else
    err = cusparseDcsrmv(handle, trans, A->m, A->n, A->nnz, &alpha,
        descrA, A->val, A->ptr, A->ind, x->data, &beta, y->data);
  CusparseCheckError(err);
  return err;
}

}

#endif  // CML_SPBLAS_CUH_

