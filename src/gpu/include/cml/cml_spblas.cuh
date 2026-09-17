#ifndef CML_SPBLAS_CUH_
#define CML_SPBLAS_CUH_

#include "cml_spmat.cuh"
#include "cml_utils.cuh"
#include "cml_vector.cuh"

namespace cml {

namespace {

// CUDA 12 removed the legacy cusparse<t>csrmv routines. This helper reproduces
// y = alpha * A * x + beta * y for a CSR matrix using the generic cusparseSpMV
// API. Callers always use CUSPARSE_OPERATION_NON_TRANSPOSE (transpose is baked
// into the stored pointers), so x has length n (cols) and y has length m (rows).
template <typename T>
cusparseStatus_t SpMvCsr(cusparseHandle_t handle, int m, int n, int nnz,
                         const T *alpha, const T *csr_val,
                         const int *csr_row_ptr, const int *csr_col_ind,
                         const T *x, const T *beta, T *y,
                         cudaDataType val_type) {
  cusparseSpMatDescr_t mat_a;
  cusparseDnVecDescr_t vec_x, vec_y;
  cusparseStatus_t err = cusparseCreateCsr(
      &mat_a, m, n, nnz, const_cast<int *>(csr_row_ptr),
      const_cast<int *>(csr_col_ind), const_cast<T *>(csr_val),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      val_type);
  if (err != CUSPARSE_STATUS_SUCCESS) return err;
  cusparseCreateDnVec(&vec_x, n, const_cast<T *>(x), val_type);
  cusparseCreateDnVec(&vec_y, m, y, val_type);
  size_t buffer_size = 0;
  err = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha,
                                mat_a, vec_x, beta, vec_y, val_type,
                                CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
  void *buffer = nullptr;
  if (buffer_size > 0) CudaCheckError(cudaMalloc(&buffer, buffer_size));
  err = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a,
                     vec_x, beta, vec_y, val_type, CUSPARSE_SPMV_ALG_DEFAULT,
                     buffer);
  if (buffer) cudaFree(buffer);
  cusparseDestroySpMat(mat_a);
  cusparseDestroyDnVec(vec_x);
  cusparseDestroyDnVec(vec_y);
  return err;
}

}  // namespace

template <typename I>
cusparseStatus_t spblas_gemv(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             cusparseMatDescr_t descrA, double alpha,
                             const spmat<double, I, CblasRowMajor> *A,
                             const vector<double> *x, double beta,
                             vector<double> *y) {
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = SpMvCsr<double>(handle, A->m, A->n, A->nnz, &alpha, A->val, A->ptr,
                          A->ind, x->data, &beta, y->data, CUDA_R_64F);
  else
    err = SpMvCsr<double>(handle, A->n, A->m, A->nnz, &alpha, A->val + A->nnz,
                          A->ptr + ptr_len(*A), A->ind + A->nnz, x->data, &beta,
                          y->data, CUDA_R_64F);
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
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = SpMvCsr<double>(handle, A->m, A->n, A->nnz, &alpha, A->val + A->nnz,
                          A->ptr + ptr_len(*A), A->ind + A->nnz, x->data, &beta,
                          y->data, CUDA_R_64F);
  else
    err = SpMvCsr<double>(handle, A->n, A->m, A->nnz, &alpha, A->val, A->ptr,
                          A->ind, x->data, &beta, y->data, CUDA_R_64F);
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
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = SpMvCsr<float>(handle, A->m, A->n, A->nnz, &alpha, A->val, A->ptr,
                         A->ind, x->data, &beta, y->data, CUDA_R_32F);
  else
    err = SpMvCsr<float>(handle, A->n, A->m, A->nnz, &alpha, A->val + A->nnz,
                         A->ptr + ptr_len(*A), A->ind + A->nnz, x->data, &beta,
                         y->data, CUDA_R_32F);
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
  cusparseStatus_t err;
  if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    err = SpMvCsr<float>(handle, A->m, A->n, A->nnz, &alpha, A->val + A->nnz,
                         A->ptr + ptr_len(*A), A->ind + A->nnz, x->data, &beta,
                         y->data, CUDA_R_32F);
  else
    err = SpMvCsr<float>(handle, A->n, A->m, A->nnz, &alpha, A->val, A->ptr,
                         A->ind, x->data, &beta, y->data, CUDA_R_32F);
  CusparseCheckError(err);
  return err;
}

}

#endif  // CML_SPBLAS_CUH_

