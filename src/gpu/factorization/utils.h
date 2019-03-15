#ifndef SRC_GPU_FACTORIZATION_CUSPARSE_UTILS_H
#define SRC_GPU_FACTORIZATION_CUSPARSE_UTILS_H

#include <cublas_v2.h>
#include <cusparse.h>

inline cusparseStatus_t
csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
      cusparseOperation_t transB, int m, int n, int k, int nnz,
      const float *alpha, const cusparseMatDescr_t descrA, const float *csrValA,
      const int *csrRowPtrA, const int *csrColIndA, const float *B, int ldb,
      const float *beta, float *C, int ldc)
{
    return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                           csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

inline cusparseStatus_t
csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
      cusparseOperation_t transB, int m, int n, int k, int nnz,
      const double *alpha, const cusparseMatDescr_t descrA,
      const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
      const double *B, int ldb, const double *beta, double *C, int ldc)
{
    return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                           csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

inline cublasStatus_t geam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *beta, const float *B, int ldb, float *C,
                           int ldc)
{
    return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                       C, ldc);
}

inline cublasStatus_t geam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *beta, const double *B, int ldb,
                           double *C, int ldc)
{
    return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                       C, ldc);
}
#endif