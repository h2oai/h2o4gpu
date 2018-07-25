/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef KM_BLAS_CUH_
#define KM_BLAS_CUH_

#include <cublas_v2.h>
#include "../../utils/utils.cuh"

// C++ Wrappers for cublas

namespace h2o4gpu {
namespace Matrix {

namespace Blas {
// LEVEL 1
inline void axpy(cublasHandle_t handle, int n,
                 const double *alpha,
                 const double *x, int incx,
                 double *y, int incy) {
  safe_cublas(cublasDaxpy(handle, n,
                           alpha,
                           x, incx,
                           y, incy));}

inline void axpy(cublasHandle_t handle, int n,
                 const float *alpha,
                 const float *x, int incx,
                 float *y, int incy) {
  safe_cublas(cublasSaxpy(handle, n,
                           alpha,
                           x, incx,
                           y, incy));}

inline void axpy(cublasHandle_t handle, int n,
                 const int *alpha,
                 const int *x, int incx,
                 int *y, int incy) {
  safe_cublas(cublasSaxpy(handle, n,
                           (const float *)alpha,
                           (const float *)x, incx,
                           (float *)y, incy));}
// LEVEL 3
inline void gemm(cublasHandle_t handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host or device pointer */
                 const float *A,
                 int lda,
                 const float *B,
                 int ldb,
                 const float *beta, /* host or device pointer */
                 float *C,
                 int ldc) {
  safe_cublas(cublasSgemm(handle,
                           transa, transb,
                           m, n, k,
                           alpha, /* host or device pointer */
                           A, lda,
                           B, ldb,
                           beta, /* host or device pointer */
                           C, ldc));}

inline void gemm(cublasHandle_t handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const double *alpha, /* host or device pointer */
                 const double *A,
                 int lda,
                 const double *B,
                 int ldb,
                 const double *beta, /* host or device pointer */
                 double *C,
                 int ldc) {
  safe_cublas(cublasDgemm(handle,
                           transa,
                           transb,
                           m,
                           n,
                           k,
                           alpha, /* host or device pointer */
                           A,
                           lda,
                           B,
                           ldb,
                           beta, /* host or device pointer */
                           C,
                           ldc));}

inline void gemm(cublasHandle_t handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const int *alpha, /* host or device pointer */
                 const int *A,
                 int lda,
                 const int *B,
                 int ldb,
                 const int *beta, /* host or device pointer */
                 int *C,
                 int ldc) {
  safe_cublas(cublasSgemm(handle,
                           transa, transb,
                           m, n, k,
                           (const float*)alpha, /* host or device pointer */
                           (const float*)A, lda,
                           (const float*)B, ldb,
                           (const float*)beta, /* host or device pointer */
                           (float*)C, ldc));}

/* -- gemm_batched --*/
inline void gemm_batched(cublasHandle_t handle,
                         cublasOperation_t transa, 
                         cublasOperation_t transb,
                         int m, int n, int k,
                         const double *alpha,
                         const double *Aarray[], int lda,
                         const double *Barray[], int ldb,
                         const double *beta,
                         double          *Carray[], int ldc, 
                         int batchCount) {
  safe_cublas(cublasDgemmBatched(handle,
                                  transa, 
                                  transb,
                                  m, n, k,
                                  alpha,
                                  Aarray, lda,
                                  Barray, ldb,
                                  beta,
                                  Carray, ldc, 
                                  batchCount));
}

inline void gemm_batched(cublasHandle_t handle,
                         cublasOperation_t transa, 
                         cublasOperation_t transb,
                         int m, int n, int k,
                         const float *alpha,
                         const float *Aarray[], int lda,
                         const float *Barray[], int ldb,
                         const float *beta,
                         float *Carray[], int ldc, 
                         int batchCount) {
  safe_cublas(cublasSgemmBatched(handle,
                                  transa, 
                                  transb,
                                  m, n, k,
                                  alpha,
                                  Aarray, lda,
                                  Barray, ldb,
                                  beta,
                                  Carray, ldc, 
                                  batchCount));
}

inline void gemm_batched(cublasHandle_t handle,
                         cublasOperation_t transa, 
                         cublasOperation_t transb,
                         int m, int n, int k,
                         const int *alpha,
                         const int *Aarray[], int lda,
                         const int *Barray[], int ldb,
                         const int *beta,
                         float *Carray[], int ldc, 
                         int batchCount) {
  safe_cublas(cublasSgemmBatched(handle,
                                  transa, 
                                  transb,
                                  m, n, k,
                                  (const float *)alpha,
                                  (const float * const *)Aarray, lda,
                                  (const float * const *)Barray, ldb,
                                  (const float *)beta,
                                  (float * const *)Carray, ldc, 
                                  batchCount));
}

/* -- gemm_strided_batched -- */
inline void gemm_strided_batched(
    cublasHandle_t handle, 
    cublasOperation_t transA, cublasOperation_t transB,
    int M, int N, int K, 
    const double* alpha,
    const double* A, int ldA, int strideA, 
    const double* B, int ldB, int strideB, 
    const double* beta,
    double* C, int ldC, int strideC,
    int batchCount) {
  safe_cublas(cublasDgemmStridedBatched(handle,
                                         transA, 
                                         transB,
                                         M, N, K,
                                         alpha,
                                         A, ldA,
                                         strideA,
                                         B, ldB,
                                         strideB,
                                         beta,
                                         C, ldC, 
                                         strideC, 
                                         batchCount));
}

inline void gemm_strided_batched(
    cublasHandle_t handle, 
    cublasOperation_t transA, cublasOperation_t transB,
    int M, int N, int K, 
    const float* alpha,
    const float* A, int ldA, int strideA, 
    const float* B, int ldB, int strideB, 
    const float* beta,
    float* C, int ldC, int strideC,
    int batchCount) {
  safe_cublas(cublasSgemmStridedBatched(handle,
                                         transA, 
                                         transB,
                                         M, N, K,
                                         alpha,
                                         A, ldA,
                                         strideA,
                                         B, ldB,
                                         strideB,
                                         beta,
                                         C, ldC, 
                                         strideC, 
                                         batchCount));
}

inline void gemm_strided_batched(
    cublasHandle_t handle, 
    cublasOperation_t transA, cublasOperation_t transB,
    int M, int N, int K, 
    const int* alpha,
    const int* A, int ldA, int strideA, 
    const int* B, int ldB, int strideB, 
    const int* beta,
    int* C, int ldC, int strideC,
    int batchCount) {
  safe_cublas(cublasSgemmStridedBatched(handle,
                                         transA, 
                                         transB,
                                         M, N, K,
                                         (const float*)alpha,
                                         (const float*)A, ldA,
                                         strideA,
                                         (const float*)B, ldB,
                                         strideB,
                                         (const float*)beta,
                                         (float*)C, ldC, 
                                         strideC, 
                                         batchCount));
}

}  // Blas
}  // Matrix
}  // h2o4gpu

#endif  // KM_BLAS_CUH_