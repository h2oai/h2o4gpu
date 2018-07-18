#ifndef KM_BLAS_CUH_
#define KM_BLAS_CUH_

#include <cublas_v2.h>
#include "KmConfig.h"

// C++ Wrappers for cublas

namespace H2O4GPU {
namespace KMeans {

namespace Blas {
// LEVEL 1
inline void axpy(cublasHandle_t handle, int n,
                           const float *alpha,
                           const float *x, int incx,
                           float *y, int incy) {
  CUBLAS_CHECK(cublasSaxpy(handle, n,
                           alpha,
                           x, incx,
                           y, incy));}

inline void axpy(cublasHandle_t handle, int n,
                 const double *alpha,
                 const double *x, int incx,
                 double *y, int incy) {
  CUBLAS_CHECK(cublasDaxpy(handle, n,
                           alpha,
                           x, incx,
                           y, incy));}

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
  CUBLAS_CHECK(cublasSgemm(handle,
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
                 const double *alpha, /* host or device pointer */
                 const double *A,
                 int lda,
                 const double *B,
                 int ldb,
                 const double *beta, /* host or device pointer */
                 double *C,
                 int ldc) {
  CUBLAS_CHECK(cublasDgemm(handle,
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

}  // Blas

}  // KMeans
}  // H2O4GPU

#endif  // KM_BLAS_CUH_