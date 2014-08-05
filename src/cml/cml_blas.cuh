#ifndef CML_BLAS_CUH_
#define CML_BLAS_CUH_

#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "cml_matrix.cuh"
#include "cml_utils.cuh"
#include "cml_vector.cuh"


// Cuda Matrix Library
namespace cml {

namespace {
  cublasFillMode_t InvFillMode(cublasFillMode_t uplo) {
    return uplo == CUBLAS_FILL_MODE_LOWER
        ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  }
  cublasOperation_t InvOp(cublasOperation_t trans) {
    return trans == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
}

//
// BLAS LEVEL 1
//

// Axpy.
cublasStatus_t blas_axpy(cublasHandle_t handle, double alpha,
                         const vector<double> *x, vector<double> *y) {
  cublasStatus_t err = cublasDaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_axpy(cublasHandle_t handle, float alpha,
                         const vector<float> *x, vector<float> *y) {
  cublasStatus_t err = cublasSaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

// Nrm2.
template <typename T>
struct Square : thrust::unary_function<T, T> {
  __device__ T operator()(const T &x) {
    return x * x;
  }
};

template <typename T>
T blas_nrm2(cublasHandle_t handle, vector<T> *x) {
  strided_range<thrust::device_ptr<T> > strided_x(
      thrust::device_pointer_cast(x->data),
      thrust::device_pointer_cast(x->data + x->stride * x->size), x->stride);
  T nrm2 = sqrt(thrust::transform_reduce(strided_x.begin(), strided_x.end(),
      Square<T>(), static_cast<T>(0.0), thrust::plus<T>()));
  return nrm2;
}

cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<float> *x,
                         float *result) {
  cublasStatus_t err = cublasSnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

// Scal.
cublasStatus_t blas_scal(cublasHandle_t handle, const double alpha,
                         vector<double> *x) {
  cublasStatus_t err = cublasDscal(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_scal(cublasHandle_t handle, const float alpha,
                         vector<float> *x) {
  cublasStatus_t err = cublasSscal(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_scal(cublasHandle_t handle, const double *alpha,
                         vector<double> *x) {
  cublasStatus_t err = cublasDscal(handle, static_cast<int>(x->size), alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_scal(cublasHandle_t handle, const float *alpha,
                         vector<float> *x) {
  cublasStatus_t err = cublasSscal(handle, static_cast<int>(x->size), alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

// Asum.
cublasStatus_t blas_asum(cublasHandle_t handle, const vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDasum(handle, x->size, x->data, x->stride, result);
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_asum(cublasHandle_t handle, const vector<float> *x,
                         float *result) {
  cublasStatus_t err = cublasSasum(handle, x->size, x->data, x->stride, result);
  CublasCheckError(err);
  return err;
}

template <typename T>
T blas_asum(cublasHandle_t handle, const vector<T> *x) {
  T result;
  blas_asum(handle, x, &result);
  return result;
}

// Dot.
cublasStatus_t blas_dot(cublasHandle_t handle, const vector<double> *x,
                        const vector<double> *y, double *result) {
  cublasStatus_t err = cublasDdot(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), y->data, static_cast<int>(y->stride), result);
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_dot(cublasHandle_t handle, const vector<float> *x,
                        const vector<float> *y, float *result) {
  cublasStatus_t err = cublasSdot(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), y->data, static_cast<int>(y->stride), result);
  CublasCheckError(err);
  return err;
}

//
// BLAS LEVEL 2
//

// Gemv.
template <CBLAS_ORDER O>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         double alpha, matrix<double, O> *A,
                         const vector<double> *x, double beta,
                         vector<double> *y) {
  cublasStatus_t err;
  if (O == CblasColMajor) {
    err = cublasDgemv(handle, trans, static_cast<int>(A->size1),
        static_cast<int>(A->size2), &alpha, A->data, static_cast<int>(A->tda),
        x->data, static_cast<int>(x->stride), &beta, y->data,
        static_cast<int>(y->stride));
   } else {
    trans = InvOp(trans);
    err = cublasDgemv(handle, trans, static_cast<int>(A->size2),
        static_cast<int>(A->size1), &alpha, A->data, static_cast<int>(A->tda),
        x->data, static_cast<int>(x->stride), &beta, y->data,
        static_cast<int>(y->stride));
  }
  CublasCheckError(err);
  return err;
}

template <CBLAS_ORDER O>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         float alpha, matrix<float, O> *A, const vector<float> *x,
                         float beta, vector<float> *y) {
  cublasStatus_t err;
  if (O == CblasColMajor) {
    err = cublasSgemv(handle, trans, static_cast<int>(A->size1),
        static_cast<int>(A->size2), &alpha, A->data, static_cast<int>(A->tda),
        x->data, static_cast<int>(x->stride), &beta, y->data,
        static_cast<int>(y->stride));
   } else {
    trans = InvOp(trans);
    err = cublasSgemv(handle, trans, static_cast<int>(A->size2),
        static_cast<int>(A->size1), &alpha, A->data, static_cast<int>(A->tda),
        x->data, static_cast<int>(x->stride), &beta, y->data,
        static_cast<int>(y->stride));
  }
  CublasCheckError(err);
  return err;
}

// Trsv.
template <CBLAS_ORDER O>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<double, O> *A, vector<double> *x) {
  if (O == CblasRowMajor) {
    uplo = InvFillMode(uplo);
    trans = InvOp(trans);
  }
  cublasStatus_t err = cublasDtrsv(handle, uplo, trans, diag,
      static_cast<int>(A->size1), A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <CBLAS_ORDER O>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<float, O> *A, vector<float> *x) {
  if (O == CblasRowMajor) {
    uplo = InvFillMode(uplo);
    trans = InvOp(trans);
  }
  cublasStatus_t err = cublasStrsv(handle, uplo, trans, diag,
      static_cast<int>(A->size1), A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

//
// BLAS LEVEL 3
//

// Syrk.
template <CBLAS_ORDER O>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, const float alpha,
                         const matrix<float, O> *A, const float beta,
                         matrix<float, O> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  if (O == CblasRowMajor) {
    uplo = InvFillMode(uplo);
    trans = InvOp(trans);
  }
  cublasStatus_t err = cublasSsyrk(handle, uplo, trans,
      static_cast<int>(C->size1), k, &alpha, A->data, static_cast<int>(A->tda),
      &beta, C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

template <CBLAS_ORDER O>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, const double alpha,
                         const matrix<double, O> *A, const double beta,
                         matrix<double, O> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  if (O == CblasRowMajor) {
    uplo = InvFillMode(uplo);
    trans = InvOp(trans);
  }
  cublasStatus_t err = cublasDsyrk(handle, uplo, trans,
      static_cast<int>(C->size1), k, &alpha, A->data, static_cast<int>(A->tda),
      &beta, C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

// Gemm.
template <CBLAS_ORDER O>
cublasStatus_t blas_gemm(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const float alpha,
                         const matrix<float, O> *A, const matrix<float, O> *B,
                         const float beta, matrix<float, O> *C) {
  int k = transa == CUBLAS_OP_N ? A->size2 : A->size1;
  cublasStatus_t err;
  if (O == CblasColMajor)
    err = cublasSgemm(handle, transa, transb, static_cast<int>(C->size1),
        static_cast<int>(C->size2), k, &alpha, A->data,
        static_cast<int>(A->tda), B->data, static_cast<int>(B->tda), &beta,
        C->data, static_cast<int>(C->tda));
  else
    err = cublasSgemm(handle, transb, transa, static_cast<int>(C->size2),
        static_cast<int>(C->size1), k, &alpha, B->data,
        static_cast<int>(B->tda), A->data, static_cast<int>(A->tda), &beta,
        C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

template <CBLAS_ORDER O>
cublasStatus_t blas_gemm(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const double alpha,
                         const matrix<double, O> *A, const matrix<double, O> *B,
                         const double beta, matrix<double, O> *C) {
  int k = transa == CUBLAS_OP_N ? A->size2 : A->size1;
  cublasStatus_t err;
  if (O == CblasColMajor)
    err = cublasDgemm(handle, transa, transb, static_cast<int>(C->size1),
        static_cast<int>(C->size2), k, &alpha, A->data,
        static_cast<int>(A->tda), B->data, static_cast<int>(B->tda), &beta,
        C->data, static_cast<int>(C->tda));
  else
    err = cublasDgemm(handle, transb, transa, static_cast<int>(C->size2),
        static_cast<int>(C->size1), k, &alpha, B->data,
        static_cast<int>(B->tda), A->data, static_cast<int>(A->tda), &beta,
        C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}



// Geam.
// template <typename T>
// cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
//                          cublasOperation_t transb, const T *alpha,
//                          const matrix<T> *A, const T *beta,
//                          const matrix<T> *B, const matrix<T> *C);
// 
// template <>
// cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
//                          cublasOperation_t transb, const double *alpha,
//                          const matrix<double, O> *A, const double *beta,
//                          const matrix<double, O> *B, const matrix<double, O> *C) {
//   cublasStatus_t err = cublasDgeam(handle, transa, transb,
//       static_cast<int>(C->size1), static_cast<int>(C->size2), alpha, A->data,
//       static_cast<int>(A->tda), beta, B->data, static_cast<int>(B->tda),
//       C->data, static_cast<int>(C->tda));
//   CublasCheckError(err);
//   return err;
// }
// 
// template <>
// cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
//                          cublasOperation_t transb, const float *alpha,
//                          const matrix<float, O> *A, const float *beta,
//                          const matrix<float, O> *B, const matrix<float, O> *C) {
//   cublasStatus_t err = cublasSgeam(handle, transa, transb,
//       static_cast<int>(C->size1), static_cast<int>(C->size2), alpha, A->data,
//       static_cast<int>(A->tda), beta, B->data, static_cast<int>(B->tda),
//       C->data, static_cast<int>(C->tda));
//   CublasCheckError(err);
//   return err;
// }
// Symv.
// template <typename T>
// cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
//                         T alpha, matrix<T> *A, const vector<T> *x, T beta,
//                         vector<T> *y);
// 
// template <>
// cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
//                          double alpha, matrix<double, O> *A,
//                          const vector<double> *x, double beta,
//                          vector<double> *y) {
//   cublasStatus_t err = cublasDsymv(handle, uplo, static_cast<int>(A->size1),
//       &alpha, A->data, static_cast<int>(A->tda), x->data,
//       static_cast<int>(x->stride), &beta, y->data, static_cast<int>(y->stride));
//   CublasCheckError(err);
//   return err;
// }
// 
// template <>
// cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
//                          float alpha, matrix<float, O> *A, const vector<float> *x,
//                          float beta, vector<float> *y) {
//   cublasStatus_t err = cublasSsymv(handle, uplo, static_cast<int>(A->size1),
//       &alpha, A->data, static_cast<int>(A->tda), x->data,
//       static_cast<int>(x->stride), &beta, y->data, static_cast<int>(y->stride));
//   CublasCheckError(err);
//   return err;
// }

}  // namespace cml

#endif  // CML_BLAS_CUH_

