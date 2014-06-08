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

// Syrk
template <typename T>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, const T alpha,
                         const matrix<T> *A, const T beta,
                         matrix<T> *C);

template <>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, const float alpha,
                         const matrix<float> *A, const float beta,
                         matrix<float> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  cublasStatus_t err = cublasSsyrk(handle, uplo, trans,
      static_cast<int>(C->size1), k, &alpha, A->data, static_cast<int>(A->tda),
      &beta, C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, const double alpha,
                             const matrix<double> *A, const double beta,
                             matrix<double> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  cublasStatus_t err = cublasDsyrk(handle, uplo, trans,
      static_cast<int>(C->size1), k, &alpha, A->data, static_cast<int>(A->tda),
      &beta, C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

// Geam.
template <typename T>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const T *alpha,
                         const matrix<T> *A, const T *beta,
                         const matrix<T> *B, const matrix<T> *C);

template <>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const double *alpha,
                         const matrix<double> *A, const double *beta,
                         const matrix<double> *B, const matrix<double> *C) {
  cublasStatus_t err = cublasDgeam(handle, transa, transb,
      static_cast<int>(C->size1), static_cast<int>(C->size2), alpha, A->data,
      static_cast<int>(A->tda), beta, B->data, static_cast<int>(B->tda),
      C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const float *alpha,
                         const matrix<float> *A, const float *beta,
                         const matrix<float> *B, const matrix<float> *C) {
  cublasStatus_t err = cublasSgeam(handle, transa, transb,
      static_cast<int>(C->size1), static_cast<int>(C->size2), alpha, A->data,
      static_cast<int>(A->tda), beta, B->data, static_cast<int>(B->tda),
      C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

// Axpy.
template <typename T>
cublasStatus_t blas_axpy(cublasHandle_t handle, T alpha, const vector<T> *x,
                         vector<T> *y);

template <>
cublasStatus_t blas_axpy(cublasHandle_t handle, double alpha,
                         const vector<double> *x, vector<double> *y) {
  cublasStatus_t err = cublasDaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_axpy(cublasHandle_t handle, float alpha,
                         const vector<float> *x, vector<float> *y) {
  cublasStatus_t err = cublasSaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

// Gemv.
template <typename T>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         T alpha, matrix<T> *A, const vector<T> *x, T beta,
                         vector<T> *y);

template <>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         double alpha, matrix<double> *A,
                         const vector<double> *x, double beta,
                         vector<double> *y) {
  cublasStatus_t err = cublasDgemv(handle, trans, static_cast<int>(A->size1),
      static_cast<int>(A->size2), &alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), &beta, y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         float alpha, matrix<float> *A, const vector<float> *x,
                         float beta, vector<float> *y) {
  cublasStatus_t err = cublasSgemv(handle, trans, static_cast<int>(A->size1),
      static_cast<int>(A->size2), &alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), &beta, y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

// Symv.
template <typename T>
cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
                        T alpha, matrix<T> *A, const vector<T> *x, T beta,
                        vector<T> *y);

template <>
cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
                         double alpha, matrix<double> *A,
                         const vector<double> *x, double beta,
                         vector<double> *y) {
  cublasStatus_t err = cublasDsymv(handle, uplo, static_cast<int>(A->size1),
      &alpha, A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride), &beta, y->data, static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
                         float alpha, matrix<float> *A, const vector<float> *x,
                         float beta, vector<float> *y) {
  cublasStatus_t err = cublasSsymv(handle, uplo, static_cast<int>(A->size1),
      &alpha, A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride), &beta, y->data, static_cast<int>(y->stride));
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
  return sqrt(thrust::transform_reduce(thrust::device_pointer_cast(x->data),
      thrust::device_pointer_cast(x->data + x->size), Square<T>(),
      static_cast<T>(0.0), thrust::plus<T>()));
}

template <typename T>
cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<T> *x, T *result);

template <>
cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<float> *x,
                         float *result) {
  cublasStatus_t err = cublasSnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

// Trsv.
template <typename T>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<T> *A, vector<T> *x);

template <>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<double> *A, vector<double> *x) {
  cublasStatus_t err = cublasDtrsv(handle, uplo, trans, diag,
      static_cast<int>(A->size1), A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<float> *A, vector<float> *x) {
  cublasStatus_t err = cublasStrsv(handle, uplo, trans, diag,
      static_cast<int>(A->size1), A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

// Scal.
template <typename T>
cublasStatus_t blas_scal(cublasHandle_t handle, const T alpha, vector<T> *x);

template <>
cublasStatus_t blas_scal(cublasHandle_t handle, const double alpha,
                         vector<double> *x) {
  cublasStatus_t err = cublasDscal(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_scal(cublasHandle_t handle, const float alpha,
                         vector<float> *x) {
  cublasStatus_t err = cublasSscal(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <typename T>
cublasStatus_t blas_scal(cublasHandle_t handle, const T *alpha, vector<T> *x);

template <>
cublasStatus_t blas_scal(cublasHandle_t handle, const double *alpha,
                         vector<double> *x) {
  cublasStatus_t err = cublasDscal(handle, static_cast<int>(x->size), alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_scal(cublasHandle_t handle, const float *alpha,
                         vector<float> *x) {
  cublasStatus_t err = cublasSscal(handle, static_cast<int>(x->size), alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <typename T>
cublasStatus_t blas_asum(cublasHandle_t handle, const vector<T> *x, T *result);

template <>
cublasStatus_t blas_asum(cublasHandle_t handle, const vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDasum(handle, x->size, x->data, x->stride, result);
  CublasCheckError(err);
  return err;
}

template <>
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

}  // namespace cml

#endif  // CML_BLAS_CUH_

