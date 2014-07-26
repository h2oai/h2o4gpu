#ifndef GSL_BLAS_H_
#define GSL_BLAS_H_

#include "gsl_cblas.h"
#include "gsl_matrix.h"
#include "gsl_vector.h"

// Gnu Scientific Library
namespace gsl {

// Syrk
template <typename T>
void blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, T alpha,
               const matrix<T> *A, T beta, matrix<T> *C);

template <>
void blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, double alpha,
               const matrix<double> *A, double beta, matrix<double> *C) {
  const size_t N = C->size2;
  const size_t K = (Trans == CblasNoTrans) ? A->size2 : A->size1;

  cblas_dsyrk(CblasRowMajor, Uplo, Trans, static_cast<int>(N),
      static_cast<int>(K), alpha, A->data, static_cast<int>(A->tda), beta,
      C->data, static_cast<int>(C->tda));
}

template <>
void blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, float alpha,
               const matrix<float> *A, float beta, matrix<float> *C) {
  const size_t N = C->size2;
  const size_t K = (Trans == CblasNoTrans) ? A->size2 : A->size1;

  cblas_ssyrk(CblasRowMajor, Uplo, Trans, static_cast<int>(N),
      static_cast<int>(K), alpha, A->data, static_cast<int>(A->tda), beta,
      C->data, static_cast<int>(C->tda));
}

// Gemm
template <typename T>
void blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, T alpha,
               const matrix<T> *A, const matrix<T> *B, T beta, matrix<T> *C);

template <>
void blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, double alpha,
               const matrix<double> *A, const matrix<double> *B, double beta,
               matrix<double> *C) {
  const size_t M = C->size1;
  const size_t N = C->size2;
  const size_t NA = (TransA == CblasNoTrans) ? A->size2 : A->size1;

  cblas_dgemm(CblasRowMajor, TransA, TransB, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(NA), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda), beta,
      C->data, static_cast<int>(C->tda));
}

template <>
void blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, float alpha,
               const matrix<float> *A, const matrix<float> *B, float beta,
               matrix<float> *C) {
  const size_t M = C->size1;
  const size_t N = C->size2;
  const size_t NA = (TransA == CblasNoTrans) ? A->size2 : A->size1;

  cblas_sgemm(CblasRowMajor, TransA, TransB, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(NA), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda), beta,
      C->data, static_cast<int>(C->tda));
}

// Axpy.
template <typename T>
void blas_axpy(T alpha, const vector<T> *x, vector<T> *y);

template <>
void blas_axpy(double alpha, const vector<double> *x, vector<double> *y) {
  cblas_daxpy(static_cast<int>(x->size), alpha, x->data,
      static_cast<int>(x->stride), y->data, static_cast<int>(y->stride));
}

template <>
void blas_axpy(float alpha, const vector<float> *x, vector<float> *y) {
  cblas_saxpy(static_cast<int>(x->size), alpha, x->data,
      static_cast<int>(x->stride), y->data, static_cast<int>(y->stride));
}

// Gemv.
template <typename T>
void blas_gemv(CBLAS_TRANSPOSE_t TransA, T alpha, matrix<T> *A,
               const vector<T> *x, T beta, vector<T> *y);

template <>
void blas_gemv(CBLAS_TRANSPOSE_t TransA, double alpha, matrix<double> *A,
               const vector<double> *x, double beta, vector<double> *y) {
  cblas_dgemv(CblasRowMajor, TransA, static_cast<int>(A->size1),
      static_cast<int>(A->size2), alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), beta, y->data,
      static_cast<int>(y->stride));
}

template <>
void blas_gemv(CBLAS_TRANSPOSE_t TransA, float alpha, matrix<float> *A,
               const vector<float> *x, float beta, vector<float> *y) {
  cblas_sgemv(CblasRowMajor, TransA, static_cast<int>(A->size1),
      static_cast<int>(A->size2), alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), beta, y->data,
      static_cast<int>(y->stride));
}

// Symv.
template <typename T>
void blas_symv(CBLAS_UPLO_t Uplo, T alpha, matrix<T> *A, const vector<T> *x,
               T beta, vector<T> *y);

template <>
void blas_symv(CBLAS_UPLO_t Uplo, double alpha, matrix<double> *A,
               const vector<double> *x, double beta, vector<double> *y) {
  cblas_dsymv(CblasRowMajor, Uplo, static_cast<int>(A->size1), alpha, A->data,
      static_cast<int>(A->tda), x->data, static_cast<int>(x->stride),
      beta, y->data, static_cast<int>(y->stride));
}

template <>
void blas_symv(CBLAS_UPLO_t Uplo, float alpha, matrix<float> *A,
               const vector<float> *x, float beta, vector<float> *y) {
  cblas_ssymv(CblasRowMajor, Uplo, static_cast<int>(A->size1), alpha, A->data,
      static_cast<int>(A->tda), x->data, static_cast<int>(x->stride),
      beta, y->data, static_cast<int>(y->stride));
}

// Nrm2.
template <typename T>
T blas_nrm2(vector<T> *x);

template <>
double blas_nrm2(vector<double> *x) {
  return cblas_dnrm2(static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride));
}

template <>
float blas_nrm2(vector<float> *x) {
  return cblas_snrm2(static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride));
}

// Trsv.
template <typename T>
void blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, const matrix<T> *A, vector<T> *x);

template <>
void blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, const matrix<double> *A, vector<double> *x) {
  cblas_dtrsv(CblasRowMajor, Uplo, TransA, Diag, static_cast<int>(A->size1),
      A->data, static_cast<int>(A->tda), x->data, static_cast<int>(x->stride));
}

template <>
void blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, const matrix<float> *A, vector<float> *x) {
  cblas_strsv(CblasRowMajor, Uplo, TransA, Diag, static_cast<int>(A->size1),
      A->data, static_cast<int>(A->tda), x->data, static_cast<int>(x->stride));
}

// Trsm.
template <typename T>
void blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, T alpha, const matrix<T> *A, matrix<T> *B);

template <>
void blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, double alpha, const matrix<double> *A,
               matrix<double> *B) {
  cblas_dtrsm(CblasRowMajor, Side, Uplo, TransA, Diag,
      static_cast<int>(B->size1), static_cast<int>(B->size2), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda));
}

template <>
void blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, float alpha, const matrix<float> *A,
               matrix<float> *B) {
  cblas_strsm(CblasRowMajor, Side, Uplo, TransA, Diag,
      static_cast<int>(B->size1), static_cast<int>(B->size2), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda));
}

// Scal.
template <typename T>
void blas_scal(const T alpha, vector<T> *x);

template <>
void blas_scal(const double alpha, vector<double> *x) {
  cblas_dscal(static_cast<int>(x->size), alpha, x->data,
     static_cast<int>(x->stride));
}

template <>
void blas_scal(const float alpha, vector<float> *x) {
  cblas_sscal(static_cast<int>(x->size), alpha, x->data,
     static_cast<int>(x->stride));
}

// Asum.
template <typename T>
T blas_asum(const vector<T> *x);

template <>
double blas_asum(const vector<double> *x) {
  return cblas_dasum(static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride));
}

template <>
float blas_asum(const vector<float> *x) {
  return cblas_sasum(static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride));
}

// Dot.
template <typename T>
void blas_dot(const vector<T> *x, const vector<T> *y, T *result);

template <>
void blas_dot(const vector<double> *x, const vector<double> *y,
              double *result) {
  *result = cblas_ddot(static_cast<int>(x->size), x->data,
     static_cast<int>(x->stride), y->data, static_cast<int>(y->stride));
}

template <>
void blas_dot(const vector<float> *x, const vector<float> *y,
              float *result) {
  *result = cblas_sdot(static_cast<int>(x->size), x->data,
     static_cast<int>(x->stride), y->data, static_cast<int>(y->stride));
}

}  // namespace gsl

#endif  // GSL_BLAS_H_

