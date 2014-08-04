#ifndef GSL_BLAS_H_
#define GSL_BLAS_H_

#include "gsl_cblas.h"
#include "gsl_matrix.h"
#include "gsl_vector.h"


// Gnu Scientific Library
namespace gsl {

//
// BLAS LEVEL 1
//

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

//
// BLAS LEVEL 2
//

// Gemv.
template <CBLAS_ORDER O>
void blas_gemv_(CBLAS_TRANSPOSE_t TransA, double alpha, matrix<double, O> *A,
                const vector<double> *x, double beta, vector<double> *y) {
  cblas_dgemv(O, TransA, static_cast<int>(A->size1),
      static_cast<int>(A->size2), alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), beta, y->data,
      static_cast<int>(y->stride));
}
template <CBLAS_ORDER O>
void blas_gemv_(CBLAS_TRANSPOSE_t TransA, float alpha, matrix<float, O> *A,
                const vector<float> *x, float beta, vector<float> *y) {
  cblas_sgemv(O, TransA, static_cast<int>(A->size1),
      static_cast<int>(A->size2), alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), beta, y->data,
      static_cast<int>(y->stride));
}
template <typename T, CBLAS_ORDER O>
void blas_gemv(CBLAS_TRANSPOSE_t TransA, T alpha, matrix<T, O> *A,
               const vector<T> *x, T beta, vector<T> *y) {
  blas_gemv_(TransA, alpha, A, x, beta, y);
}

// Trsv.
template <CBLAS_ORDER O>
void blas_trsv_(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                CBLAS_DIAG_t Diag, const matrix<double, O> *A,
                vector<double> *x) {
  cblas_dtrsv(O, Uplo, TransA, Diag, static_cast<int>(A->size1),
      A->data, static_cast<int>(A->tda), x->data, static_cast<int>(x->stride));
}
template <CBLAS_ORDER O>
void blas_trsv_(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                CBLAS_DIAG_t Diag, const matrix<float, O> *A,
                vector<float> *x) {
  cblas_strsv(O, Uplo, TransA, Diag, static_cast<int>(A->size1),
      A->data, static_cast<int>(A->tda), x->data, static_cast<int>(x->stride));
}
template <typename T, CBLAS_ORDER O>
void blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, const matrix<T, O> *A, vector<T> *x) {
  blas_trsv_(Uplo, TransA, Diag, A, x);
}

//
// BLAS LEVEL 3
//

// Syrk
template <CBLAS_ORDER O>
void blas_syrk_(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, double alpha,
                const matrix<double, O> *A, double beta, matrix<double, O> *C) {
  const size_t N = C->size2;
  const size_t K = (Trans == CblasNoTrans) ? A->size2 : A->size1;

  cblas_dsyrk(O, Uplo, Trans, static_cast<int>(N),
      static_cast<int>(K), alpha, A->data, static_cast<int>(A->tda), beta,
      C->data, static_cast<int>(C->tda));
}
template <CBLAS_ORDER O>
void blas_syrk_(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, float alpha,
                const matrix<float, O> *A, float beta, matrix<float, O> *C) {
  const size_t N = C->size2;
  const size_t K = (Trans == CblasNoTrans) ? A->size2 : A->size1;

  cblas_ssyrk(O, Uplo, Trans, static_cast<int>(N),
      static_cast<int>(K), alpha, A->data, static_cast<int>(A->tda), beta,
      C->data, static_cast<int>(C->tda));
}
template <typename T, CBLAS_ORDER O>
void blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, T alpha,
               const matrix<T, O> *A, T beta, matrix<T, O> *C) {
  blas_syrk_(Uplo, Trans, alpha, A, beta, C);
}

// Gemm
template <CBLAS_ORDER O>
void blas_gemm_(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB,
                double alpha, const matrix<double, O> *A,
                const matrix<double, O> *B, double beta, matrix<double, O> *C) {
  const size_t M = C->size1;
  const size_t N = C->size2;
  const size_t NA = (TransA == CblasNoTrans) ? A->size2 : A->size1;

  cblas_dgemm(O, TransA, TransB, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(NA), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda), beta,
      C->data, static_cast<int>(C->tda));
}
template <CBLAS_ORDER O>
void blas_gemm_(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB,
                float alpha, const matrix<float, O> *A,
                const matrix<float, O> *B, float beta, matrix<float, O> *C) {
  const size_t M = C->size1;
  const size_t N = C->size2;
  const size_t NA = (TransA == CblasNoTrans) ? A->size2 : A->size1;

  cblas_sgemm(O, TransA, TransB, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(NA), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda), beta,
      C->data, static_cast<int>(C->tda));
}
template <typename T, CBLAS_ORDER O>
void blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, T alpha,
               const matrix<T, O> *A, const matrix<T, O> *B, T beta,
               matrix<T, O> *C) {
  blas_gemm_(TransA, TransB, alpha, A, B, beta, C);  
}

// Trsm
template <CBLAS_ORDER O>
void blas_trsm_(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                CBLAS_DIAG_t Diag, double alpha, const matrix<double, O> *A,
                matrix<double, O> *B) {
  cblas_dtrsm(O, Side, Uplo, TransA, Diag,
      static_cast<int>(B->size1), static_cast<int>(B->size2), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda));
}

template <CBLAS_ORDER O>
void blas_trsm_(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                CBLAS_DIAG_t Diag, float alpha, const matrix<float, O> *A,
                matrix<float, O> *B) {
  cblas_strsm(O, Side, Uplo, TransA, Diag,
      static_cast<int>(B->size1), static_cast<int>(B->size2), alpha, A->data,
      static_cast<int>(A->tda), B->data, static_cast<int>(B->tda));
}

template <typename T, CBLAS_ORDER O>
void blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, T alpha, const matrix<T, O> *A,
               matrix<T, O> *B) {
  blas_trsm_(Side, Uplo, TransA, Diag, alpha, A, B);
}

}  // namespace gsl

#endif  // GSL_BLAS_H_

