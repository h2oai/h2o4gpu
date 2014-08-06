#include <mex.h>

#include "../gsl/cblas.h"

#define INT mwSignedIndex

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
float  sdot_(const INT *N, const float  *X, const INT *incX, const float  *Y,
             const INT *incY);
float  cblas_sdot(const int N, const float  *X, const int incX, const float  *Y,
                  const int incY) {
  INT N_ = N, incX_ = incX, incY_ = incY;
  return sdot_(&N_, X, &incX_, Y, &incY_);
}
double ddot_(const INT *N, const double *X, const INT *incX, const double *Y,
             const INT *incY);
double cblas_ddot(const int N, const double *X, const int incX, const double *Y,
                  const int incY) {
  INT N_ = N, incX_ = incX, incY_ = incY;
  return ddot_(&N_, X, &incX_, Y, &incY_);
}

float  snrm2_(const INT *N, const float *X, const INT *incX);
float  cblas_snrm2(const int N, const float *X, const int incX) {
  INT N_ = N, incX_ = incX;
  return snrm2_(&N_, X, &incX_);
}
double dnrm2_(const INT *N, const double *X, const INT *incX);
double cblas_dnrm2(const int N, const double *X, const int incX) {
  INT N_ = N, incX_ = incX;
  return dnrm2_(&N_, X, &incX_);
}

float  sasum_(const INT *N, const float *X, const INT *incX);
float  cblas_sasum(const int N, const float *X, const int incX) {
  INT N_ = N, incX_ = incX;
  return sasum_(&N_, X, &incX_);
}
double dasum_(const INT *N, const double *X, const INT *incX);
double cblas_dasum(const int N, const double *X, const int incX) {
  INT N_ = N, incX_ = incX;
  return dasum_(&N_, X, &incX_);
}

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */
void saxpy_(const INT *N, const float *alpha, const float *X, const INT *incX,
            float *Y, const INT *incY);
void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY) {
  INT N_ = N, incX_ = incX, incY_ = incY;
  saxpy_(&N_, &alpha, X, &incX_, Y, &incY_);
}
void daxpy_(const INT *N, const double *alpha, const double *X, const INT *incX,
            double *Y, const INT *incY);
void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY) {
  INT N_ = N, incX_ = incX, incY_ = incY;
  daxpy_(&N_, &alpha, X, &incX_, Y, &incY_);
}

void sscal_(const INT *N, const float *alpha, float *X, const INT *incX);
void cblas_sscal(const int N, const float alpha, float *X, const int incX) {
  INT N_ = N, incX_ = incX;
  sscal_(&N_, &alpha, X, &incX_);
}
void dscal_(const INT *N, const double *alpha, double *X, const INT *incX);
void cblas_dscal(const int N, const double alpha, double *X, const int incX) {
  INT N_ = N, incX_ = incX;
  dscal_(&N_, &alpha, X, &incX_);
}

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

void sgemv_(const char *trans, const INT *M, const INT *N, const float *alpha,
            const float *A, const INT *lda, const float* X, const INT *incX,
            const float *beta, float *Y, const INT *incY);
void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY) {
   char TA;
   INT M_ = M, N_ = N, lda_ = lda, incX_ = incX, incY_ = incY;
   if (order == CblasColMajor) {
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else TA = 'C';
      sgemv_(&TA, &M_, &N_, &alpha, A, &lda_, X, &incX_, &beta, Y, &incY_);
   } else {
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else TA = 'N';
      sgemv_(&TA, &N_, &M_, &alpha, A, &lda_, X, &incX_, &beta, Y, &incY_);
   }
}
void dgemv_(const char *trans, const INT *M, const INT *N, const double *alpha,
            const double *A, const INT *lda, const double* X, const INT *incX,
            const double *beta, double *Y, const INT *incY);
void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY) {
   char TA;
   INT M_ = M, N_ = N, lda_ = lda, incX_ = incX, incY_ = incY;
   if (order == CblasColMajor) {
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else TA = 'C';

      dgemv_(&TA, &M_, &N_, &alpha, A, &lda_, X, &incX_, &beta, Y, &incY_);
   } else {
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else TA = 'N';

      dgemv_(&TA, &N_, &M_, &alpha, A, &lda_, X, &incX_, &beta, Y, &incY_);
   }
}

void strsv_(const char *uplo, const char *trans, const char *diag, const INT *N,
            const float *A, const INT *lda, float *X, const INT *incX);
void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, float *X,
                 const int incX) {
  char TA, UL, DI;
  INT N_ = N, lda_ = lda, incX_ = incX;
  if (order == CblasColMajor) {
    if (Uplo == CblasUpper) UL = 'U';
    else UL = 'L';

    if (TransA == CblasNoTrans) TA = 'N';
    else if (TransA == CblasTrans) TA = 'T';
    else TA = 'C';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    strsv_(&UL, &TA, &DI, &N_, A, &lda_, X, &incX_);
  } else {
    if (Uplo == CblasUpper) UL = 'L';
    else UL = 'U';

    if (TransA == CblasNoTrans) TA = 'T';
    else if (TransA == CblasTrans) TA = 'N';
    else TA = 'N';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    strsv_(&UL, &TA, &DI, &N_, A, &lda_, X, &incX_);
  }
}

void dtrsv_(const char *uplo, const char *TransA, const char *diag,
            const INT *N, const double *A, const INT *lda, double *X,
            const INT *incX);
void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, double *X,
                 const int incX) {
  char TA, UL, DI;
  INT N_ = N, lda_ = lda, incX_ = incX;
  if (order == CblasColMajor) {
    if (Uplo == CblasUpper) UL = 'U';
    else UL = 'L';

    if (TransA == CblasNoTrans) TA = 'N';
    else if (TransA == CblasTrans) TA = 'T';
    else TA = 'C';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    dtrsv_(&UL, &TA, &DI, &N_, A, &lda_, X, &incX_);
  } else {
    if (Uplo == CblasUpper) UL = 'L';
    else UL = 'U';

    if (TransA == CblasNoTrans) TA = 'T';
    else if (TransA == CblasTrans) TA = 'N';
    else TA = 'N';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    dtrsv_(&UL, &TA, &DI, &N_, A, &lda_, X, &incX_);
  }
}

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

void sgemm_(const char *TransA, const char *TransB, const INT *M, const INT *N,
            const INT *K, const float *alpha, const float *A, const INT *lda,
            const float *B, const INT *ldb, const float *beta, float *C,
            const INT *ldc);
void cblas_sgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc) {
  char TA, TB;   
  INT M_ = M, N_ = N, K_ = K, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
  if (Order == CblasColMajor) {
    if (TransA == CblasTrans) TA = 'T';
    else if (TransA == CblasConjTrans) TA = 'C';
    else TA = 'N';

    if (TransB == CblasTrans) TB = 'T';
    else if (TransB == CblasConjTrans) TB = 'C';
    else TB = 'N';

    sgemm_(&TA, &TB, &M_, &N_, &K_, &alpha, A, &lda_, B, &ldb_, &beta, C,
         &ldc_);
  } else {
    if (TransA == CblasTrans) TB = 'T';
    else if (TransA == CblasConjTrans) TB = 'C';
    else TB = 'N';

    if (TransB == CblasTrans) TA = 'T';
    else if (TransB == CblasConjTrans) TA = 'C';
    else TA = 'N';

    sgemm_(&TA, &TB, &N_, &M_, &K_, &alpha, B, &ldb_, A, &lda_, &beta, C,
        &ldc_);
  }
}
void dgemm_(const char *TransA, const char *TransB, const INT *M, const INT *N,
            const INT *K, const double *alpha, const double *A, const INT *lda,
            const double *B, const INT *ldb, const double *beta, double *C,
            const INT *ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) {
  char TA, TB;   
  INT M_ = M, N_ = N, K_ = K, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
  if (Order == CblasColMajor) {
    if (TransA == CblasTrans) TA = 'T';
    else if (TransA == CblasConjTrans) TA = 'C';
    else TA = 'N';

    if (TransB == CblasTrans) TB = 'T';
    else if (TransB == CblasConjTrans) TB = 'C';
    else TB = 'N';

    dgemm_(&TA, &TB, &M_, &N_, &K_, &alpha, A, &lda_, B, &ldb_, &beta, C,
        &ldc_);
  } else {
    if (TransA == CblasTrans) TB = 'T';
    else if (TransA == CblasConjTrans) TB = 'C';
    else TB = 'N';

    if (TransB == CblasTrans) TA = 'T';
    else if (TransB == CblasConjTrans) TA = 'C';
    else TA = 'N';

    dgemm_(&TA, &TB, &N_, &M_, &K_, &alpha, B, &ldb_, A, &lda_, &beta, C,
        &ldc_);
  }
}


void ssyrk_(const char *Uplo, const char *Trans, const INT *N, const INT *K,
            const float *alpha, const float *A, const INT *lda,
            const float *beta, float *C, const INT *ldc);
void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const float *A, const int lda,
                 const float beta, float *C, const int ldc) {
  char UL, TR;   
  INT N_ = N, K_ = K, lda_ = lda, ldc_ = ldc;
  if (Order == CblasColMajor) {
    if (Uplo == CblasUpper) UL = 'U';
    else UL = 'L';

    if (Trans == CblasTrans) TR ='T';
    else if (Trans == CblasConjTrans) TR = 'C';
    else TR = 'N';

    ssyrk_(&UL, &TR, &N_, &K_, &alpha, A, &lda_, &beta, C, &ldc_);
  } else {
    if (Uplo == CblasUpper) UL = 'L';
    else UL = 'U';

    if (Trans == CblasTrans) TR = 'N';
    else if (Trans == CblasConjTrans) TR = 'N';
    else TR = 'T';

    ssyrk_(&UL, &TR, &N_, &K_, &alpha, A, &lda_, &beta, C, &ldc_);
  } 
}

void dsyrk_(const char *Uplo, const char *Trans, const INT *N, const INT *K,
            const double *alpha, const double *A, const INT *lda,
            const double *beta, double *C, const INT *ldc);
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double beta, double *C, const int ldc) {
  char UL, TR;   
  INT N_ = N, K_ = K, lda_ = lda, ldc_ = ldc;
  if (Order == CblasColMajor) {
    if (Uplo == CblasUpper) UL = 'U';
    else UL = 'L';

    if (Trans == CblasTrans) TR ='T';
    else if (Trans == CblasConjTrans) TR = 'C';
    else TR = 'N';

    dsyrk_(&UL, &TR, &N_, &K_, &alpha, A, &lda_, &beta, C, &ldc_);
  } else {
    if (Uplo == CblasUpper) UL = 'L';
    else UL = 'U';

    if (Trans == CblasTrans) TR = 'N';
    else if (Trans == CblasConjTrans) TR = 'N';
    else TR = 'T';

    dsyrk_(&UL, &TR, &N_, &K_, &alpha, A, &lda_, &beta, C, &ldc_);
  } 
}


void strsm_(const char *Side, const char *Uplo, const char *TransA,
            const char *Diag, const INT *M, const INT *N, const float *alpha,
            const float *A, const INT *lda, float *B, const INT *ldb);
void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb) {
  char UL, TA, SD, DI;
  INT M_ = M, N_ = N, lda_ = lda, ldb_ = ldb;
  if (Order == CblasColMajor) {
    if (Side == CblasRight) SD = 'R';
    else SD = 'L';

    if (Uplo == CblasUpper) UL = 'U';
    else UL = 'L';

    if (TransA == CblasTrans) TA = 'T';
    else if (TransA == CblasConjTrans) TA = 'C';
    else  TA = 'N';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    strsm_(&SD, &UL, &TA, &DI, &M_, &N_, &alpha, A, &lda_, B, &ldb_);
  } else {
    if (Side == CblasRight) SD = 'L';
    else SD = 'R';

    if (Uplo == CblasUpper) UL = 'L';
    else UL = 'U';

    if (TransA == CblasTrans) TA = 'T';
    else if (TransA == CblasConjTrans) TA = 'C';
    else TA = 'N';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    strsm_(&SD, &UL, &TA, &DI, &N_, &M_, &alpha, A, &lda_, B, &ldb_);
  } 
}
void dtrsm_(const char *Side, const char *Uplo, const char *TransA,
            const char *Diag, const INT *M, const INT *N, const double *alpha,
            const double *A, const INT *lda, double *B, const INT *ldb);
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb) {
  char UL, TA, SD, DI;
  INT M_ = M, N_ = N, lda_ = lda, ldb_ = ldb;
  if (Order == CblasColMajor) {
    if (Side == CblasRight) SD = 'R';
    else SD = 'L';

    if (Uplo == CblasUpper) UL = 'U';
    else UL = 'L';

    if (TransA == CblasTrans) TA = 'T';
    else if (TransA == CblasConjTrans) TA = 'C';
    else  TA = 'N';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    dtrsm_(&SD, &UL, &TA, &DI, &M_, &N_, &alpha, A, &lda_, B, &ldb_);
  } else {
    if (Side == CblasRight) SD = 'L';
    else SD = 'R';

    if (Uplo == CblasUpper) UL = 'L';
    else UL = 'U';

    if (TransA == CblasTrans) TA = 'T';
    else if (TransA == CblasConjTrans) TA = 'C';
    else TA = 'N';

    if (Diag == CblasUnit) DI = 'U';
    else DI = 'N';

    dtrsm_(&SD, &UL, &TA, &DI, &N_, &M_, &alpha, A, &lda_, B, &ldb_);
  } 
}

#ifdef __cplusplus
}
#endif

