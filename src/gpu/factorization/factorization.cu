#include "als.h"
#include "cuda_utils.h"
#include "solver/factorization.h"

template <class T>
int make_factorization_data(
    const int m, const int n, const int f, const long nnz, const long nnz_test,
    const int *csrRowIndexHostPtr, const int *csrColIndexHostPtr,
    const T *csrValHostPtr, const int *cscRowIndexHostPtr,
    const int *cscColIndexHostPtr, const T *cscValHostPtr,
    const int *cooRowIndexHostPtr, const int *cooColIndexHostPtr,
    const T *cooValHostPtr, T *thetaTHost, T *XTHost,
    const int *cooRowIndexTestHostPtr, const int *cooColIndexTestHostPtr,
    const T *cooValTestHostPtr,

    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr, T **csrValDevicePtr,
    int **cscRowIndexDevicePtr, int **cscColIndexDevicePtr, T **cscValDevicePtr,
    int **cooRowIndexDevicePtr, int **cooColIndexDevicePtr, T **cooValDevicePtr,
    T **thetaTDevice, T **XTDevice, int **cooRowIndexTestDevicePtr,
    int **cooColIndexTestDevicePtr, T **cooValDeviceTestPtr) {

  CUDACHECK(cudaMalloc((void **)cooRowIndexDevicePtr,
                       nnz * sizeof(**cooRowIndexDevicePtr)));
  CUDACHECK(cudaMemcpy(*cooRowIndexDevicePtr, cooRowIndexHostPtr,
                       (size_t)(nnz * sizeof(**cooRowIndexDevicePtr)),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMalloc((void **)cooColIndexDevicePtr,
                       nnz * sizeof(**cooColIndexDevicePtr)));
  CUDACHECK(cudaMemcpy(*cooColIndexDevicePtr, cooColIndexHostPtr,
                       (size_t)(nnz * sizeof(**cooColIndexDevicePtr)),
                       cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMalloc((void **)cooValDevicePtr, nnz * sizeof(**cooValDevicePtr)));
  CUDACHECK(cudaMemcpy(*cooValDevicePtr, cooValHostPtr,
                       (size_t)(nnz * sizeof(**cooValDevicePtr)),
                       cudaMemcpyHostToDevice));

  CUDACHECK(cudaMalloc((void **)cscRowIndexDevicePtr,
                       nnz * sizeof(**cscRowIndexDevicePtr)));
  CUDACHECK(cudaMalloc((void **)cscColIndexDevicePtr,
                       (n + 1) * sizeof(**cscColIndexDevicePtr)));
  CUDACHECK(
      cudaMalloc((void **)cscValDevicePtr, nnz * sizeof(**cscValDevicePtr)));
  // dimension: F*N
  CUDACHECK(cudaMalloc((void **)thetaTDevice, f * n * sizeof(**thetaTDevice)));
  // dimension: M*F
  CUDACHECK(cudaMalloc((void **)XTDevice, f * m * sizeof(**XTDevice)));
  CUDACHECK(cudaMemcpy(*cscRowIndexDevicePtr, cscRowIndexHostPtr,
                       (size_t)nnz * sizeof(**cscRowIndexDevicePtr),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(*cscColIndexDevicePtr, cscColIndexHostPtr,
                       (size_t)(n + 1) * sizeof(**cscColIndexDevicePtr),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(*cscValDevicePtr, cscValHostPtr,
                       (size_t)(nnz * sizeof(**cscValDevicePtr)),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(*thetaTDevice, thetaTHost,
                       (size_t)(n * f * sizeof(**thetaTDevice)),
                       cudaMemcpyHostToDevice));
  // CG needs XT
  CUDACHECK(cudaMemcpy(*XTDevice, XTHost, (size_t)(m * f * sizeof(**XTDevice)),
                       cudaMemcpyHostToDevice));

  CUDACHECK(cudaMalloc((void **)csrRowIndexDevicePtr,
                       (m + 1) * sizeof(**csrRowIndexDevicePtr)));
  CUDACHECK(cudaMalloc((void **)csrColIndexDevicePtr,
                       nnz * sizeof(**csrColIndexDevicePtr)));
  CUDACHECK(
      cudaMalloc((void **)csrValDevicePtr, nnz * sizeof(**csrValDevicePtr)));
  CUDACHECK(cudaMemcpy(*csrRowIndexDevicePtr, csrRowIndexHostPtr,
                       (size_t)((m + 1) * sizeof(**csrRowIndexDevicePtr)),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(*csrColIndexDevicePtr, csrColIndexHostPtr,
                       (size_t)(nnz * sizeof(**csrColIndexDevicePtr)),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(*csrValDevicePtr, csrValHostPtr,
                       (size_t)(nnz * sizeof(**csrValDevicePtr)),
                       cudaMemcpyHostToDevice));

  if (cooColIndexTestHostPtr && cooRowIndexTestHostPtr && cooValTestHostPtr) {
    CUDACHECK(cudaMalloc((void **)cooRowIndexTestDevicePtr,
                         nnz_test * sizeof(**cooRowIndexTestDevicePtr)));
    CUDACHECK(cudaMalloc((void **)cooColIndexTestDevicePtr,
                         nnz_test * sizeof(**cooColIndexTestDevicePtr)));
    CUDACHECK(cudaMalloc((void **)cooValDeviceTestPtr,
                         nnz_test * sizeof(**cooValDeviceTestPtr)));
    CUDACHECK(
        cudaMemcpy(*cooRowIndexTestDevicePtr, cooRowIndexTestHostPtr,
                   (size_t)(nnz_test * sizeof(**cooRowIndexTestDevicePtr)),
                   cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(*cooColIndexTestDevicePtr, cooColIndexTestHostPtr,
                   (size_t)(nnz_test * sizeof(**cooColIndexTestDevicePtr)),
                   cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(*cooValDeviceTestPtr, cooValTestHostPtr,
                         (size_t)(nnz_test * sizeof(**cooValDeviceTestPtr)),
                         cudaMemcpyHostToDevice));
  }
  return 0;
}

template <typename T>
T factorization_score(const int m, const int n, const int f, const long nnz,
                      const T lambda, T **thetaTDevice, T **XTDevice,
                      int **cooRowIndexDevicePtr, int **cooColIndexDevicePtr,
                      T **cooValDevicePtr) {
  ALSFactorization<T> factorization(m, n, f, lambda, *thetaTDevice, *XTDevice);
  return factorization.Score(*cooRowIndexDevicePtr, *cooColIndexDevicePtr,
                             *cooValDevicePtr, nnz);
}

template <typename T>
int run_factorization_step(
    const int m, const int n, const int f, const long nnz, const T lambda,
    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr, T **csrValDevicePtr,
    int **cscRowIndexDevicePtr, int **cscColIndexDevicePtr, T **cscValDevicePtr,
    T **thetaTDevice, T **XTDevice, const int X_BATCH, const int THETA_BATCH) {
  ALSFactorization<T> factorization(m, n, f, lambda, *thetaTDevice, *XTDevice);
  factorization.Iter(*csrRowIndexDevicePtr, *csrColIndexDevicePtr,
                     *csrValDevicePtr, *cscRowIndexDevicePtr,
                     *cscColIndexDevicePtr, *cscValDevicePtr, nnz, X_BATCH,
                     THETA_BATCH);
  return 0;
}

float factorization_score_float(const int m, const int n, const int f,
                                const long nnz, const float lambda,
                                float **thetaTDevice, float **XTDevice,
                                int **cooRowIndexDevicePtr,
                                int **cooColIndexDevicePtr,
                                float **cooValDevicePtr) {
  return factorization_score<float>(m, n, f, nnz, lambda, thetaTDevice,
                                    XTDevice, cooRowIndexDevicePtr,
                                    cooColIndexDevicePtr, cooValDevicePtr);
}

double factorization_score_double(const int m, const int n, const int f,
                                  const long nnz, const float lambda,
                                  double **thetaTDevice, double **XTDevice,
                                  int **cooRowIndexDevicePtr,
                                  int **cooColIndexDevicePtr,
                                  double **cooValDevicePtr) {
  return 0.0;
}

int run_factorization_step_double(
    const int m, const int n, const int f, const long nnz, const double lambda,
    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    double **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, double **cscValDevicePtr, double **thetaTDevice,
    double **XTDevice, const int X_BATCH, const int THETA_BATCH) {
  return 1;
  // TODO: implement
  // return run_factorization_step<double>(m, n, f, nnz, nnz_test,
  //     csrRowIndexDevicePtr, csrColIndexDevicePtr, csrValDevicePtr,
  //     cscRowIndexDevicePtr, cscColIndexDevicePtr, cscValDevicePtr,
  //     cooRowIndexDevicePtr, thetaTDevice, XTDevice, cooRowIndexTestDevicePtr,
  //     cooColIndexTestDevicePtr, cooValTestDevicePtr);
}

int run_factorization_step_float(
    const int m, const int n, const int f, const long nnz, const float lambda,
    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    float **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, float **cscValDevicePtr, float **thetaTDevice,
    float **XTDevice, const int X_BATCH, const int THETA_BATCH) {
  return run_factorization_step<float>(
      m, n, f, nnz, lambda, csrRowIndexDevicePtr, csrColIndexDevicePtr,
      csrValDevicePtr, cscRowIndexDevicePtr, cscColIndexDevicePtr,
      cscValDevicePtr, thetaTDevice, XTDevice, X_BATCH, THETA_BATCH);
}

int make_factorization_data_double(
    const int m, const int n, const int f, const long nnz, const long nnz_test,
    const int *csrRowIndexHostPtr, const int *csrColIndexHostPtr,
    const double *csrValHostPtr, const int *cscRowIndexHostPtr,
    const int *cscColIndexHostPtr, const double *cscValHostPtr,
    const int *cooRowIndexHostPtr, const int *cooColIndexHostPtr,
    const double *cooValHostPtr, double *thetaTHost, double *XTHost,
    const int *cooRowIndexTestHostPtr, const int *cooColIndexTestHostPtr,
    const double *cooValTestHostPtr,

    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    double **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, double **cscValDevicePtr,
    int **cooRowIndexDevicePtr, int **cooColIndexDevicePtr,
    double **cooValDevicePtr, double **thetaTDevice, double **XTDevice,
    int **cooRowIndexTestDevicePtr, int **cooColIndexTestDevicePtr,
    double **cooValTestDevicePtr) {
  return make_factorization_data<double>(
      m, n, f, nnz, nnz_test, csrRowIndexHostPtr, csrColIndexHostPtr,
      csrValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
      cooRowIndexHostPtr, cooColIndexHostPtr, cooValHostPtr, thetaTHost, XTHost,
      cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValTestHostPtr,

      csrRowIndexDevicePtr, csrColIndexDevicePtr, csrValDevicePtr,
      cscRowIndexDevicePtr, cscColIndexDevicePtr, cscValDevicePtr,
      cooRowIndexDevicePtr, cooColIndexDevicePtr, cooValDevicePtr, thetaTDevice,
      XTDevice, cooRowIndexTestDevicePtr, cooColIndexTestDevicePtr,
      cooValTestDevicePtr);
}

int make_factorization_data_float(
    const int m, const int n, const int f, const long nnz, const long nnz_test,
    const int *csrRowIndexHostPtr, const int *csrColIndexHostPtr,
    const float *csrValHostPtr, const int *cscRowIndexHostPtr,
    const int *cscColIndexHostPtr, const float *cscValHostPtr,
    const int *cooRowIndexHostPtr, const int *cooColIndexHostPtr,
    const float *cooValHostPtr, float *thetaTHost, float *XTHost,
    const int *cooRowIndexTestHostPtr, const int *cooColIndexTestHostPtr,
    const float *cooValTestHostPtr,

    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    float **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, float **cscValDevicePtr,
    int **cooRowIndexDevicePtr, int **cooColIndexDevicePtr,
    float **cooValDevicePtr, float **thetaTDevice, float **XTDevice,
    int **cooRowIndexTestDevicePtr, int **cooColIndexTestDevicePtr,
    float **cooValTestDevicePtr) {
  return make_factorization_data<float>(
      m, n, f, nnz, nnz_test, csrRowIndexHostPtr, csrColIndexHostPtr,
      csrValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
      cooRowIndexHostPtr, cooColIndexHostPtr, cooValHostPtr, thetaTHost, XTHost,
      cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValTestHostPtr,

      csrRowIndexDevicePtr, csrColIndexDevicePtr, csrValDevicePtr,
      cscRowIndexDevicePtr, cscColIndexDevicePtr, cscValDevicePtr,
      cooRowIndexDevicePtr, cooColIndexDevicePtr, cooValDevicePtr, thetaTDevice,
      XTDevice, cooRowIndexTestDevicePtr, cooColIndexTestDevicePtr,
      cooValTestDevicePtr);
}