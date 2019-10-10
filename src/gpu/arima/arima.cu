#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "arima.h"
#include "cuda_utils2.h"
#include "utils.cuh"

namespace h2o4gpu {

#define BLOCK_SIZE 32

inline cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle,
                                                    int m, int n, float *A,
                                                    int lda, int *lwork) {
  return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
}

inline cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle,
                                                    int m, int n, double *A,
                                                    int lda, int *lwork) {
  return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
}

inline cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m,
                                         int n, float *A, int lda, float *TAU,
                                         float *Workspace, int Lwork,
                                         int *devInfo) {
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

inline cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m,
                                         int n, double *A, int lda, double *TAU,
                                         double *Workspace, int Lwork,
                                         int *devInfo) {
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

inline cusolverStatus_t cusolverDnTormqr(cusolverDnHandle_t handle,
                                         cublasSideMode_t side,
                                         cublasOperation_t trans, int m, int n,
                                         int k, const float *A, int lda,
                                         const float *tau, float *C, int ldc,
                                         float *work, int lwork, int *devInfo) {
  return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                          work, lwork, devInfo);
}

inline cusolverStatus_t cusolverDnTormqr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const double *A, int lda, const double *tau, double *C,
    int ldc, double *work, int lwork, int *devInfo) {
  return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                          work, lwork, devInfo);
}

inline cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const float *alpha, const float *A, int lda,
                                  float *B, int ldb) {
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

inline cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const double *alpha, const double *A, int lda,
                                  double *B, int ldb) {
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

template <class T>
__global__ void copy_kernel(const T *__restrict d_in, T *__restrict d_out,
                            const int M, const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < N) && (j < N)) d_out[j * N + i] = d_in[j * M + i];
}

template <class T>
__global__ void ts_data_to_matrix_kernel(const T *__restrict data, T *X,
                                         const int ldx, const int n) {
  // row
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // col, time axis
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO: optimize with shared memory(read data only once)
  if (i < n && i < ldx) {
    X[j * ldx + i] = data[j + i];
  }
}

template <class T>
__global__ void update_residual_kernel(T *residual, const T *data, const T *phi,
                                       const int p, const T *last_residual,
                                       const T *theta, const int q,
                                       const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: optimize with shared memory(read data only once)
  // at least cache phi and theta
  if (i < n) {
    T AR_prediction = 0;
    for (int j = 0; j < p; ++j) {
      AR_prediction += data[i + j + 1] * phi[j];
    }

    T MA_prediction = 0;
    for (int j = 0; j < q; ++j) {
      MA_prediction += last_residual[i + j + 1] * theta[j];
    }

    residual[i] = data[i] - AR_prediction - MA_prediction;
  }
}

LeastSquaresSolver::LeastSquaresSolver(int rows, int cols)
    : rows(rows), cols(cols) {
  safe_cusolver(cusolverDnCreate(&this->solver_handle));
  safe_cublas(cublasCreate(&this->cublas_handle));
}

LeastSquaresSolver::~LeastSquaresSolver() {
  safe_cusolver(cusolverDnDestroy(this->solver_handle));
  safe_cublas(cublasDestroy(this->cublas_handle));
}

template <typename T>
void LeastSquaresSolver::Solve(T *A, T *B) {
  int work_size = 0;
  int *devInfo;

  OK(cudaMalloc(&devInfo, sizeof(int)));

  /**********************************/
  /* COMPUTING THE QR DECOMPOSITION */
  /**********************************/

  // --- CUDA QR GEQRF preliminary operations
  T *d_TAU;
  OK(cudaMalloc(&d_TAU, min(rows, cols) * sizeof(T)));
  safe_cusolver(cusolverDnTgeqrf_bufferSize(solver_handle, rows, cols, A, rows,
                                            &work_size));
  T *work;
  OK(cudaMalloc(&work, work_size * sizeof(T)));

  // CUDA GEQRF execution: The matrix R is overwritten in upper triangular
  // part of A, including diagonal
  // elements. The matrix Q is not formed explicitly, instead, a sequence of
  // householder vectors are stored in lower triangular part of A.

  safe_cusolver(cusolverDnTgeqrf(solver_handle, rows, cols, A, rows, d_TAU,
                                 work, work_size, devInfo));
  int devInfo_h = 0;
  OK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  assert(devInfo_h == 0);

  /*****************************/
  /* SOLVING THE LINEAR SYSTEM */
  /*****************************/

  // --- CUDA ORMQR execution: Computes the multiplication Q^T * B and stores it
  // in B
  safe_cusolver(cusolverDnTormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                 rows, 1, min(rows, cols), A, rows, d_TAU, B,
                                 rows, work, work_size, devInfo));

  OK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  OK(cudaFree(d_TAU));
  OK(cudaFree(devInfo));
  OK(cudaFree(work));
  assert(devInfo_h == 0);

  // --- Solving an upper triangular linear system R * x = Q^T * B

  const T alpha = 1.;
  safe_cublas(cublasTtrsm(
      cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, cols, 1, &alpha, A, rows, B, cols));

  OK(cudaDeviceSynchronize());
}

template <class T>
__global__ void differencing(T *out, const T *in, const int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (i < n - 1)
      out[i] = in[i] - in[i + 1];
    else
      out[n - 1] = NAN;
  }
}

template <class T>
__global__ void undifferencing(T *out, const T *in, const int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (i > 0)
      out[i] = in[i] - in[i - 1];
    else
      out[0] = NAN;
  }
}

template <class T>
ARIMAModel<T>::ARIMAModel(int p, int d, int q, int length)
    : p(p), d(d), q(q), length(length) {
  assert(q >= 0);
  assert(d >= 0);
  assert(p >= 0);
  assert(length > 0);
  assert(p > 0 || q > 0);
  if (q > 0) {
    OK(cudaMallocHost(&this->theta, sizeof(T) * q));
    memset(this->theta, 0, sizeof(T) * q);
    OK(cudaMalloc(&this->d_theta, sizeof(T) * q));
    OK(cudaMemset(this->d_theta, 0, sizeof(T) * q));
  }

  if (p > 0) {
    OK(cudaMallocHost(&this->phi, sizeof(T) * p));
    memset(this->phi, 0, sizeof(T) * p);
    OK(cudaMalloc(&this->d_phi, sizeof(T) * p));
    OK(cudaMemset(this->d_phi, 0, sizeof(T) * p));
  }

  OK(cudaMalloc(&this->d_buffer, sizeof(T) * this->DifferencedLength()));
};

template <class T>
ARIMAModel<T>::~ARIMAModel() {
  if (q > 0) {
    OK(cudaFreeHost(this->theta));
    OK(cudaFree(this->d_theta));
  }
  if (p > 0) {
    OK(cudaFreeHost(this->phi));
    OK(cudaFree(this->d_phi));
  }
  OK(cudaFree(this->d_buffer));
}

template <class T>
void ARIMAModel<T>::Difference(T *out, const T *in, int length) {
  int block_size, grid_size;
  compute1DInvokeConfig(length, &grid_size, &block_size, differencing<T>);
  differencing<T><<<grid_size, block_size>>>(out, in, length);
}

template <class T>
void ARIMAModel<T>::AsMatrix(const T *ts_data, T *A, int depth, int lda,
                             int length) {
  if (depth > 0) {
    int n = length - depth + 1;
    dim3 grid_size(DIVUP(n, BLOCK_SIZE), depth);
    dim3 block_size(BLOCK_SIZE, 1);
    ts_data_to_matrix_kernel<T><<<grid_size, block_size>>>(ts_data, A, lda, n);
  }
}

template <class T>
void ARIMAModel<T>::AsMatrix(const T *ts_a, const T *ts_b, T *A, int a_depth,
                             int b_depth, int lda, int length) {
  ARIMAModel<T>::AsMatrix(ts_a, A, a_depth, lda, length);
  ARIMAModel<T>::AsMatrix(ts_b, A + a_depth * lda, b_depth, lda, length);
}

template <class T>
void ARIMAModel<T>::Apply(T *residual, const T *ts_data, const T *phi,
                          const int p, const T *last_residual, const T *theta,
                          const int q, int length) {
  int block_size, grid_size;
  compute1DInvokeConfig(length - max(p, q), &grid_size, &block_size,
                        update_residual_kernel<T>);

  update_residual_kernel<T><<<grid_size, block_size>>>(
      residual, ts_data, phi, p, last_residual, theta, q, length - max(p, q));
}

template <class T>
void ARIMAModel<T>::Fit(const T *data, const int maxiter) {
  OK(cudaMalloc(&this->d_data_src, sizeof(T) * this->length));
  OK(cudaMalloc(&this->d_data_differenced, sizeof(T) * this->length));
  OK(cudaMalloc(&this->d_last_residual, sizeof(T) * this->length));
  OK(cudaMemset(this->d_last_residual, 0, sizeof(T) * this->length));

  OK(cudaMemcpy(this->d_data_src, data, sizeof(T) * this->length,
                cudaMemcpyHostToDevice));

  for (auto i = 0; i < this->d; ++i) {
    this->Difference(this->d_data_differenced, this->d_data_src, this->length);
    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());
    std::swap(this->d_data_src, this->d_data_differenced);
  }

  OK(cudaMemcpy(this->d_data_differenced, this->d_data_src,
                this->length * sizeof(T), cudaMemcpyDeviceToDevice));

  OK(cudaMemcpy(this->d_buffer, this->d_data_src,
                sizeof(T) * this->DifferencedLength(),
                cudaMemcpyDeviceToDevice));

  if (this->p > 0) {
    T *X;
    OK(cudaMalloc(&X, sizeof(T) * this->ARLength() * this->p));
    this->AsMatrix(this->d_data_src + 1, X, this->p, this->ARLength(),
                   this->DifferencedLength());

    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());

    LeastSquaresSolver solver(this->ARLength(), this->p);
    solver.Solve(X, this->d_data_differenced);

    OK(cudaMemcpy(this->d_phi, this->d_data_differenced, sizeof(T) * this->p,
                  cudaMemcpyDeviceToDevice));

    OK(cudaMemcpy(this->phi, this->d_data_differenced, sizeof(T) * this->p,
                  cudaMemcpyDeviceToHost));
    OK(cudaFree(X));
  }

  if (this->q > 0) {
    T *X;
    int rows = min(this->MALength() == 0 ? this->ARLength() : this->MALength(),
                   this->ARLength() == 0 ? this->MALength() : this->ARLength());

    OK(cudaMalloc(&X, sizeof(T) * rows * (this->q + this->p)));

    for (int i = 0; i < maxiter; ++i) {
      OK(cudaMemcpy(this->d_data_src, this->d_buffer,
                    sizeof(T) * this->DifferencedLength(),
                    cudaMemcpyDeviceToDevice));

      this->Apply(this->d_data_differenced, this->d_data_src, this->d_phi,
                  this->p, this->d_last_residual, this->d_theta, this->q,
                  this->DifferencedLength());
      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());

      this->AsMatrix(this->d_data_src + 1, this->d_data_differenced + 1, X,
                     this->p, this->q, rows, this->DifferencedLength());

      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());
      LeastSquaresSolver solver(rows, this->q + this->p);

      solver.Solve(X, this->d_data_src);

      if (this->p > 0) {
        OK(cudaMemcpy(this->d_phi, this->d_data_src, sizeof(T) * this->p,
                      cudaMemcpyDeviceToDevice));
      }
      OK(cudaDeviceSynchronize());
      OK(cudaGetLastError());

      OK(cudaMemcpy(this->d_theta, this->d_data_src + this->p,
                    sizeof(T) * this->q, cudaMemcpyDeviceToDevice));

      std::swap(this->d_last_residual, this->d_data_differenced);
    }

    OK(cudaMemcpy(this->phi, this->d_data_src, sizeof(T) * this->p,
                  cudaMemcpyDeviceToHost));

    OK(cudaMemcpy(this->theta, this->d_data_src + this->p, sizeof(T) * this->q,
                  cudaMemcpyDeviceToHost));

    OK(cudaFree(X));
  }
  OK(cudaFree(this->d_data_src));
  OK(cudaFree(this->d_data_differenced));
  OK(cudaFree(this->d_last_residual));
}

template class ARIMAModel<float>;
template class ARIMAModel<double>;
}  // namespace h2o4gpu

template <typename T>
void arima_fit(const int p, const int d, const int q, const T *ts_data,
               const int length, T *theta, T *phi, const int maxiter) {
  h2o4gpu::ARIMAModel<T> model(p, d, q, length);
  model.Fit(ts_data, maxiter);
  if (p > 0) std::memcpy(phi, model.Phi(), sizeof(T) * p);
  if (q > 0) std::memcpy(theta, model.Theta(), sizeof(T) * q);
}

void arima_fit_float(const int p, const int d, const int q,
                     const float *ts_data, const int length, float *theta,
                     float *phi, const int maxiter) {
  arima_fit<float>(p, d, q, ts_data, length, theta, phi, maxiter);
}

void arima_fit_double(const int p, const int d, const int q,
                      const double *ts_data, const int length, double *theta,
                      double *phi, const int maxiter) {
  arima_fit<double>(p, d, q, ts_data, length, theta, phi, maxiter);
}