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
__global__ void update_AR_kernel(const T *data, T *residual, const T *phi,
                                 const int p, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: optimize with shared memory(read data only once)
  if (i < n) {
    T AR_prediction = data[i];
    for (int j = 0; j < p; ++j) {
      AR_prediction -= data[i + j + 1] * phi[j];
    }
    residual[i] = AR_prediction;
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

void LeastSquaresSolver::Solve(float *A, float *B) {
  int work_size = 0;
  int *devInfo;

  OK(cudaMalloc(&devInfo, sizeof(int)));

  /**********************************/
  /* COMPUTING THE QR DECOMPOSITION */
  /**********************************/

  // --- CUDA QR GEQRF preliminary operations
  float *d_TAU;
  OK(cudaMalloc(&d_TAU, min(rows, cols) * sizeof(float)));
  safe_cusolver(cusolverDnSgeqrf_bufferSize(solver_handle, rows, cols, A, rows,
                                            &work_size));
  float *work;
  OK(cudaMalloc(&work, work_size * sizeof(float)));

  // CUDA GEQRF execution: The matrix R is overwritten in upper triangular
  // part of A, including diagonal
  // elements. The matrix Q is not formed explicitly, instead, a sequence of
  // householder vectors are stored in lower triangular part of A.

  safe_cusolver(cusolverDnSgeqrf(solver_handle, rows, cols, A, rows, d_TAU,
                                 work, work_size, devInfo));
  int devInfo_h = 0;
  OK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  assert(devInfo_h == 0);

  /*****************************/
  /* SOLVING THE LINEAR SYSTEM */
  /*****************************/

  // --- CUDA ORMQR execution: Computes the multiplication Q^T * C and stores it
  // in d_C
  safe_cusolver(cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                 rows, 1, min(rows, cols), A, rows, d_TAU, B,
                                 rows, work, work_size, devInfo));

  OK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  OK(cudaFree(d_TAU));

  assert(devInfo_h == 0);

  // --- Solving an upper triangular linear system R * x = Q^T * B

  const float alpha = 1.;
  safe_cublas(cublasStrsm(
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
void ARIMAModel<T>::ApplyAR(T *residual, const T *ts_data, const T *phi, int p,
                            int length) {
  int block_size, grid_size;
  compute1DInvokeConfig(length - p, &grid_size, &block_size,
                        update_AR_kernel<T>);

  update_AR_kernel<T>
      <<<grid_size, block_size>>>(ts_data, residual, phi, p, length - p);
}

template <class T>
void ARIMAModel<T>::Fit(const T *data) {
  OK(cudaMalloc(&this->d_data_src, sizeof(T) * this->length));
  OK(cudaMalloc(&this->d_data_differenced, sizeof(T) * this->length));

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
    this->ApplyAR(this->d_data_differenced, this->d_data_src, this->d_phi,
                  this->p, this->DifferencedLength());
    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());

    T *X;
    int rows = min(this->MALength() == 0 ? this->ARLength() : this->MALength(),
                   this->ARLength() == 0 ? this->MALength() : this->ARLength());

    OK(cudaMalloc(&X, sizeof(T) * rows * (this->q + this->p)));

    this->AsMatrix(this->d_data_src + 1, this->d_data_differenced + 1, X,
                   this->p, this->q, rows, this->DifferencedLength());

    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());
    LeastSquaresSolver solver(rows, this->q + this->p);

    solver.Solve(X, this->d_data_src);

    if (this->p > 0) {
      OK(cudaMemcpy(this->d_phi, this->d_data_src, sizeof(T) * this->p,
                    cudaMemcpyDeviceToDevice));

      OK(cudaMemcpy(this->phi, this->d_data_src, sizeof(T) * this->p,
                    cudaMemcpyDeviceToHost));
    }
    OK(cudaDeviceSynchronize());
    OK(cudaGetLastError());

    OK(cudaMemcpy(this->d_theta, this->d_data_src + this->p,
                  sizeof(T) * this->q, cudaMemcpyDeviceToDevice));

    OK(cudaMemcpy(this->theta, this->d_data_src + this->p, sizeof(T) * this->q,
                  cudaMemcpyDeviceToHost));

    OK(cudaFree(X));
  }
  OK(cudaFree(this->d_data_src));
  OK(cudaFree(this->d_data_differenced));
}

void arima_fit_float(int p, int d, int q, int n, float *data) {}

void arima_fit_double(int p, int d, int q, int n, double *data) {}

template class ARIMAModel<float>;
// template class ARIMAModel<double>;
}  // namespace h2o4gpu