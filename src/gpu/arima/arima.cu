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
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // time axis
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO: optimize with shared memory(read data only once)
  if (i < n) X[j * ldx + i] = data[j + i];
}

LeastSquaresSolver::LeastSquaresSolver(int rows, int cols)
    : rows(rows), cols(cols) {
  safe_cusolver(cusolverDnCreate(&this->solver_handle));
  safe_cublas(cublasCreate(&this->cublas_handle));
}

void LeastSquaresSolver::Solve(float *A, float *B) {
  int work_size = 0;
  int *devInfo;

  OK(cudaMalloc(&devInfo, sizeof(int)));
  // --- CUDA solver initialization
  cusolverDnHandle_t solver_handle;
  safe_cusolver(cusolverDnCreate(&solver_handle));

  // --- CUBLAS initialization
  cublasHandle_t cublas_handle;
  safe_cublas(cublasCreate(&cublas_handle));

  /**********************************/
  /* COMPUTING THE QR DECOMPOSITION */
  /**********************************/

  // --- CUDA QR GEQRF preliminary operations
  float *d_TAU;
  OK(cudaMalloc((void **)&d_TAU, min(rows, cols) * sizeof(float)));
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
  OK(cudaDeviceSynchronize());
  int devInfo_h = 0;
  OK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  assert(devInfo_h == 0);
  //   if (devInfo_h != 0) std::cout << "Unsuccessful gerf execution\n\n";

  /*****************************/
  /* SOLVING THE LINEAR SYSTEM */
  /*****************************/

  // --- CUDA ORMQR execution: Computes the multiplication Q^T * C and stores it
  // in d_C
  safe_cusolver(cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                 rows, 1, min(rows, cols), A, rows, d_TAU, B,
                                 rows, work, work_size, devInfo));

  OK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

  assert(devInfo_h == 0);
  // --- Reducing the linear system size
  float *d_R;
  OK(cudaMalloc(&d_R, cols * cols * sizeof(float)));
  //   float *d_B;
  //   OK(cudaMalloc(&d_B, cols * sizeof(float)));
  //   dim3 Grid(DIVUP(cols, BLOCK_SIZE), DIVUP(cols, BLOCK_SIZE));
  //   dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
  //   copy_kernel<float><<<Grid, Block>>>(A, d_R, rows, cols);
  //   OK(cudaMemcpy(d_B, B, cols * sizeof(float), cudaMemcpyDeviceToDevice));

  // --- Solving an upper triangular linear system R * x = Q^T * B
  // R*x = B

  thrust::device_vector<float> d_r(d_R, d_R + cols * cols);
  thrust::host_vector<float> h_r = d_r;

  const float alpha = 1.;
  safe_cublas(cublasStrsm(
      cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, cols, 1, &alpha, A, rows, B, cols));

  OK(cudaDeviceSynchronize());
  OK(cudaFree(d_R));
}

template <class T>
__global__ void differencing(T *out, const T *in, const int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (i > 0)
      out[i] = in[i] - in[i - 1];
    else
      out[0] = NAN;
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
    : p(p), d(d), q(q), length(length){};

template <class T>
ARIMAModel<T>::~ARIMAModel() {
  OK(cudaFree(this->d_data_src));
  OK(cudaFree(this->d_data_differenced));
}

template <class T>
void ARIMAModel<T>::AsMatrix(T *ts_data, T *A, int depth, int lda, int length) {
  int n = length - depth + 1;
  dim3 grid_size(DIVUP(n, BLOCK_SIZE), depth);
  dim3 block_size(BLOCK_SIZE, 1);
  ts_data_to_matrix_kernel<T><<<grid_size, block_size>>>(ts_data, A, lda, n);
}

template <class T>
void ARIMAModel<T>::Fit(const T *data) {
  OK(cudaMalloc(&this->d_data_src, sizeof(T) * this->length));
  OK(cudaMalloc(&this->d_data_differenced, sizeof(T) * this->length));

  OK(cudaMemcpy(this->d_data_src, data, sizeof(T) * this->length,
                cudaMemcpyHostToDevice));

  int block_size, grid_size;
  compute1DInvokeConfig(this->length, &grid_size, &block_size, differencing<T>);
  for (auto i = 0; i < this->d; ++i) {
    differencing<T><<<grid_size, block_size>>>(this->d_data_differenced,
                                               this->d_data_src, this->length);
    OK(cudaDeviceSynchronize());
    std::swap(this->d_data_src, this->d_data_differenced);
  }
}

template <class T>
void ARIMAModel<T>::AR(T *X, T *residual) {}

template <class T>
void ARIMAModel<T>::MA(T *epsilon, T *residual) {}

void arima_fit_float(int p, int d, int q, int n, float *data) {}

void arima_fit_double(int p, int d, int q, int n, double *data) {}

template class ARIMAModel<float>;
template class ARIMAModel<double>;
}  // namespace h2o4gpu