#ifndef SRC_INCLUDE_SOLVER_ARIMA_H

namespace h2o4gpu {

#include <cuda_runtime.h>
#include <cusolverDn.h>

template <class T>
__global__ void ts_data_to_matrix_kernel(const T* __restrict data, T* X,
                                         const int ldx, const int n);

template <class T>
class ARIMAModel {
 public:
  ARIMAModel(int p, int d, int q, int length);
  ~ARIMAModel();
  const int p;
  const int d;
  const int q;
  const int length;

  static void AsMatrix(T* ts_data, T* A, int depth, int lda, int length);

  void Fit(const T* data);
  void AR(T* X, T* residual);
  void MA(T* epsilon, T* residual);

 private:
  T* d_data_src;
  T* d_data_differenced;
};

class LeastSquaresSolver {
 public:
  LeastSquaresSolver(int rows, int cols);
  void Solve(float* A, float* B);
  const int rows;
  const int cols;

 private:
  cusolverDnHandle_t solver_handle;
  cublasHandle_t cublas_handle;
};

void arima_fit_float(int p, int d, int q, int n, float* data);

void arima_fit_double(int p, int d, int q, int n, double* data);

}  // namespace h2o4gpu
#endif