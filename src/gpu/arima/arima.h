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
  /**
   * @brief Construct a new ARIMAModel object
   *
   * @param p ARIMA AR parameter
   * @param d ARIMA differencing order
   * @param q ARIMA MA parameter
   * @param length length of time series
   */
  ARIMAModel(int p, int d, int q, int length);
  ~ARIMAModel();
  const int p;
  const int d;
  const int q;
  const int length;

  /**
   * @brief perform time series data differencing
   *
   * @param out
   * @param in
   * @param length
   */
  static void Difference(T* out, const T* in, int length);
  /**
   * @brief convert ts data into a matrix where every row is shifted by time
   *
   * @param ts_data ts data
   * @param A matrix
   * @param depth how many time series components are taken to create a row
   * @param lda A's leading dimension
   * @param length ts length
   */
  static void AsMatrix(const T* ts_data, T* A, int depth, int lda, int length);
  /**
   * @brief convert two ts arrays into a matrix where every row is created by
   * appending data from each array and shifted by time.
   *
   * @param ts_a first ts array
   * @param ts_b second ts array
   * @param A resulting matrix
   * @param a_depth ts depth for the first array
   * @param b_depth ts depth for the second array
   * @param lda A's leading dimesion size
   * @param length ts length
   */
  static void AsMatrix(const T* ts_a, const T* ts_b, T* A, int a_depth,
                       int b_depth, int lda, int length);
  /**
   * @brief applies model and computes residuals
   *
   * @param residual
   * @param ts_data
   * @param phi
   * @param p
   * @param length
   */
  static void Apply(T* residual, const T* ts_data, const T* phi, const int p,
                    const T* last_residual, const T* theta, const int q,
                    int length);

  /**
   * @brief fit ARIMA model
   *
   * @param data
   */
  void Fit(const T* data, const int maxiter = 1);

  inline int ARLength() { return this->DifferencedLength() - this->p; }
  inline int MALength() { return this->DifferencedLength() - this->q; }
  inline int DifferencedLength() { return this->length - this->d; }
  inline T* Theta() { return this->theta; }
  inline T* Phi() { return this->phi; }

 private:
  T* d_data_src;
  T* d_data_differenced;
  T* d_last_residual;
  T* d_buffer;
  T* d_theta;
  T* d_phi;
  T* theta;
  T* phi;
};

class LeastSquaresSolver {
 public:
  LeastSquaresSolver(int rows, int cols);
  ~LeastSquaresSolver();
  template <class T>
  void Solve(T* A, T* B);
  const int rows;
  const int cols;

 private:
  cusolverDnHandle_t solver_handle;
  cublasHandle_t cublas_handle;
};

}  // namespace h2o4gpu

// void arima_fit_float(int p, int d, int q, int n, float* data,
//                      const int maxiter);

// void arima_fit_double(int p, int d, int q, int n, double* data,
//                       const int maxiter);

#endif