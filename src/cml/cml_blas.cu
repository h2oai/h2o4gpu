#include "cml_blas.cuh"

namespace cml {

cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<float> *x,
                         float *result) {
  cublasStatus_t err = cublasSnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

// Scal.
cublasStatus_t blas_scal(cublasHandle_t handle, const double alpha,
                         vector<double> *x) {
  cublasStatus_t err = cublasDscal(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_scal(cublasHandle_t handle, const float alpha,
                         vector<float> *x) {
  cublasStatus_t err = cublasSscal(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_scal(cublasHandle_t handle, const double *alpha,
                         vector<double> *x) {
  cublasStatus_t err = cublasDscal(handle, static_cast<int>(x->size), alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_scal(cublasHandle_t handle, const float *alpha,
                         vector<float> *x) {
  cublasStatus_t err = cublasSscal(handle, static_cast<int>(x->size), alpha,
      x->data, static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

// Asum.
cublasStatus_t blas_asum(cublasHandle_t handle, const vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDasum(handle, x->size, x->data, x->stride, result);
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_asum(cublasHandle_t handle, const vector<float> *x,
                         float *result) {
  cublasStatus_t err = cublasSasum(handle, x->size, x->data, x->stride, result);
  CublasCheckError(err);
  return err;
}

// Dot.
cublasStatus_t blas_dot(cublasHandle_t handle, const vector<double> *x,
                        const vector<double> *y, double *result) {
  cublasStatus_t err = cublasDdot(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), y->data, static_cast<int>(y->stride), result);
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_dot(cublasHandle_t handle, const vector<float> *x,
                        const vector<float> *y, float *result) {
  cublasStatus_t err = cublasSdot(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), y->data, static_cast<int>(y->stride), result);
  CublasCheckError(err);
  return err;
}

// Axpy.
cublasStatus_t blas_axpy(cublasHandle_t handle, double alpha,
                         const vector<double> *x, vector<double> *y) {
  cublasStatus_t err = cublasDaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

cublasStatus_t blas_axpy(cublasHandle_t handle, float alpha,
                         const vector<float> *x, vector<float> *y) {
  cublasStatus_t err = cublasSaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

}  // namespace cml

