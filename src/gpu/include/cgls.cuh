////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015, Christopher Fougner                                    //
// All rights reserved.                                                       //
//                                                                            //
// Redistribution and use in source and binary forms, with or without         //
// modification, are permitted provided that the following conditions are     //
// met:                                                                       //
//                                                                            //
//   1. Redistributions of source code must retain the above copyright        //
//      notice, this list of conditions and the following disclaimer.         //
//                                                                            //
//   2. Redistributions in binary form must reproduce the above copyright     //
//      notice, this list of conditions and the following disclaimer in the   //
//      documentation and/or other materials provided with the distribution.  //
//                                                                            //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        //
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  //
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR //
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          //
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      //
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        //
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         //
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     //
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       //
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         //
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               //
////////////////////////////////////////////////////////////////////////////////

//  CGLS Conjugate Gradient Least Squares
//  Attempts to solve the least squares problem
//
//    min. ||Ax - b||_2^2 + s ||x||_2^2
//
//  using the Conjugate Gradient for Least Squares method. This is more stable
//  than applying CG to the normal equations. Supports both generic operators
//  for computing Ax and A^Tx as well as a sparse matrix version.
//
//  ------------------------------ GENERIC  ------------------------------------
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  F          - Generic GEMV-like functor type with signature
//               int gemv(char op, T alpha, const T *x, T beta, T *y). Upon
//               exit, y should take on the value y := alpha*op(A)x + beta*y.
//               If successful the functor must return 0, otherwise a non-zero
//               value should be returned.
//
//  Function Arguments:
//  A          - Operator that computes Ax and A^Tx.
//
//  (m, n)     - Matrix dimensions of A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter s. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended > 100).
//
//  quiet      - Disable printing to console.
//
//  ------------------------------ SPARSE --------------------------------------
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  O          - Sparse ordering (cgls::CSC or cgls::CSR).
//
//  Function Arguments:
//  val        - Array of matrix values. The array should be of length nnz.
//
//  ptr        - Column pointer if (O is CSC) or row pointer if (O is CSR).
//               The array should be of length m+1.
//
//  ind        - Row indices if (O is CSC) or column indices if (O is CSR).
//               The array should be of length nnz.
//
//  (m, n)     - Matrix dimensions of A.
//
//  nnz        - Number of non-zeros in A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter s. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended > 100).
//
//  quiet      - Disable printing to console.
//
//  ----------------------------------------------------------------------------
//
//  Returns:
//  0 : CGLS converged to the desired tolerance tol within maxit iterations.
//  1 : The vector b had norm less than eps, solution likely x = 0.
//  2 : CGLS iterated maxit times but did not converge.
//  3 : Matrix (A'*A + shift*I) seems to be singular or indefinite.
//  4 : Likely instable, (A'*A + shift*I) indefinite and norm(x) decreased.
//  5 : Error in applying operator A.
//  6 : Error in applying operator A^T.
//
//  Reference:
//  http://web.stanford.edu/group/SOL/software/cgls/
//

#ifndef CGLS_CUH_
#define CGLS_CUH_

#include <assert.h>
#include <stdio.h>

#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <limits>

// Macro to distinguish between cublas and thrust nrm2.
// #define CGLS_USE_THRUST_NRM2
// #define CGLS_DISABLE_ERROR_CHECK

// Macro to check for CUDA errors.
#ifndef CGLS_DISABLE_ERROR_CHECK
#define CGLS_CUDA_CHECK_ERR() \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      printf("%s:%d:%s\n ERROR_CUDA: %s\n", __FILE__, __LINE__, __func__, \
             cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)
#else
#define CGLS_CUDA_CHECK_ERR()
#endif

namespace cgls {

// Data type for sparse format.
enum CGLS_ORD { CSC, CSR };

// Data type for indices. Don't change this unless Nvidia some day
// changes their API (a la MKL).
typedef int INT;

// Abstract GEMV-like operator.
template <typename T>
struct Gemv {
  virtual ~Gemv() { };
  virtual int operator()(char op, const T alpha, const T *x, const T beta,
                         T *y) const = 0;
};

// File-level functions and classes.
namespace {

// Converts 'n' or 't' to a cusparseOperation_t variable.
inline cusparseOperation_t OpToCusparseOp(char op) {
  assert(op == 'n' || op == 'N' || op == 't' || op == 'T');
  return (op == 'n' || op == 'N') ?
      CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
}

inline cusparseOperation_t OpToCusparseCxOp(char op) {
  assert(op == 'n' || op == 'N' || op == 't' || op == 'T');
  return (op == 'n' || op == 'N') ?
      CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
}

// Sparse matrix-vector multiply templates.
template <typename T, CGLS_ORD O>
class Spmv : Gemv<T> {
 private:
  cusparseHandle_t _handle;
  cusparseMatDescr_t _descr;

  INT _m, _n, _nnz;
  const T *_val;
  const INT *_ptr, *_ind;

 public:
  Spmv(INT m, INT n, INT nnz, const T *val, const INT *ptr, const INT *ind)
      : _m(m), _n(n), _nnz(nnz), _val(val), _ptr(ptr), _ind(ind) {
    cusparseCreate(&_handle);
    cusparseCreateMatDescr(&_descr);
    CGLS_CUDA_CHECK_ERR();
  }
  ~Spmv() {
    cusparseDestroy(_handle);
    cusparseDestroyMatDescr(_descr);
    CGLS_CUDA_CHECK_ERR();
  }
  int operator()(char op, const T alpha, const T *x, const T beta, T *y) const;
};

// Double CSR and CSC.
template <>
inline int Spmv<double, CSR>::operator()(char op, const double alpha,
                                         const double *x, const double beta,
                                         double *y) const {
  cusparseStatus_t err = cusparseDcsrmv(_handle, OpToCusparseOp(op), _m, _n,
      _nnz, &alpha, _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

template <>
inline int Spmv<double, CSC>::operator()(char op, const double alpha,
                                         const double *x, const double beta,
                                         double *y) const {
  cusparseOperation_t cu_op = OpToCusparseOp(op);
  if (cu_op == CUSPARSE_OPERATION_TRANSPOSE)
    cu_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    cu_op = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseStatus_t err = cusparseDcsrmv(_handle, cu_op, _n, _m, _nnz, &alpha,
      _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

// Float CSR and CSC.
template <>
inline int Spmv<float, CSR>::operator()(char op, const float alpha,
                                        const float *x, const float beta,
                                        float *y) const {
  cusparseStatus_t err = cusparseScsrmv(_handle, OpToCusparseOp(op), _m, _n,
      _nnz, &alpha, _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

template <>
inline int Spmv<float, CSC>::operator()(char op, const float alpha,
                                        const float *x, const float beta,
                                        float *y) const {
  cusparseOperation_t cu_op = OpToCusparseOp(op);
  if (cu_op == CUSPARSE_OPERATION_TRANSPOSE)
    cu_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    cu_op = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseStatus_t err = cusparseScsrmv(_handle, cu_op, _n, _m, _nnz, &alpha,
      _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err;
}

// Double Complex CSR and CSC.
template <>
inline int Spmv<cuDoubleComplex, CSR>::
    operator()(char op, const cuDoubleComplex alpha, const cuDoubleComplex *x,
               const cuDoubleComplex beta, cuDoubleComplex *y) const {
  cusparseStatus_t err = cusparseZcsrmv(_handle, OpToCusparseCxOp(op), _m, _n,
      _nnz, &alpha, _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

template <>
inline int Spmv<cuDoubleComplex, CSC>::
    operator()(char op, const cuDoubleComplex alpha, const cuDoubleComplex *x,
               const cuDoubleComplex beta, cuDoubleComplex *y) const {
  cusparseOperation_t cu_op = OpToCusparseCxOp(op);
  if (cu_op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    cu_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    cu_op = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  cusparseStatus_t err = cusparseZcsrmv(_handle, cu_op, _n, _m, _nnz, &alpha,
      _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

// Single Complex CSR and CSC.
template <>
inline int Spmv<cuFloatComplex, CSR>::
    operator()(char op, const cuFloatComplex alpha, const cuFloatComplex *x,
               const cuFloatComplex beta, cuFloatComplex *y) const {
  cusparseStatus_t err = cusparseCcsrmv(_handle, OpToCusparseCxOp(op), _m, _n,
      _nnz, &alpha, _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

template <>
inline int Spmv<cuFloatComplex, CSC>::
    operator()(char op, const cuFloatComplex alpha, const cuFloatComplex *x,
               const cuFloatComplex beta, cuFloatComplex *y) const {
  cusparseOperation_t cu_op = OpToCusparseCxOp(op);
  if (cu_op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    cu_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    cu_op = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  cusparseStatus_t err = cusparseCcsrmv(_handle, cu_op, _n, _m, _nnz, &alpha,
      _descr, _val, _ptr, _ind, x, &beta, y);
  CGLS_CUDA_CHECK_ERR();
  return err != CUSPARSE_STATUS_SUCCESS;
}

// Class for sparse matrix and its transpose.
template <typename T, CGLS_ORD O>
class SpmvNT : Gemv<T> {
 private:
  Spmv<T, O> A;
  Spmv<T, O> At;

 public:
  SpmvNT(INT m, INT n, INT nnz, const T *val_a, const INT *ptr_a,
         const INT *ind_a, const T *val_at, const INT *ptr_at,
         const INT *ind_at)
      : A(m, n, nnz, val_a, ptr_a, ind_a),
        At(n, m, nnz, val_at, ptr_at, ind_at) { }
  int operator()(char op, const T alpha, const T *x, const T beta, T *y) const {
    switch (O) {
      case CSR:
        if (op == 'n' || op == 'N')
          return A('n', alpha, x, beta, y);
        else
          return At('n', alpha, x, beta, y);
      case CSC:
        if (op == 'n' || op == 'N')
          return At('t', alpha, x, beta, y);
        else
          return A('t', alpha, x, beta, y);
      default:
        assert(false);
        return 1;
    }
  }
};

// Axpy function.
inline cublasStatus_t axpy(cublasHandle_t handle, INT n, double *alpha,
                           const double *x, INT incx, double *y, INT incy) {
  cublasStatus_t err = cublasDaxpy(handle, n, alpha, x, incx, y, incy);
  CGLS_CUDA_CHECK_ERR();
  return err;
}

inline cublasStatus_t axpy(cublasHandle_t handle, INT n, float *alpha,
                           const float *x, INT incx, float *y, INT incy) {
  cublasStatus_t err = cublasSaxpy(handle, n, alpha, x, incx, y, incy);
  CGLS_CUDA_CHECK_ERR();
  return err;
}

inline cublasStatus_t axpy(cublasHandle_t handle, INT n, cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, INT incx,
                           cuDoubleComplex *y, INT incy) {
  cublasStatus_t err = cublasZaxpy(handle, n, alpha, x, incx, y, incy);
  CGLS_CUDA_CHECK_ERR();
  return err;
}

inline cublasStatus_t axpy(cublasHandle_t handle, INT n, cuFloatComplex *alpha,
                           const cuFloatComplex *x, INT incx, cuFloatComplex *y,
                           INT incy) {
  cublasStatus_t err = cublasCaxpy(handle, n, alpha, x, incx, y, incy);
  CGLS_CUDA_CHECK_ERR();
  return err;
}

#ifdef CGLS_USE_THRUST_NRM2

// 2-Norm based on thrust, potentially not as stable as cuBLAS version.
template <typename T>
struct NormSquared : thrust::unary_function<T, double> {
  inline __device__ double operator()(const T &x);
};
template <>
inline __device__ double NormSquared<double>::operator()(const double &x) {
  return x * x;
}

template <>
inline __device__ double NormSquared<float>::operator()(const float &x) {
  return static_cast<double>(x) * static_cast<double>(x);
}

template <>
inline __device__ double NormSquared<cuDoubleComplex>::
    operator()(const cuDoubleComplex &x) {
  return static_cast<double>(x.x) * static_cast<double>(x.x) +
      static_cast<double>(x.y) * static_cast<double>(x.y);
}

template <>
inline __device__ double NormSquared<cuFloatComplex>::
    operator()(const cuFloatComplex &x) {
  return static_cast<double>(x.x) * static_cast<double>(x.x) +
      static_cast<double>(x.y) * static_cast<double>(x.y);
}

template <typename T>
void nrm2(cublasHandle_t hdl, INT n, const T *x, double *result) {
  *result = sqrt(thrust::transform_reduce(thrust::device_pointer_cast(x),
      thrust::device_pointer_cast(x + n), NormSquared<T>(), 0.,
      thrust::plus<double>()));
  CGLS_CUDA_CHECK_ERR();
}

#else

// cuBLAS nrm2 implementation.
void nrm2(cublasHandle_t hdl, INT n, const double *x, double *result) {
  cublasDnrm2(hdl, n, x, static_cast<INT>(1), result);
  CGLS_CUDA_CHECK_ERR();
}

void nrm2(cublasHandle_t hdl, INT n, const float *x, double *result) {
  float result_float;
  cublasSnrm2(hdl, n, x, static_cast<INT>(1), &result_float);
  *result = static_cast<double>(result_float);
  CGLS_CUDA_CHECK_ERR();
}

void nrm2(cublasHandle_t hdl, INT n, const cuDoubleComplex *x, double *result) {
  cublasDznrm2(hdl, n, x, static_cast<INT>(1), result);
  CGLS_CUDA_CHECK_ERR();
}

void nrm2(cublasHandle_t hdl, INT n, const cuFloatComplex *x, double *result) {
  float result_float;
  cublasScnrm2(hdl, n, x, static_cast<INT>(1), &result_float);
  *result = static_cast<double>(result_float);
  CGLS_CUDA_CHECK_ERR();
}

#endif

// Casting from double to float, double, complex_float, and complex_double.
template <typename T>
T StaticCast(double x);

template <>
inline double StaticCast<double>(double x) {
 return x;
}

template <>
inline float StaticCast<float>(double x) {
 return static_cast<float>(x);
}

template <>
inline cuDoubleComplex StaticCast<cuDoubleComplex>(double x) {
 return make_cuDoubleComplex(x, 0.);
}

template <>
inline cuFloatComplex StaticCast<cuFloatComplex>(double x) {
 return make_cuFloatComplex(x, 0.f);
}

// Numeric limit epsilon for float, double, complex_float, and complex_double.
template <typename T>
double Epsilon();

template<>
inline double Epsilon<double>() {
  return std::numeric_limits<double>::epsilon();
}

template<>
inline double Epsilon<cuDoubleComplex>() {
  return std::numeric_limits<double>::epsilon();
}

template<>
inline double Epsilon<float>() {
  return std::numeric_limits<float>::epsilon();
}

template<>
inline double Epsilon<cuFloatComplex>() {
  return std::numeric_limits<float>::epsilon();
}

}  // namespace

// Conjugate Gradient Least Squares.
template <typename T, typename F>
int Solve(cublasHandle_t handle, const F& A, const INT m, const INT n,
          const T *b, T *x, const double shift, const double tol,
          const int maxit, bool quiet) {
  // Variable declarations.
  T *p, *q, *r, *s;
  double gamma, normp, normq, norms, norms0, normx, xmax;
  char fmt[] = "%5d %9.2e %12.5g\n";
  int err = 0, k = 0, flag = 0, indefinite = 0;

  // Constant declarations.
  const T kNegOne   = StaticCast<T>(-1.);
  const T kZero     = StaticCast<T>( 0.);
  const T kOne      = StaticCast<T>( 1.);
  const T kNegShift = StaticCast<T>(-shift);
  const double kEps = Epsilon<T>();

  // Memory Allocation.
  cudaMalloc(&p, n * sizeof(T));
  cudaMalloc(&q, m * sizeof(T));
  cudaMalloc(&r, m * sizeof(T));
  cudaMalloc(&s, n * sizeof(T));
  CGLS_CUDA_CHECK_ERR();

  cudaMemcpy(r, b, m * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);
  CGLS_CUDA_CHECK_ERR();

  // r = b - A*x.
  nrm2(handle, n, x, &normx);
  cudaDeviceSynchronize();
  CGLS_CUDA_CHECK_ERR();
  if (normx > 0.) {
    err = A('n', kNegOne, x, kOne, r);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();
    if (err)
      flag = 5;
  }

  // s = A'*r - shift*x.
  err = A('t', kOne, r, kNegShift, s);
  cudaDeviceSynchronize();
  CGLS_CUDA_CHECK_ERR();
  if (err)
    flag = 6;

  // Initialize.
  cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);
  nrm2(handle, n, s, &norms);
  norms0 = norms;
  gamma = norms0 * norms0;
  nrm2(handle, n, x, &normx);
  xmax = normx;
  cudaDeviceSynchronize();
  CGLS_CUDA_CHECK_ERR();

  if (norms < kEps)
    flag = 1;

  if (!quiet)
    printf("    k     normx        resNE\n");

  for (k = 0; k < maxit && !flag; ++k) {
    // q = A * p.
    err = A('n', kOne, p, kZero, q);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();
    if (err) {
      flag = 5;
      break;
    }

    // delta = norm(p)^2 + shift*norm(q)^2.
    nrm2(handle, n, p, &normp);
    nrm2(handle, m, q, &normq);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();
    double delta = normq * normq + shift * normp * normp;

    if (delta <= 0.)
      indefinite = 1;
    if (delta == 0.)
      delta = kEps;
    T alpha = StaticCast<T>(gamma / delta);
    T neg_alpha = StaticCast<T>(-gamma / delta);

    // x = x + alpha*p.
    // r = r - alpha*q.
    axpy(handle, n, &alpha, p, 1, x,  1);
    axpy(handle, m, &neg_alpha, q, 1, r,  1);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();

    // s = A'*r - shift*x.
    cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);
    err = A('t', kOne, r, kNegShift, s);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();
    if (err) {
      flag = 6;
      break;
    }

    // Compute beta.
    nrm2(handle, n, s, &norms);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();
    double gamma1 = gamma;
    gamma = norms * norms;
    T beta = StaticCast<T>(gamma / gamma1);

    // p = s + beta*p.
    axpy(handle, n, &beta, p, 1, s, 1);
    cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();

    // Convergence check.
    nrm2(handle, n, x, &normx);
    cudaDeviceSynchronize();
    CGLS_CUDA_CHECK_ERR();
    xmax = std::max(xmax, normx);
    bool converged = (norms <= norms0 * tol) || (normx * tol >= 1.);
    if (!quiet && (converged || k % 10 == 0))
      printf(fmt, k, normx, norms / norms0);
    if (converged)
      break;
  }

  // Determine exit status.
  double shrink = normx / xmax;
  if (k == maxit)
    flag = 2;
  else if (indefinite)
    flag = 3;
  else if (shrink * shrink <= tol)
    flag = 4;

  // Free variables and return;
  cudaFree(p);
  cudaFree(q);
  cudaFree(r);
  cudaFree(s);
  CGLS_CUDA_CHECK_ERR();
  return flag;
}

// Sparse CGLS.
template <typename T, CGLS_ORD O>
int Solve(const T *val, const INT *ptr, const INT *ind, const INT m,
          const INT n, const INT nnz, const T *b, T *x, const double shift,
          const double tol, const int maxit, bool quiet) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  CGLS_CUDA_CHECK_ERR();

  Spmv<T, O> A(m, n, nnz, val, ptr, ind);
  int status = Solve(handle, A, m, n, b, x, shift, tol, maxit, quiet);

  cublasDestroy(handle);
  CGLS_CUDA_CHECK_ERR();
  return status;
}

// Sparse CGLS with both A and A^T.
template <typename T, CGLS_ORD O>
int Solve(const T *val_a, const INT *ptr_a, const INT *ind_a, const T *val_at,
          const INT *ptr_at, const INT *ind_at, const INT m, const INT n,
          const INT nnz, const T *b, T *x, const double shift, const double tol,
          const int maxit, bool quiet) {

  cublasHandle_t handle;
  cublasCreate(&handle);
  CGLS_CUDA_CHECK_ERR();

  SpmvNT<T, O> A(m, n, nnz, val_a, ptr_a, ind_a, val_at, ptr_at, ind_at);
  int status = Solve(handle, A, m, n, b, x, shift, tol, maxit, quiet);

  cublasDestroy(handle);
  CGLS_CUDA_CHECK_ERR();
  return status;
}

}  // namespace cgls

#endif  // CGLS_CUH_

