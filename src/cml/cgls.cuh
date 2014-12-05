////////////////////////////////////////////////////////////////////////////////
// Copyright 2014 Chris Fougner.                                              //
//                                                                            //
// This program is free software: you can redistribute it and/or modify       //
// it under the terms of the GNU General Public License as published by       //
// the Free Software Foundation, either version 3 of the License, or          //
// (at your option) any later version.                                        //
//                                                                            //
// This program is distributed in the hope that it will be useful,            //
// but WITHOUT ANY WARRANTY; without even the implied warranty of             //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              //
// GNU General Public License for more details.                               //
//                                                                            //
// You should have received a copy of the GNU General Public License          //
// along with this program.  If not, see http://www.gnu.org/licenses/.        //
////////////////////////////////////////////////////////////////////////////////

//  CGLS Conjugate Gradient Least Squares
//  Attempts to solve the least squares problem
//
//    min. ||Ax - b||_2^2 + s ||x||_2^2
//
//  using the Conjugate Gradient for Least Squares method. This is more stable
//  than applying CG to the normal equations.
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  F          - Sparse ordering (cgls::CSC or cgls::CSR).
//
//  Function Arguments:
//  handle_s   - Cusparse handle.
//
//  handle_b   - Cublas handle.
//
//  descr      - Cusparse matrix descriptor (i.e. 0- or 1-based indexing)
//
//  val        - Array of matrix values. The array should be of length nnz.
//
//  ptr        - Column pointer if (F is CSC) or row pointer if (F is CSR).
//               The array should be of length m+1.
//
//  ind        - Row indices if (F is CSC) or column indices if (F is CSR).
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
//  Returns:
//  0 : CGLS converged to the desired tolerance tol within maxit iterations.
//  1 : The vector b had norm less than eps, solution likely x = 0.
//  2 : CGLS iterated maxit times but did not converge.
//  3 : Matrix (A'*A + shift*I) seems to be singular or indefinite.
//  4 : Likely instable, (A'*A + shift*I) indefinite and norm(x) decreased.
//
//  Reference:
//  http://web.stanford.edu/group/SOL/software/cgls/
//

#ifndef CGLS_CUH_
#define CGLS_CUH_

#include <stdio.h>

#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>

namespace cgls {

// Data type for sparse format.
enum CGLS_ORD { CSC, CSR };

// Data type for indices. Don't change this unless Nvidia some day
// changes their API (a la MKL).
typedef int INT;

// Templated BLAS operations. Skip this part if you're looking for CGLS.
namespace {

// Sparse matrix-vector multiply templates.
template <typename T, CGLS_ORD F>
cusparseStatus_t spmv(cusparseHandle_t handle, cusparseOperation_t transA,
                      INT m, INT n, INT nnz, const T *alpha,
                      cusparseMatDescr_t descrA, const T *val, const INT *ptr,
                      const INT *ind, const T *x, const T *beta, T *y);

template <>
cusparseStatus_t spmv<double, CSR>(cusparseHandle_t handle,
                                   cusparseOperation_t transA, INT m, INT n,
                                   INT nnz, const double *alpha,
                                   cusparseMatDescr_t descrA, const double *val,
                                   const INT *ptr, const INT *ind,
                                   const double *x, const double *beta,
                                   double *y) {
  return cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, val, ptr,
      ind, x, beta, y);
}

template <>
cusparseStatus_t spmv<double, CSC>(cusparseHandle_t handle,
                                   cusparseOperation_t transA, INT m, INT n,
                                   INT nnz, const double *alpha,
                                   cusparseMatDescr_t descrA, const double *val,
                                   const INT *ptr, const INT *ind,
                                   const double *x, const double *beta,
                                   double *y) {
  if (transA == CUSPARSE_OPERATION_TRANSPOSE)
    transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    transA = CUSPARSE_OPERATION_TRANSPOSE;
  return cusparseDcsrmv(handle, transA, n, m, nnz, alpha, descrA, val, ptr,
      ind, x, beta, y);
}

template <>
cusparseStatus_t spmv<float, CSR>(cusparseHandle_t handle,
                                  cusparseOperation_t transA, INT m, INT n,
                                  INT nnz, const float *alpha,
                                  cusparseMatDescr_t descrA, const float *val,
                                  const INT *ptr, const INT *ind,
                                  const float *x, const float *beta,
                                  float *y) {
  return cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, val, ptr,
      ind, x, beta, y);
}

template <>
cusparseStatus_t spmv<float, CSC>(cusparseHandle_t handle,
                                  cusparseOperation_t transA, INT m, INT n,
                                  INT nnz, const float *alpha,
                                  cusparseMatDescr_t descrA, const float *val,
                                  const INT *ptr, const INT *ind,
                                  const float *x, const float *beta,
                                  float *y) {
  if (transA == CUSPARSE_OPERATION_TRANSPOSE)
    transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    transA = CUSPARSE_OPERATION_TRANSPOSE;
  return cusparseScsrmv(handle, transA, n, m, nnz, alpha, descrA, val, ptr,
      ind, x, beta, y);
}

// AXPY function.
cublasStatus_t axpy(cublasHandle_t handle, INT n, double *alpha,
                    const double *x, INT incx, double *y, INT incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t axpy(cublasHandle_t handle, INT n, float *alpha,
                    const float *x, INT incx, float *y, INT incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

// 2-Norm based on thrust.
template <typename T>
struct Square : thrust::unary_function<T, T> {
  __device__ T operator()(const T &x) {
    return x * x;
  }
};

template <typename T>
void nrm2(INT n, const T *x, T *result) {
  *result = sqrt(thrust::transform_reduce(thrust::device_pointer_cast(x),
      thrust::device_pointer_cast(x + n), Square<T>(), static_cast<T>(0),
      thrust::plus<T>()));
}

}  // namespace

// TODO(chris): Check cuda errors.
// Conjugate Gradient Least Squares. This version depends only on the matrix
// A and may in practice be much slower than the version using A and A^T
// separately. This is because of constant memory allocation and freeing.
template <typename T, CGLS_ORD F>
INT solve(cusparseHandle_t handle_s, cublasHandle_t handle_b,
          cusparseMatDescr_t descr, const T *val, const INT *ptr,
          const INT *ind, const INT m, const INT n, const INT nnz, const T *b,
          T *x, const T shift, const T tol, const INT maxit, bool quiet) {
  // Variable declarations.
  T *p, *q, *r, *s;
  T gamma, normp, normq, norms, norms0, normx, xmax;
  char fmt[] = "%5d %9.2e %12.5g\n";
  INT k, flag = 0, indefinite = 0;

  // Constant declarations.
  const T kNegOne = static_cast<T>(-1);
  const T kZero = static_cast<T>(0);
  const T kOne = static_cast<T>(1);
  const T kNegShift = static_cast<T>(-shift);
  const T kEps = static_cast<T>(1e-16);

  // Memory Allocation.
  cudaMalloc(&p, n * sizeof(T));
  cudaMalloc(&q, m * sizeof(T));
  cudaMalloc(&r, m * sizeof(T));
  cudaMalloc(&s, n * sizeof(T));

  cudaMemcpy(r, b, m * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);

  // r = b - A*x.
  spmv<T, F>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &kNegOne,
      descr, val, ptr, ind, x, &kOne, r);

  // s = A'*r - shift*x.
  spmv<T, F>(handle_s, CUSPARSE_OPERATION_TRANSPOSE, m, n, nnz, &kOne,
      descr, val, ptr, ind, r, &kNegShift, s);

  // Initialize.
  cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);
  nrm2(n, s, &norms);
  cudaDeviceSynchronize();
  norms0 = norms;
  gamma = norms0 * norms0;
  nrm2(n, x, &normx);
  cudaDeviceSynchronize();
  xmax = normx;

  if (norms < kEps)
    flag = 1;

  if (!quiet)
    printf("    k     normx        resNE\n");

  for (k = 0; k < maxit && !flag; ++k) {
    // q = A * p.
    spmv<T, F>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &kOne,
        descr, val, ptr, ind, p, &kZero, q);

    // delta = norm(p)^2 + shift*norm(q)^2.
    nrm2(n, p, &normp);
    nrm2(m, q, &normq);
    cudaDeviceSynchronize();
    T delta = normq * normq + shift * normp * normp;

    if (delta <= 0)
      indefinite = 1;
    if (delta == 0)
      delta = kEps;
    T alpha = gamma / delta;
    T neg_alpha = -alpha;

    // x = x + alpha*p.
    // r = r - alpha*q.
    axpy(handle_b, n, &alpha, p, 1, x,  1);
    axpy(handle_b, m, &neg_alpha, q, 1, r,  1);

    // s = A'*r - shift*x.
    cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);
    spmv<T, F>(handle_s, CUSPARSE_OPERATION_TRANSPOSE, m, n, nnz, &kOne,
        descr, val, ptr, ind, r, &kNegShift, s);

    // Compute beta.
    nrm2(n, s, &norms);
    cudaDeviceSynchronize();
    T gamma1 = gamma;
    gamma = norms * norms;
    T beta = gamma / gamma1;

    // p = s + beta*p.
    axpy(handle_b, n, &beta, p, 1, s, 1);
    cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);

    // Convergence check.
    nrm2(n, x, &normx);
    cudaDeviceSynchronize();
    xmax = std::max(xmax, normx);
    bool converged = (norms <= norms0 * tol) || (normx * tol >= 1);
    if (!quiet && (converged || k % 10 == 0))
      printf(fmt, k, normx, norms / norms0);
    if (converged)
      break;
  }

  // Determine exit status.
  T shrink = normx / xmax;
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
  return flag;
}

// CGLS, with pre-initialized cusparseHandle and cublasHandle.
template <typename T, CGLS_ORD F>
INT solve(cusparseMatDescr_t descr, const T *val, const INT *ptr,
          const INT *ind, const INT m, const INT n, const INT nnz,
          const T *b, T *x, const T shift, const T tol, const INT maxit,
          bool quiet) {
  cusparseHandle_t handle_s;
  cublasHandle_t handle_b;
  cusparseCreate(&handle_s);
  cublasCreate(&handle_b);
  int flag = solve<T, F>(handle_s, handle_b, descr, val, ptr, ind, m, n, nnz, b,
      x, shift, tol, maxit, quiet);
  cusparseDestroy(handle_s);
  cublasDestroy(handle_b);
  return flag;
}

// CGLS, with pre-initialized cusparseMatDescr, cusparseHandle and cublasHandle.
template <typename T, CGLS_ORD F>
INT solve(const T *val, const INT *ptr, const INT *ind, const INT m,
          const INT n, const INT nnz, const T *b, T *x, const T shift,
          const T tol, const INT maxit, bool quiet) {
  cusparseHandle_t handle_s;
  cublasHandle_t handle_b;
  cusparseMatDescr_t descr;
  cusparseCreate(&handle_s);
  cublasCreate(&handle_b);
  cusparseCreateMatDescr(&descr);
  int flag = solve<T, F>(handle_s, handle_b, descr, val, ptr, ind, m, n, nnz, b,
      x, shift, tol, maxit, quiet);
  cusparseDestroy(handle_s);
  cublasDestroy(handle_b);
  cusparseDestroyMatDescr(descr);
  return flag;
}

// This version requires both A and A^T
template <typename T, CGLS_ORD F>
INT solve(cusparseHandle_t handle_s, cublasHandle_t handle_b,
          cusparseMatDescr_t descr, const T *val_a, const INT *ptr_a,
          const INT *ind_a, const T *val_at, const INT *ptr_at,
          const INT *ind_at, const INT m, const INT n, const INT nnz,
          const T *b, T *x, const T shift, const T tol, const INT maxit,
          bool quiet) {
  // Variable declarations.
  T *p, *q, *r, *s;
  T gamma, normp, normq, norms, norms0, normx, xmax;
  char fmt[] = "%5d %9.2e %12.5g\n";
  INT k, flag = 0, indefinite = 0;

  // Constant declarations.
  const T kNegOne = static_cast<T>(-1);
  const T kZero = static_cast<T>(0);
  const T kOne = static_cast<T>(1);
  const T kNegShift = static_cast<T>(-shift);
  const T kEps = static_cast<T>(1e-16);

  // Memory Allocation.
  cudaMalloc(&p, n * sizeof(T));
  cudaMalloc(&q, m * sizeof(T));
  cudaMalloc(&r, m * sizeof(T));
  cudaMalloc(&s, n * sizeof(T));

  cudaMemcpy(r, b, m * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);

  // r = b - A*x.
  if (F == CSR)
    spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
        &kNegOne, descr, val_a, ptr_a, ind_a, x, &kOne, r);
  else
    spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
        &kNegOne, descr, val_at, ptr_at, ind_at, x, &kOne, r);

  cudaDeviceSynchronize();

  // s = A'*r - shift*x.
  if (F == CSR)
    spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, n, m, nnz, &kOne,
        descr, val_at, ptr_at, ind_at, r, &kNegShift, s);
  else
    spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, n, m, nnz, &kOne,
        descr, val_a, ptr_a, ind_a, r, &kNegShift, s);

  // Initialize.
  cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);
  nrm2(n, s, &norms);
  nrm2(n, x, &normx);
  cudaDeviceSynchronize();
  norms0 = norms;
  gamma = norms0 * norms0;
  xmax = normx;

  if (norms < kEps)
    flag = 1;

  if (!quiet && !flag)
    printf("    k     normx        resNE\n");

  for (k = 0; k < maxit && !flag; ++k) {
    // q = A * p.
    if (F == CSR)
      spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &kOne,
          descr, val_a, ptr_a, ind_a, p, &kZero, q);
    else
      spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &kOne,
          descr, val_at, ptr_at, ind_at, p, &kZero, q);
    cudaDeviceSynchronize();

    // delta = norm(p)^2 + shift*norm(q)^2.
    nrm2(n, p, &normp);
    nrm2(m, q, &normq);
    cudaDeviceSynchronize();
    T delta = normq * normq + shift * normp * normp;

    if (delta <= 0)
      indefinite = 1;
    if (delta == 0)
      delta = kEps;
    T alpha = gamma / delta;
    T neg_alpha = -alpha;

    // x = x + alpha*p.
    // r = r - alpha*q.
    axpy(handle_b, n, &alpha, p, 1, x,  1);
    axpy(handle_b, m, &neg_alpha, q, 1, r,  1);

    // s = A'*r - shift*x.
    cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);
    if (F == CSR)
      spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, n, m, nnz, &kOne,
          descr, val_at, ptr_at, ind_at, r, &kNegShift, s);
    else
      spmv<T, CSR>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, n, m, nnz, &kOne,
          descr, val_a, ptr_a, ind_a, r, &kNegShift, s);
    cudaDeviceSynchronize();

    // Compute beta.
    nrm2(n, s, &norms);
    cudaDeviceSynchronize();
    T gamma1 = gamma;
    gamma = norms * norms;
    T beta = gamma / gamma1;

    // p = s + beta*p.
    axpy(handle_b, n, &beta, p, 1, s, 1);
    cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);

    // Convergence check.
    nrm2(n, x, &normx);
    xmax = std::max(xmax, normx);
    bool converged = (norms <= norms0 * tol) || (normx * tol >= 1);
    if (!quiet && (converged || k % 10 == 0))
      printf(fmt, k, normx, norms / norms0);
    if (converged)
      break;
  }

  // Determine exit status.
  T shrink = normx / xmax;
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
  return flag;
}

// CGLS, with pre-initialized cusparseHandle and cublasHandle.
template <typename T, CGLS_ORD F>
INT solve(cusparseMatDescr_t descr, const T *val_a, const INT *ptr_a,
          const INT *ind_a, const T *val_at, const INT *ptr_at,
          const INT *ind_at, const INT m, const INT n, const INT nnz,
          const T *b, T *x, const T shift, const T tol, const INT maxit,
          bool quiet) {
  cusparseHandle_t handle_s;
  cublasHandle_t handle_b;
  cusparseCreate(&handle_s);
  cublasCreate(&handle_b);
  int flag = solve<T, F>(handle_s, handle_b, descr, val_a, ptr_a, ind_a, val_at,
      ptr_at, ind_at, m, n, nnz, b, x, shift, tol, maxit, quiet);
  cusparseDestroy(handle_s);
  cublasDestroy(handle_b);
  return flag;
}

// CGLS, with pre-initialized cusparseMatDescr, cusparseHandle and cublasHandle.
template <typename T, CGLS_ORD F>
INT solve(const T *val_a, const INT *ptr_a, const INT *ind_a, const T *val_at,
          const INT *ptr_at, INT *ind_at, const INT m, const INT n,
          const INT nnz, const T *b, T *x, const T shift, const T tol,
          const INT maxit, bool quiet) {
  cusparseHandle_t handle_s;
  cublasHandle_t handle_b;
  cusparseMatDescr_t descr;
  cusparseCreate(&handle_s);
  cublasCreate(&handle_b);
  cusparseCreateMatDescr(&descr);
  int flag = solve<T, F>(handle_s, handle_b, descr, val_a, ptr_a, ind_a, val_at,
      ptr_at, ind_at, m, n, nnz, b, x, shift, tol, maxit, quiet);
  cusparseDestroy(handle_s);
  cublasDestroy(handle_b);
  cusparseDestroyMatDescr(descr);
  return flag;
}

}  // namespace cgls

#endif  // CGLS_CUH_

