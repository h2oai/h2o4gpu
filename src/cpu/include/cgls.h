/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
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

#ifndef CGLS_H_
#define CGLS_H_

#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <complex>
#include <limits>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "util.h"

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
inline std::complex<double> StaticCast<std::complex<double> >(double x) {
 return x;
}

template <>
inline std::complex<float> StaticCast<std::complex<float> >(double x) {
 return static_cast<float>(x);
}

// Numeric limit epsilon for float, double, complex_float, and complex_double.
template <typename T>
double Epsilon();

template<>
inline double Epsilon<double>() {
  return std::numeric_limits<double>::epsilon();
}

template<>
inline double Epsilon<std::complex<double>	>() {
  return std::numeric_limits<double>::epsilon();
}

template<>
inline double Epsilon<float>() {
  return std::numeric_limits<float>::epsilon();
}

template<>
inline double Epsilon<std::complex<float> >() {
  return std::numeric_limits<float>::epsilon();
}

}  // namespace

// Conjugate Gradient Least Squares.
template <typename T, typename F>
int Solve(const F& A, const INT m, const INT n, const T *b, T *x,
          const double shift, const double tol, const int maxit, bool quiet) {
  // Variable declarations.
  gsl::vector<T> p, q, r, s, x_vec;
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
  p = gsl::vector_calloc<T>(n);
  q = gsl::vector_calloc<T>(m);
  r = gsl::vector_calloc<T>(m);
  s = gsl::vector_calloc<T>(n);

  gsl::vector_memcpy(&r, b);
  gsl::vector_memcpy(&s, x);

  // Make x a gsl vector.
  x_vec = gsl::vector_view_array(x, n);

  // r = b - A*x.
  normx = gsl::blas_nrm2(&x_vec);
  if (normx > 0.) {
    err = A('n', kNegOne, x_vec.data, kOne, r.data);
    if (err)
      flag = 5;
  }

  // s = A'*r - shift*x.
  err = A('t', kOne, r.data, kNegShift, s.data);
  if (err)
    flag = 6;

  // Initialize.
  gsl::vector_memcpy(&p, &s);
  norms = gsl::blas_nrm2(&s);
  norms0 = norms;
  gamma = norms0 * norms0;
  normx = gsl::blas_nrm2(&x_vec);
  xmax = normx;

  if (norms < kEps)
    flag = 1;

  if (!quiet)
    printf("    k     normx        resNE\n");

  for (k = 0; k < maxit && !flag; ++k) {
    // q = A * p.
    err = A('n', kOne, p.data, kZero, q.data);
    if (err) {
      flag = 5;
      break;
    }

    // delta = norm(p)^2 + shift*norm(q)^2.
    normp = gsl::blas_nrm2(&p);
    normq = gsl::blas_nrm2(&q);
    double delta = normq * normq + shift * normp * normp;

    if (delta <= 0.)
      indefinite = 1;
    if (delta == 0.)
      delta = kEps;
    T alpha = StaticCast<T>(gamma / delta);
    T neg_alpha = StaticCast<T>(-gamma / delta);

    // x = x + alpha*p.
    // r = r - alpha*q.
    gsl::blas_axpy(alpha, &p, &x_vec);
    gsl::blas_axpy(neg_alpha, &q, &r);

    // s = A'*r - shift*x.
    gsl::vector_memcpy(&s, &x_vec);
    err = A('t', kOne, r.data, kNegShift, s.data);
    if (err) {
      flag = 6;
      break;
    }

    // Compute beta.
    norms = gsl::blas_nrm2(&s);
    double gamma1 = gamma;
    gamma = norms * norms;
    T beta = StaticCast<T>(gamma / gamma1);

    // p = s + beta*p.
    gsl::blas_axpy(beta, &p, &s);
    gsl::vector_memcpy(&p, &s);

    // Convergence check.
    normx = gsl::blas_nrm2(&x_vec);
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
  gsl::vector_free(&p);
  gsl::vector_free(&q);
  gsl::vector_free(&r);
  gsl::vector_free(&s);
  return flag;
}

}  // namespace cgls

#endif  // CGLS_H_

