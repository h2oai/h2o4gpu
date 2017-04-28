#ifndef EQUIL_HELPER_H_
#define EQUIL_HELPER_H_

#include <algorithm>
#include <cmath>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_rand.h"
#include "gsl/gsl_vector.h"
#include "matrix/matrix.h"
#include "util.h"

namespace h2oaiglm {
namespace {

// Different norm types.
enum NormTypes { kNorm1, kNorm2, kNormFro };

// TODO: Figure out a better value for this constant
const double kSinkhornConst        = 1e-8;
const double kNormEstTol           = 1e-3;
const unsigned int kEquilIter      = 50u;
const unsigned int kNormEstMaxIter = 50u;

////////////////////////////////////////////////////////////////////////////////
///////////////////////// Helper Functions /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ReciprF : std::unary_function<T, T> {
  T alpha;
  ReciprF() : alpha(1) { }
  ReciprF(T alpha) : alpha(alpha) { }
  T operator()(T x) { return alpha / x; }
};

template <typename T>
struct AbsF : std::unary_function<T, T> {
  inline double Abs(double x) { return fabs(x); }
  inline float Abs(float x) { return fabsf(x); }
  T operator()(T x) { return Abs(x); }
};

template <typename T>
struct IdentityF : std::unary_function<T, T> {
  T operator()(T x) { return x; }
};

template <typename T>
struct SquareF: std::unary_function<T, T> {
  T operator()(T x) { return x * x; }
};

template <typename T>
struct SqrtF : std::unary_function<T, T> {
  inline double Sqrt(double x) { return sqrt(x); }
  inline float Sqrt(float x) { return sqrtf(x); }
  T operator()(T x) { return Sqrt(x); }
};

template <typename T, typename F>
void SetSign(T* x, unsigned char *sign, size_t size, F f) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int t = 0; t < size; ++t) {
    sign[t] = 0;
    for (unsigned int i = 0; i < 8; ++i) {
      sign[t] |= static_cast<unsigned char>((x[8 * t + i] < 0) << i);
      x[8 * t + i] = f(x[8 * t + i]);
    }
  }
}

template <typename T, typename F>
void SetSignSingle(T* x, unsigned char *sign, size_t bits, F f) {
  sign[0] = 0;
  for (unsigned int i = 0; i < bits; ++i) {
    sign[0] |= static_cast<unsigned char>((x[i] < 0) << i);
    x[i] = f(x[i]);
  }
}

template <typename T, typename F>
void UnSetSign(T* x, unsigned char *sign, size_t size, F f) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int t = 0; t < size; ++t) {
    for (unsigned int i = 0; i < 8; ++i) {
      x[8 * t + i] = (1 - 2 * static_cast<int>((sign[t] >> i) & 1)) *
          f(x[8 * t + i]);
    }
  }
}

template <typename T, typename F>
void UnSetSignSingle(T* x, unsigned char *sign, size_t bits, F f) {
  for (unsigned int i = 0; i < bits; ++i)
    x[i] = (1 - 2 * static_cast<int>((sign[0] >> i) & 1)) * f(x[i]);
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////// Norm Estimation //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
T Norm2Est(const Matrix<T> *A) {
  // Same as MATLAB's method for norm estimation.

  T kTol = static_cast<T>(kNormEstTol);

  T norm_est = 0, norm_est_last;
  gsl::vector<T> x = gsl::vector_calloc<T>(A->Cols());
  gsl::vector<T> Sx = gsl::vector_calloc<T>(A->Rows());
  gsl::rand(x.data, x.size);

  unsigned int i = 0;
  for (i = 0; i < kNormEstMaxIter; ++i) {
    norm_est_last = norm_est;
    A->Mul('n', static_cast<T>(1.), x.data, static_cast<T>(0.), Sx.data);
    A->Mul('t', static_cast<T>(1.), Sx.data, static_cast<T>(0.), x.data);
    T normx = gsl::blas_nrm2(&x);
    T normSx = gsl::blas_nrm2(&Sx);
    gsl::vector_scale(&x, 1 / normx);
    norm_est = normx / normSx;
    if (std::abs(norm_est_last - norm_est) < kTol * norm_est)
      break;
  }
  DEBUG_EXPECT_LT(i, kNormEstMaxIter);

  gsl::vector_free(&x);
  gsl::vector_free(&Sx);
  return norm_est;
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////// Modified Sinkhorn Knopp //////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void SinkhornKnopp(const Matrix<T> *A, T *d, T *e, bool equillocal) {
  gsl::vector<T> d_vec = gsl::vector_view_array<T>(d, A->Rows());
  gsl::vector<T> e_vec = gsl::vector_view_array<T>(e, A->Cols());
  gsl::vector_set_all(&d_vec, static_cast<T>(1.));
  gsl::vector_set_all(&e_vec, static_cast<T>(1.));

  if(!equillocal) return;

  for (unsigned int k = 0; k < kEquilIter; ++k) {
    // e := 1 ./ (A' * d).
    A->Mul('t', static_cast<T>(1.), d, static_cast<T>(0.), e);
    gsl::vector_add_constant(&e_vec,
        static_cast<T>(kSinkhornConst) * (A->Rows() + A->Cols()) / A->Rows());
    std::transform(e, e + e_vec.size, e, ReciprF<T>(A->Rows()));

    // d := 1 ./ (A' * e).
    A->Mul('n', static_cast<T>(1.), e, static_cast<T>(0.), d);
    gsl::vector_add_constant(&d_vec,
        static_cast<T>(kSinkhornConst) * (A->Rows() + A->Cols()) / A->Cols());
    std::transform(d, d + d_vec.size, d, ReciprF<T>(A->Cols()));
  }
}

}  // namespace
}  // namespace h2oaiglm

#endif  // EQUIL_HELPER_H_

