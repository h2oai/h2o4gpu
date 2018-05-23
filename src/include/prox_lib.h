/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef PROX_LIB_H_
#define PROX_LIB_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif

#include "interface_defs.h"

// List of functions supported by the proximal operator library.
enum Function { kAbs,       // f(x) = |x|
                kExp,       // f(x) = e^x
                kHuber,     // f(x) = huber(x)
                kIdentity,  // f(x) = x
                kIndBox01,  // f(x) = I(0 <= x <= 1)
                kIndEq0,    // f(x) = I(x = 0)
                kIndGe0,    // f(x) = I(x >= 0)
                kIndLe0,    // f(x) = I(x <= 0)
                kLogistic,  // f(x) = log(1 + e^x)
                kMaxNeg0,   // f(x) = max(0, -x)
                kMaxPos0,   // f(x) = max(0, x)
                kNegEntr,   // f(x) = x log(x)
                kNegLog,    // f(x) = -log(x)
                kRecipr,    // f(x) = 1/x
                kSquare,    // f(x) = (1/2) x^2
                kZero };    // f(x) = 0

// Object associated with the generic function c * f(a * x - b) + d * x + e * x * x.
// Parameters a and c default to 1, while b, d and e default to 0.
template <typename T>
struct FunctionObj {
  Function h;
  T a, b, c, d, e;

  FunctionObj(Function h, T a, T b, T c, T d, T e)
      : h(h), a(a), b(b), c(c), d(d), e(e) { CheckConsts(); }
  FunctionObj(Function h, T a, T b, T c, T d)
      : h(h), a(a), b(b), c(c), d(d), e(0) { CheckConsts(); }
  FunctionObj(Function h, T a, T b, T c)
      : h(h), a(a), b(b), c(c), d(0), e(0) { CheckConsts(); }
  FunctionObj(Function h, T a, T b)
      : h(h), a(a), b(b), c(1), d(0), e(0) { }
  FunctionObj(Function h, T a)
      : h(h), a(a), b(0), c(1), d(0), e(0) { }
  explicit FunctionObj(Function h)
      : h(h), a(1), b(0), c(1), d(0), e(0) { }
  FunctionObj()
      : h(kZero), a(1), b(0), c(1), d(0), e(0) { }

  void CheckConsts() {
    if (c < static_cast<T>(0))
      Printf("WARNING c < 0. Function not convex. Using c = 0");
    if (e < static_cast<T>(0))
      Printf("WARNING e < 0. Function not convex. Using e = 0");
    c = std::max(c, static_cast<T>(0));
    e = std::max(e, static_cast<T>(0));
  }
};


// Local Functions.
namespace {
//  Evaluate abs(x)
__DEVICE__ inline double Abs(double x) { return fabs(x); }
__DEVICE__ inline float Abs(float x) { return fabsf(x); }

//  Evaluate acos(x)
__DEVICE__ inline double Acos(double x) { return acos(x); }
__DEVICE__ inline float Acos(float x) { return acosf(x); }

//  Evaluate cos(x)
__DEVICE__ inline double Cos(double x) { return cos(x); }
__DEVICE__ inline float Cos(float x) { return cosf(x); }

//  Evaluate e^x
__DEVICE__ inline double Exp(double x) { return exp(x); }
__DEVICE__ inline float Exp(float x) { return expf(x); }

//  Evaluate log(x)
__DEVICE__ inline double Log(double x) { return log(x); }
__DEVICE__ inline float Log(float x) { return logf(x); }

//  Evaluate max(x, y)
__DEVICE__ inline double Max(double x, double y) { return fmax(x, y); }
__DEVICE__ inline float Max(float x, float y) { return fmaxf(x, y); }

//  Evaluate max(x, y)
__DEVICE__ inline double Min(double x, double y) { return fmin(x, y); }
__DEVICE__ inline float Min(float x, float y) { return fminf(x, y); }

//  Evaluate x^y
__DEVICE__ inline double Pow(double x, double y) { return pow(x, y); }
__DEVICE__ inline float Pow(float x, float y) { return powf(x, y); }

//  Evaluate sqrt(x)
__DEVICE__ inline double Sqrt(double x) { return sqrt(x); }
__DEVICE__ inline float Sqrt(float x) { return sqrtf(x); }

// Numeric Epsilon.
template <typename T>
__DEVICE__ inline T Epsilon();
template <>
__DEVICE__ inline double Epsilon<double>() { return 4e-16; }
template <>
__DEVICE__ inline float Epsilon<float>() { return 1e-7f; }

//  Evaluate tol
template <typename T>
__DEVICE__ inline T Tol();
template <>
__DEVICE__ inline double Tol() { return 1e-10; }
template <>
__DEVICE__ inline float Tol() { return 1e-5f; }

// Evalution of max(0, x).
template <typename T>
__DEVICE__ inline T MaxPos(T x) {
  return Max(static_cast<T>(0), x);
}

//  Evalution of max(0, -x).
template <typename T>
__DEVICE__ inline T MaxNeg(T x) {
  return Max(static_cast<T>(0), -x);
}

//  Evalution of sign(x)
template <typename T>
__DEVICE__ inline T Sign(T x) {
  return x >= 0 ? 1 : -1;
}

// LambertW(Exp(x))
// Evaluate the principal branch of the Lambert W function.
// ref: http://keithbriggs.info/software/LambertW.c
template <typename T>
__DEVICE__ inline T LambertWExp(T x) {
  T w;
  if (x > static_cast<T>(100)) {
    // Approximation for x in [100, 700].
    T log_x = Log(x);
    return static_cast<T>(-0.36962844)
        + x
        - static_cast<T>(0.97284858) * log_x
        + static_cast<T>(1.3437973) / log_x;
  } else if (x < static_cast<T>(0)) {
    T p = Sqrt(static_cast<T>(2.0) * (Exp(x + static_cast<T>(1)) + static_cast<T>(1)));
    w = static_cast<T>(-1.0)
       + p * (static_cast<T>(1.0)
           + p * (static_cast<T>(-1.0 / 3.0)
               + p * static_cast<T>(11.0 / 72.0)));
  } else {
    w = x;
  }
  if (x > static_cast<T>(1.098612288668110)) {
    w -= Log(w);
  }
  for (unsigned int i = 0u; i < 10u; i++) {
    T e = Exp(w);
    T t = w * e - Exp(x);
    T p = w + static_cast<T>(1.);
    t /= e * p - static_cast<T>(0.5) * (p + static_cast<T>(1.0)) * t / p;
    w -= t;
    if (Abs(t) < Epsilon<T>() * (static_cast<T>(1) + Abs(w)))
      break;
  }
  return w;
}

// Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root.
// ref: http://math.stackexchange.com/questions/60376
template <typename T>
__DEVICE__ inline T CubicSolve(T p, T q, T r) {
  T s = p / 3, s2 = s * s, s3 = s2 * s;
  T a = -s2 + q / 3;
  T b = s3 - s * q / 2 + r / 2;
  T a3 = a * a * a;
  T b2 = b * b;
  if (a3 + b2 >= 0) {
    T A = Pow(Sqrt(a3 + b2) - b, static_cast<T>(1) / 3);
    return -s - a / A + A;
  } else {
    T A = Sqrt(-a3);
    T B = Acos(-b / A);
    T C = Pow(A, static_cast<T>(1) / 3);
    return -s + (C - a / C) * Cos(B / 3);
  }
}
}  // namespace

// Proximal operator definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and five parameters (a, b, c, d and rho)
// and returns the evaluation of
//
//   x -> Prox{c * f(a * x - b) + d * x + e * x ^ 2},
//
// where Prox{.} is the proximal operator with penalty parameter rho.
template <typename T>
__DEVICE__ inline T ProxAbs(T v, T rho) {
  return MaxPos(v - 1 / rho) - MaxNeg(v + 1 / rho);
}

template <typename T>
__DEVICE__ inline T ProxNegEntr(T v, T rho) {
  // Use double precision.
  return static_cast<T>(
      LambertWExp<double>(
          static_cast<double>((rho * v - 1) + Log(rho)))) / rho;
}

template <typename T>
__DEVICE__ inline T ProxExp(T v, T rho) {
  return v - static_cast<T>(
      LambertWExp<double>(static_cast<double>(v - Log(rho))));
}

template <typename T>
__DEVICE__ inline T ProxHuber(T v, T rho) {
  return Abs(v) < 1 + 1 / rho ? v * rho / (1 + rho) : v - Sign(v) / rho;
}

template <typename T>
__DEVICE__ inline T ProxIdentity(T v, T rho) {
  return v - 1 / rho;
}

template <typename T>
__DEVICE__ inline T ProxIndBox01(T v, T rho) {
  return v <= 0 ? 0 : v >= 1 ? 1 : v;
}

template <typename T>
__DEVICE__ inline T ProxIndEq0(T v, T rho) {
  return 0;
}

template <typename T>
__DEVICE__ inline T ProxIndGe0(T v, T rho) {
  return v <= 0 ? 0 : v;
}

template <typename T>
__DEVICE__ inline T ProxIndLe0(T v, T rho) {
  return v >= 0 ? 0 : v;
}

template <typename T>
__DEVICE__ inline T ProxLogistic(T v, T rho) {
  // Initial guess based on piecewise approximation.
  T x;
  if (v < static_cast<T>(-2.5))
    x = v;
  else if (v > static_cast<T>(2.5) + 1 / rho)
    x = v - 1 / rho;
  else
    x = (rho * v - static_cast<T>(0.5)) / (static_cast<T>(0.2) + rho);

  // Newton iteration.
  T l = v - 1 / rho, u = v;
  for (unsigned int i = 0; i < 5; ++i) {
    T inv_ex = 1 / (1 + Exp(-x));
    T f = inv_ex + rho * (x - v);
    T g = inv_ex * (1 - inv_ex) + rho;
    if (f < 0)
      l = x;
    else
      u = x;
    x = x - f / g;
    x = Min(x, u);
    x = Max(x, l);
  }

  // Guarded method if not converged.
  for (unsigned int i = 0; u - l > Tol<T>() && i < 100; ++i) {
    T g_rho = 1 / (rho * (1 + Exp(-x))) + (x - v);
    if (g_rho > 0) {
      l = Max(l, x - g_rho);
      u = x;
    } else {
      u = Min(u, x - g_rho);
      l = x;
    }
    x = (u + l) / 2;
  }
  return x;
}

template <typename T>
__DEVICE__ inline T ProxMaxNeg0(T v, T rho) {
  T z = v >= 0 ? v : 0;
  return v + 1 / rho <= 0 ? v + 1 / rho : z;
}

template <typename T>
__DEVICE__ inline T ProxMaxPos0(T v, T rho) {
  T z = v <= 0 ? v : 0;
  return v >= 1 / rho ? v - 1 / rho : z;
}

template <typename T>
__DEVICE__ inline T ProxNegLog(T v, T rho) {
  return (v + Sqrt(v * v + 4 / rho)) / 2;
}

template <typename T>
__DEVICE__ inline T ProxRecipr(T v, T rho) {
  v = Max(v, static_cast<T>(0));
  return CubicSolve(-v, static_cast<T>(0), -1 / rho);
}

template <typename T>
__DEVICE__ inline T ProxSquare(T v, T rho) {
  return rho * v / (1 + rho);
}

template <typename T>
__DEVICE__ inline T ProxZero(T v, T rho) {
  return v;
}

#define SMALL 1E-30 // ok for float or double for this purpose

// Evaluates the proximal operator of f.
template <typename T>
__DEVICE__ inline T ProxEval(const FunctionObj<T> &f_obj, T v, T rho) {
  const T a = f_obj.a, b = f_obj.b, c = f_obj.c, d = f_obj.d, e = f_obj.e;
  v = a * (v * rho - d) / (SMALL + e + rho) - b;
  rho = (e + rho) / (SMALL + c * a * a); // Assumes c>=0 , as original paper assumes.  This is so weight can be 0.
  switch (f_obj.h) {
    case kAbs: v = ProxAbs(v, rho); break;
    case kNegEntr: v = ProxNegEntr(v, rho); break;
    case kExp: v = ProxExp(v, rho); break;
    case kHuber: v = ProxHuber(v, rho); break;
    case kIdentity: v = ProxIdentity(v, rho); break;
    case kIndBox01: v = ProxIndBox01(v, rho); break;
    case kIndEq0: v = ProxIndEq0(v, rho); break;
    case kIndGe0: v = ProxIndGe0(v, rho); break;
    case kIndLe0: v = ProxIndLe0(v, rho); break;
    case kLogistic: v = ProxLogistic(v, rho); break;
    case kMaxNeg0: v = ProxMaxNeg0(v, rho); break;
    case kMaxPos0: v = ProxMaxPos0(v, rho); break;
    case kNegLog: v = ProxNegLog(v, rho); break;
    case kRecipr: v = ProxRecipr(v, rho); break;
    case kSquare: v = ProxSquare(v, rho); break;
    case kZero: default: v = ProxZero(v, rho); break;
  }
  return (v + b) / (SMALL+a); // TODO: assumes a>=0, which is normal but not required by paper.
}


// Function definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and four parameters (a, b, c, and d)
// and returns the evaluation of
//
//   x -> c * f(a * x - b) + d * x.
template <typename T>
__DEVICE__ inline T FuncAbs(T x) {
  return Abs(x);
}

template <typename T>
__DEVICE__ inline T FuncNegEntr(T x) {
  return x <= 0 ? 0 : x * Log(x);
}

template <typename T>
__DEVICE__ inline T FuncExp(T x) {
  return Exp(x);
}

template <typename T>
__DEVICE__ inline T FuncHuber(T x) {
  T xabs = Abs(x);
  T xabs2 = xabs * xabs;
  return xabs < static_cast<T>(1) ? xabs2 / 2 : xabs - static_cast<T>(0.5);
}

template <typename T>
__DEVICE__ inline T FuncIdentity(T x) {
  return x;
}

template <typename T>
__DEVICE__ inline T FuncIndBox01(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncIndEq0(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncIndGe0(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncIndLe0(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncLogistic(T x) {
  return Log(1 + Exp(x));
}

template <typename T>
__DEVICE__ inline T FuncMaxNeg0(T x) {
  return MaxNeg(x);
}

template <typename T>
__DEVICE__ inline T FuncMaxPos0(T x) {
  return MaxPos(x);
}

template <typename T>
__DEVICE__ inline T FuncNegLog(T x) {
  x = Max(static_cast<T>(0), x);
  return -Log(x);
}

template <typename T>
__DEVICE__ inline T FuncRecpr(T x) {
  x = Max(static_cast<T>(0), x);
  return 1 / x;
}

template <typename T>
__DEVICE__ inline T FuncSquare(T x) {
  return x * x / 2;
}

template <typename T>
__DEVICE__ inline T FuncZero(T x) {
  return 0;
}

// Evaluates the function f.
template <typename T>
__DEVICE__ inline T FuncEval(const FunctionObj<T> &f_obj, T x) {
  T dx = f_obj.d * x;
  T ex = f_obj.e * x * x / 2;
  x = f_obj.a * x - f_obj.b;
  switch (f_obj.h) {
    case kAbs: x = FuncAbs(x); break;
    case kNegEntr: x = FuncNegEntr(x); break;
    case kExp: x = FuncExp(x); break;
    case kHuber: x = FuncHuber(x); break;
    case kIdentity: x = FuncIdentity(x); break;
    case kIndBox01: x = FuncIndBox01(x); break;
    case kIndEq0: x = FuncIndEq0(x); break;
    case kIndGe0: x = FuncIndGe0(x); break;
    case kIndLe0: x = FuncIndLe0(x); break;
    case kLogistic: x = FuncLogistic(x); break;
    case kMaxNeg0: x = FuncMaxNeg0(x); break;
    case kMaxPos0: x = FuncMaxPos0(x); break;
    case kNegLog: x = FuncNegLog(x); break;
    case kRecipr: x = FuncRecpr(x); break;
    case kSquare: x = FuncSquare(x); break;
    case kZero: default: x = FuncZero(x); break;
  }
  return f_obj.c * x + dx + ex;
}


// Projection onto subgradient definitions
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and five parameters (a, b, c, d, and e)
// and returns the evaluation of
//
//   x -> ProjSubgrad{c * f(a * x - b) + d * x + (1/2) e * x ^ 2},
//
// where ProjSubgrad{.} is the projection  onto the subgradient of the function.
template <typename T>
__DEVICE__ inline T ProjSubgradAbs(T v, T x) {
  if (x < static_cast<T>(0.))
    return static_cast<T>(-1.);
  else if (x > static_cast<T>(0.))
    return static_cast<T>(1.);
 else
    return Max(static_cast<T>(-1.), Min(static_cast<T>(1.), v));
}

template <typename T>
__DEVICE__ inline T ProjSubgradNegEntr(T v, T x) {
  return -Log(x) - static_cast<T>(1.);
}

template <typename T>
__DEVICE__ inline T ProjSubgradExp(T v, T x) {
  return Exp(x);
}

template <typename T>
__DEVICE__ inline T ProjSubgradHuber(T v, T x) {
  return Max(static_cast<T>(-1.), Min(static_cast<T>(1.), x));
}

template <typename T>
__DEVICE__ inline T ProjSubgradIdentity(T v, T x) {
  return static_cast<T>(1.);
}

template <typename T>
__DEVICE__ inline T ProjSubgradIndBox01(T v, T x) {
  if (x <= static_cast<T>(0.))
    return Min(static_cast<T>(0.), v);
  else if (x >= static_cast<T>(1.))
    return Max(static_cast<T>(0.), v);
  else
    return static_cast<T>(0.);
}

template <typename T>
__DEVICE__ inline T ProjSubgradIndEq0(T v, T x) {
  return v;
}

template <typename T>
__DEVICE__ inline T ProjSubgradIndGe0(T v, T x) {
  if (x <= static_cast<T>(0.))
    return Min(static_cast<T>(0.), v);
  else
    return static_cast<T>(0.);
}

template <typename T>
__DEVICE__ inline T ProjSubgradIndLe0(T v, T x) {
  if (x >= static_cast<T>(0.))
    return Max(static_cast<T>(0.), v);
  else
    return static_cast<T>(0.);
}

template <typename T>
__DEVICE__ inline T ProjSubgradLogistic(T v, T x) {
  return Exp(x) / (static_cast<T>(1.) + Exp(x));
}

template <typename T>
__DEVICE__ inline T ProjSubgradMaxNeg0(T v, T x) {
  if (x < static_cast<T>(0.))
    return static_cast<T>(-1.);
  else if (x > static_cast<T>(0.))
    return static_cast<T>(0.);
  else
    return Min(static_cast<T>(0.), Max(static_cast<T>(-1.), v));
}

template <typename T>
__DEVICE__ inline T ProjSubgradMaxPos0(T v, T x) {
  if (x < static_cast<T>(0.))
    return static_cast<T>(0.);
  else if (x > static_cast<T>(0.))
    return static_cast<T>(1.);
  else
    return Min(static_cast<T>(1.), Max(static_cast<T>(0.), v));
}

template <typename T>
__DEVICE__ inline T ProjSubgradNegLog(T v, T x) {
  return static_cast<T>(-1.) / x;
}

template <typename T>
__DEVICE__ inline T ProjSubgradRecipr(T v, T x) {
  return static_cast<T>(1.) / (x * x);
}

template <typename T>
__DEVICE__ inline T ProjSubgradSquare(T v, T x) {
  return x;
}

template <typename T>
__DEVICE__ inline T ProjSubgradZero(T v, T x) {
  return static_cast<T>(0.);
}

// Evaluates the projection of v onto the subgradient of f at x.
template <typename T>
__DEVICE__ inline T ProjSubgradEval(const FunctionObj<T> &f_obj, T v, T x) {
  const T a = f_obj.a, b = f_obj.b, c = f_obj.c, d = f_obj.d, e = f_obj.e;
  if (a == static_cast<T>(0.) || c == static_cast<T>(0.))
    return d + e * x;
  v = static_cast<T>(1.) / (a * c) * (v - d - e * x);
  T axb = a * x - b;
  switch (f_obj.h) {
    case kAbs: v = ProjSubgradAbs(v, axb); break;
    case kNegEntr: v = ProjSubgradNegEntr(v, axb); break;
    case kExp: v = ProjSubgradExp(v, axb); break;
    case kHuber: v = ProjSubgradHuber(v, axb); break;
    case kIdentity: v = ProjSubgradIdentity(v, axb); break;
    case kIndBox01: v = ProjSubgradIndBox01(v, axb); break;
    case kIndEq0: v = ProjSubgradIndEq0(v, axb); break;
    case kIndGe0: v = ProjSubgradIndGe0(v, axb); break;
    case kIndLe0: v = ProjSubgradIndLe0(v, axb); break;
    case kLogistic: v = ProjSubgradLogistic(v, axb); break;
    case kMaxNeg0: v = ProjSubgradMaxNeg0(v, axb); break;
    case kMaxPos0: v = ProjSubgradMaxPos0(v, axb); break;
    case kNegLog: v = ProjSubgradNegLog(v, axb); break;
    case kRecipr: v = ProjSubgradRecipr(v, axb); break;
    case kSquare: v = ProjSubgradSquare(v, axb); break;
    case kZero: default: v = ProjSubgradZero(v, axb); break;
  }
  return a * c * v + d + e * x;
}


// Evaluates the proximal operator Prox{f_obj[i]}(x_in[i]) -> x_out[i].
//
// @param f_obj Vector of function objects.
// @param rho Penalty parameter.
// @param x_in Array to which proximal operator will be applied.
// @param x_out Array to which result will be written.
template <typename T>
void ProxEval(const std::vector<FunctionObj<T> > &f_obj, T rho, const T *x_in,
              T *x_out) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < f_obj.size(); ++i)
    x_out[i] = ProxEval(f_obj[i], x_in[i], rho);
}


// Returns evalution of Sum_i Func{f_obj[i]}(x_in[i]).
//
// @param f_obj Vector of function objects.
// @param x_in Array to which function will be applied.
// @param x_out Array to which result will be written.
// @returns Evaluation of sum of functions.
template <typename T>
T FuncEval(const std::vector<FunctionObj<T> > &f_obj, const T* x_in) {
  T sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
  for (unsigned int i = 0; i < f_obj.size(); ++i)
    sum += FuncEval(f_obj[i], x_in[i]);
  return sum;
}

// Projection onto the subgradient at x_in
//   ProjSubgrad{f_obj[i]}(x_in[i], v_in[i]) -> x_out[i].
//
// @param f_obj Vector of function objects.
// @param x_in Array of points at which subgradient should be evaluated.
// @param v_in Array of points that should be projected onto the subgradient.
// @param v_out Array to which result will be written.
template <typename T>
void ProjSubgradEval(const std::vector<FunctionObj<T> > &f_obj, const T *x_in,
                     const T *v_in, T *v_out) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < f_obj.size(); ++i)
    v_out[i] = ProjSubgradEval(f_obj[i], v_in[i], x_in[i]);
}

#ifdef __CUDACC__
template <typename T>
struct ProxEvalF : thrust::binary_function<FunctionObj<T>, T, T> {
  T rho;
  __device__ ProxEvalF(T rho) : rho(rho) { }
  __device__ T operator()(const FunctionObj<T> &f_obj, T x) {
    return ProxEval(f_obj, x, rho);
  }
};

template <typename T>
void ProxEval(const thrust::device_vector<FunctionObj<T> > &f_obj, T rho,
              const T *x_in, T *x_out) {
  thrust::transform(thrust::device, f_obj.cbegin(), f_obj.cend(),
      thrust::device_pointer_cast(x_in), thrust::device_pointer_cast(x_out),
      ProxEvalF<T>(rho));
}

template <typename T>
struct FuncEvalF : thrust::binary_function<FunctionObj<T>, T, T> {
  __device__ T operator()(const FunctionObj<T> &f_obj, T x) {
    return FuncEval(f_obj, x);
  }
};

template <typename T>
T FuncEval(const thrust::device_vector<FunctionObj<T> > &f_obj, const T *x_in) {
  return thrust::inner_product(f_obj.cbegin(), f_obj.cend(),
      thrust::device_pointer_cast(x_in), static_cast<T>(0), thrust::plus<T>(),
      FuncEvalF<T>());
}

template <typename T>
struct ProjSubgradF {
  __device__ T operator()(const FunctionObj<T> &f_obj,
                          const thrust::tuple<T, T>& vx) {
    return ProjSubgradEval(f_obj, thrust::get<0>(vx), thrust::get<1>(vx));
  }
};

template <typename T>
void ProjSubgradEval(const thrust::device_vector<FunctionObj<T> > &f_obj,
                     const T *v_in, const T *x_in, T *v_out) {
  thrust::transform(thrust::device, f_obj.cbegin(), f_obj.cend(),
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::device_pointer_cast(v_in),
          thrust::device_pointer_cast(x_in))),
      thrust::device_pointer_cast(v_out), ProjSubgradF<T>());
}

#endif  // __CUDACC__

#endif  // PROX_LIB_H_

