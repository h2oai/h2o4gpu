#ifndef PROX_LIB_HPP_
#define PROX_LIB_HPP_

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif

// List of functions supported by the proximal operator library.
enum Function { kAbs,       // f(x) = |x|
                kEntr,      // f(x) = x log(x)
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
                kNegLog,    // f(x) = -log(x)
                kRecipr,    // f(x) = 1/x
                kSquare,    // f(x) = (1/2) x^2
                kZero };    // f(x) = 0

// Object associated with the generic function c * f(a * x - b) + d * x.
// Parameters a and c default to 1, while b and d default to 0.
template <typename T>
struct FunctionObj {
  Function f;
  T a, b, c, d;

  FunctionObj(Function f, T a, T b, T c, T d)
      : f(f), a(a), b(b), c(c), d(d) { check_c(); }
  FunctionObj(Function f, T a, T b, T c)
      : f(f), a(a), b(b), c(c), d(static_cast<T>(0)) { check_c(); }
  FunctionObj(Function f, T a, T b)
      : f(f), a(a), b(b), c(static_cast<T>(1)),
        d(static_cast<T>(0)) { }
  FunctionObj(Function f, T a)
      : f(f), a(a), b(static_cast<T>(0)), c(static_cast<T>(1)),
        d(static_cast<T>(0)) { }
  explicit FunctionObj(Function f)
      : f(f), a(static_cast<T>(1)), b(static_cast<T>(0)), c(static_cast<T>(1)),
        d(static_cast<T>(0)) { }

  void check_c() {
    if (c < static_cast<T>(0)) {
      fprintf(stderr, "WARNING c < 0. Function not convex. Using c = 0");
      c = static_cast<T>(0);
    }
  }
};


// Local Functions.
namespace {
// Evalution of max(0, x).
template <typename T>
__DEVICE__ inline T MaxPos(T x) {
  return fmax(static_cast<T>(0), x);
}

//  Evalution of max(0, -x).
template <typename T>
__DEVICE__ inline T MaxNeg(T x) {
  return fmax(static_cast<T>(0), -x);
}
}  // namespace


// Proximal operator definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and five parameters (a, b, c, d and rho)
// and returns the evaluation of
//
//   x -> Prox{c * f(a * x - b) + d * x},
//
// where Prox{.} is the proximal operator with penalty parameter rho.
template <typename T>
__DEVICE__ inline T ProxAbs(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = MaxPos(x_ - 1 / rho_) -
      MaxNeg(x_ + 1 / rho_);
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxEntr(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = 0 * rho_ * x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxExp(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = 0 * rho_ * x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxHuber(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = 0 * rho_ * x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxIdentity(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = x_ - 1 / rho_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxIndBox01(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T z = x_ <= 0 ? 0 : x_ >= 1 ? 1 : x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxIndEq0(T x, T a, T b, T c, T d, T rho) {
  return b / a;
}

template <typename T>
__DEVICE__ inline T ProxIndGe0(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T z = x_ <= 0 ? 0 : x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxIndLe0(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T z = x_ >= 0 ? 0 : x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxLogistic(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = 0 * rho_ * x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxMaxNeg0(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = x_ >= 0 ? x_ : 0;
  z = x_ + 1 / rho <= 0 ? x_ + 1 / rho_ : z;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxMaxPos0(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = x_ <= 0 ? x_ : 0;
  z = x_ >= 1 / rho_ ? x_ - 1 / rho_ : z;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxNegLog(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = (x_ + sqrt(x_ * x_ + 4 / rho_)) / 2;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxRecipr(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = 0 * rho_ * x_;
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxSquare(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = rho_ * x_ / (1 + rho_);
  return (z + b) / a;
}

template <typename T>
__DEVICE__ inline T ProxZero(T x, T a, T b, T c, T d, T rho) {
  return x - d / rho;
}

// Evaluates the proximal operator of f.
template <typename T>
__DEVICE__ inline T ProxEval(const FunctionObj<T> &f_obj, T x, T rho) {
  const T a = f_obj.a;
  const T b = f_obj.b;
  const T c = f_obj.c;
  const T d = f_obj.d;
  switch (f_obj.f) {
    case kAbs:
      return ProxAbs(x, a, b, c, d, rho);
    case kEntr:
      return ProxEntr(x, a, b, c, d, rho);
    case kExp:
      return ProxExp(x, a, b, c, d, rho);
    case kHuber:
      return ProxHuber(x, a, b, c, d, rho);
    case kIdentity:
      return ProxIdentity(x, a, b, c, d, rho);
    case kIndBox01:
      return ProxIndBox01(x, a, b, c, d, rho);
    case kIndEq0:
      return ProxIndEq0(x, a, b, c, d, rho);
    case kIndGe0:
      return ProxIndGe0(x, a, b, c, d, rho);
    case kIndLe0:
      return ProxIndLe0(x, a, b, c, d, rho);
    case kLogistic:
      return ProxLogistic(x, a, b, c, d, rho);
    case kMaxNeg0:
      return ProxMaxNeg0(x, a, b, c, d, rho);
    case kMaxPos0:
      return ProxMaxPos0(x, a, b, c, d, rho);
    case kNegLog:
      return ProxNegLog(x, a, b, c, d, rho);
    case kRecipr:
      return ProxRecipr(x, a, b, c, d, rho);
    case kSquare:
      return ProxSquare(x, a, b, c, d, rho);
    case kZero: default:
      return ProxZero(x, a, b, c, d, rho);
  }
}


// Function definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and four parameters (a, b, c, and d)
// and returns the evaluation of
//
//   x -> c * f(a * x - b) + d * x.
template <typename T>
__DEVICE__ inline T FuncAbs(T x, T a, T b, T c, T d) {
  return c * fabs(a * x + b) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncEntr(T x, T a, T b, T c, T d) {
  T x_ = a * x + b;
  return c * x_ * log(x_) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncExp(T x, T a, T b, T c, T d) {
  return c * exp(a * x + b) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncHuber(T x, T a, T b, T c, T d) {
  T xabs = fabs(a * x - b);
  return xabs < static_cast<T>(1) ? c * xabs * xabs + d * x : c * xabs + d * x;
}

template <typename T>
__DEVICE__ inline T FuncIdentity(T x, T a, T b, T c, T d) {
  return c * (a * x + b) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncIndBox01(T x, T a, T b, T c, T d) {
  return d * x;
}

template <typename T>
__DEVICE__ inline T FuncIndEq0(T x, T a, T b, T c, T d) {
  return d * x;
}

template <typename T>
__DEVICE__ inline T FuncIndGe0(T x, T a, T b, T c, T d) {
  return d * x;
}

template <typename T>
__DEVICE__ inline T FuncIndLe0(T x, T a, T b, T c, T d) {
  return d * x;
}

template <typename T>
__DEVICE__ inline T FuncLogistic(T x, T a, T b, T c, T d) {
  return c * log(static_cast<T>(1) + exp(a * x + b)) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncMaxNeg0(T x, T a, T b, T c, T d) {
  return c * MaxNeg(a * x - b) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncMaxPos0(T x, T a, T b, T c, T d) {
  return c * MaxPos(a * x - b) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncNegLog(T x, T a, T b, T c, T d) {
  return -c * log(a * x - b) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncRecpr(T x, T a, T b, T c, T d) {
  return 1 / x;
}

template <typename T>
__DEVICE__ inline T FuncSquare(T x, T a, T b, T c, T d) {
  T sq = (a * x - b);
  return c * sq * sq / static_cast<T>(2) + d * x;
}

template <typename T>
__DEVICE__ inline T FuncZero(T x, T a, T b, T c, T d) {
  return d * x;
}

// Evaluates the function f.
template <typename T>
__DEVICE__ inline T FuncEval(const FunctionObj<T> &f_obj, T x) {
  const T a = f_obj.a;
  const T b = f_obj.b;
  const T c = f_obj.c;
  const T d = f_obj.d;
  switch (f_obj.f) {
    case kAbs:
      return FuncAbs(x, a, b, c, d);
    case kEntr:
      return FuncEntr(x, a, b, c, d);
    case kExp:
      return FuncExp(x, a, b, c, d);
    case kHuber:
      return FuncHuber(x, a, b, c, d);
    case kIdentity:
      return FuncIdentity(x, a, b, c, d);
    case kIndBox01:
      return FuncIndBox01(x, a, b, c, d);
    case kIndEq0:
      return FuncIndEq0(x, a, b, c, d);
    case kIndGe0:
      return FuncIndGe0(x, a, b, c, d);
    case kIndLe0:
      return FuncIndLe0(x, a, b, c, d);
    case kLogistic:
      return FuncLogistic(x, a, b, c, d);
    case kMaxNeg0:
      return FuncMaxNeg0(x, a, b, c, d);
    case kMaxPos0:
      return FuncMaxPos0(x, a, b, c, d);
    case kNegLog:
      return FuncNegLog(x, a, b, c, d);
    case kRecipr:
      return FuncRecpr(x, a, b, c, d);
    case kSquare:
      return FuncSquare(x, a, b, c, d);
    case kZero: default:
      return FuncZero(x, a, b, c, d);
  }
}


// Evaluates the proximal operator Prox{f_obj[i]}(x_in[i]) -> x_out[i].
//
// @param f_obj Vector of function objects.
// @param rho Penalty parameter.
// @param x_in Array to which proximal operator will be applied.
// @param x_out Array to which result will be written.
template <typename T>
void ProxEval(const std::vector<FunctionObj<T> > &f_obj, T rho, const T* x_in,
              T* x_out) {
  //  #pragma omp parallel for
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
  //  #pragma omp parallel for reduction(+:sum)
  for (unsigned int i = 0; i < f_obj.size(); ++i)
    sum += FuncEval(f_obj[i], x_in[i]);
  return sum;
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
                    thrust::device_pointer_cast(x_in),
                    thrust::device_pointer_cast(x_out), ProxEvalF<T>(rho));
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
                               thrust::device_pointer_cast(x_in),
                               static_cast<T>(0), thrust::plus<T>(),
                               FuncEvalF<T>());
}
#endif /* __CUDACC__ */

#endif /* PROX_LIB_HPP_ */

