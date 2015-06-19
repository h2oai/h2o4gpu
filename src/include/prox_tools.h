#ifndef PROX_TOOLS_H_
#define PROX_TOOLS_H_

#include <cmath>

#ifdef __CUDACC__
#define __DEVICE__ __device__ __host__
#else
#define __DEVICE__
#endif

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

// Evalution of min(0, x).
template <typename T>
__DEVICE__ inline T MinPos(T x) {
  return Min(static_cast<T>(0), x);
}

//  Evalution of max(0, -x).
template <typename T>
__DEVICE__ inline T MaxNeg(T x) {
  return Max(static_cast<T>(0), -x);
}

//  Evalution of min(0, -x).
template <typename T>
__DEVICE__ inline T MinNeg(T x) {
  return Min(static_cast<T>(0), -x);
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

#endif  // PROX_TOOLS_H_
