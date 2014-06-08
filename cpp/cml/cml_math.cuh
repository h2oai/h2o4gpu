#ifndef CML_MATH_CUH_
#define CML_MATH_CUH_

#include <cublas_v2.h>

#include "cml_utils.cuh"

namespace cml {

template <typename T>
__device__ T math_sqrt(T x);

template <>
__device__ double math_sqrt(double x) {
  return sqrt(x);
}

template <>
__device__ float math_sqrt(float x) {
  return sqrtf(x);
}

template <typename T>
__device__ T math_rsqrt(T x);

template <>
__device__ double math_rsqrt(double x) {
  return rsqrt(x);
}

template <>
__device__ float math_rsqrt(float x) {
  return rsqrtf(x);
}

}  // namespace cml

#endif  // CML_MATH_CUH_

