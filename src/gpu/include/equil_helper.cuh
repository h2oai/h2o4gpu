#ifndef EQUIL_HELPER_CUH_
#define EQUIL_HELPER_CUH_

#include <thrust/functional.h>

template <typename T>
struct ReciprF : thrust::unary_function<T, T> {
  T alpha;
  __host__ __device__ ReciprF() : alpha(1) { }
  __host__ __device__ ReciprF(T alpha) : alpha(alpha) { }
  __host__ __device__ T operator()(T x) { return alpha / x; }
};

template <typename T>
struct AbsF : thrust::unary_function<T, T> {
  __device__ inline double Abs(double x) { return fabs(x); }
  __device__ inline float Abs(float x) { return fabsf(x); }
  __device__ T operator()(T x) { return Abs(x); }
};

template <typename T>
struct IdentityF: thrust::unary_function<T, T> {
  __device__ T operator()(T x) { return x; }
};

template <typename T>
struct SquareF: thrust::unary_function<T, T> {
  __device__ T operator()(T x) { return x * x; }
};

template <typename T>
struct SqrtF : thrust::unary_function<T, T> {
  __device__ inline double Sqrt(double x) { return sqrt(x); }
  __device__ inline float Sqrt(float x) { return sqrtf(x); }
  __device__ T operator()(T x) { return Sqrt(x); }
};

template <typename T, typename F>
void __global__ __SetSign(T* x, unsigned char *sign, size_t size, F f) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x) {
    sign[t] = 0;
    for (unsigned int i = 0; i < 8; ++i) {
      sign[t] |= static_cast<unsigned char>(x[8 * t + i] < 0) << i; 
      x[8 * t + i] = f(x[8 * t + i]);
    }
  }
}

template <typename T, typename F>
void __global__ __SetSignSingle(T* x, unsigned char *sign, size_t bits, F f) {
  sign[0] = 0;
  for (unsigned int i = 0; i < bits; ++i) {
    sign[0] |= static_cast<unsigned char>(x[i] < 0) << i; 
    x[i] = f(x[i]);
  }
}

template <typename T, typename F>
void __global__ __UnSetSign(T* x, unsigned char *sign, size_t size, F f) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x) {
    for (unsigned int i = 0; i < 8; ++i) {
      x[8 * t + i] = (1 - 2 * static_cast<int>((sign[t] >> i) & 1)) *
          f(x[8 * t + i]);
    }
  }
}

template <typename T, typename F>
void __global__ __UnSetSignSingle(T* x, unsigned char *sign, size_t bits, F f) {
  for (unsigned int i = 0; i < bits; ++i)
    x[i] = (1 - 2 * static_cast<int>((sign[0] >> i) & 1)) * f(x[i]);
}

#endif  // EQUIL_HELPER_CUH_

