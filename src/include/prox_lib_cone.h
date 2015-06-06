#ifndef PROX_LIB_CONE_H_
#define PROX_LIB_CONE_H_

#include <assert.h>

#include <vector>

#ifdef __CUDACC__
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif

typedef unsigned int CONE_IDX;

enum Cone { kConeZero,       // { x : x = 0 }
            kConeNonNeg,     // { x : x >= 0 }
            kConeNonPos,     // { x : x <= 0 }
            kConeSoc,        // { (p, x) : ||x||_2 <= p }
            kConeSdp,        // { X : X >= 0 }
            kConeExpPrimal,  // { (x, y, z) : y > 0, y e^(x/y) <= z }
            kConeExpDual };  // { (u, v, w) : u < 0, -u e^(v/u) <= ew }

struct ConeConstraint {
  Cone cone;
  std::vector<CONE_IDX> idx;
  ConeConstraint(Cone cone, const std::vector<CONE_IDX>& idx)
      : cone(cone), idx(idx) { };
};

struct ConeConstraintRaw {
  Cone cone;
  CONE_IDX *idx;
  CONE_IDX size;
};

inline bool IsSeparable(Cone cone) {
  if (cone == kConeZero || cone == kConeNonNeg || cone == kConeNonPos)
    return true;
  return false;
}

// Shared GPU/CPU code.
namespace {
const double kExp1 = 2.718281828459045;

template <typename T>
__DEVICE__ void ProjectExpPrimalCone(const CONE_IDX *idx, T *v) {
  T &x = v[idx[0]];
  T &y = v[idx[1]];
  T &z = v[idx[2]];
  if (x > 0 && x * Exp(y / z) <= -kExp1 * z) {
    x = y = z = static_cast<T>(0);
  } else if (x < 0 && y < 0) {
    x = static_cast<T>(0);
    y = Max(static_cast<T>(0), y);
  } else if (!(y > 0 && y * Exp(x / y) <= z)) {
    assert(false && "TODO");
  }
}

template <typename T>
__DEVICE__ void ProjectExpDualCone(const CONE_IDX *idx, T *v) {
  assert(false && "TODO");
}
}  // namespace

// CPU code.
namespace {
template <typename T, typename F>
void ApplyCpu(const F& f, const ConeConstraintRaw& cone_constr, T *v) {
  for (CONE_IDX i = 0; i < cone_constr.size; ++i)
    v[cone_constr.idx[i]] = f(v[cone_constr.idx[i]]);
}
}  // namespace

template <typename T>
inline void ProxConeZeroCpu(const ConeConstraintRaw& cone_constr, T *v) {
  auto f = [](int) { return static_cast<T>(0); };
  ApplyCpu(f, cone_constr, v);
}

template <typename T>
inline void ProxConeNonNegCpu(const ConeConstraintRaw& cone_constr, T *v) {
  auto f = [](T x) { return std::max(x, static_cast<T>(0)); };
  ApplyCpu(f, cone_constr, v);
}

template <typename T>
inline void ProxConeNonPosCpu(const ConeConstraintRaw& cone_constr, T *v) {
  auto f = [](T x) { return std::min(x, static_cast<T>(0)); };
  ApplyCpu(f, cone_constr, v);
}

template <typename T>
inline void ProxConeSocCpu(const ConeConstraintRaw& cone_constr, T *v) {
  T nrm = static_cast<T>(0);
  for (CONE_IDX i = 1; i < cone_constr.size; ++i)
    nrm += v[cone_constr.idx[i]] * v[cone_constr.idx[i]];
  nrm = std::sqrt(nrm);

  T p = v[cone_constr.idx[0]];
  if (nrm <= -p) {
    auto f = [](T) { return static_cast<T>(0); };
    ApplyCpu(f, cone_constr, v);
  } else if (nrm >= std::abs(p)) {
    T scale = (static_cast<T>(1) + p / nrm) / 2;
    v[cone_constr.idx[0]] = nrm;
    auto f = [scale](T x) { return scale * x; };
    ApplyCpu(f, cone_constr, v);
  }
}

template <typename T>
inline void ProxConeSdpCpu(const ConeConstraintRaw& cone_constr, T *v) {
  assert(false && "SDP Not implemented on CPU");
}


template <typename T>
inline void ProxConeExpPrimalCpu(const ConeConstraintRaw& cone_constr, T *v) {
  ProjectExpPrimalCone(cone_constr.idx, v);
}

template <typename T>
inline void ProxConeExpDualCpu(const ConeConstraintRaw& cone_constr, T *v) {
  ProjectExpDualCone(cone_constr.idx, v);
}

template <typename T>
void ProxEvalConeCpu(const std::vector<ConeConstraintRaw>& cone_constr_vec,
                     CONE_IDX size, const T *x_in, T *x_out) {
  memcpy(x_out, x_in, size * sizeof(T));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (const auto& cone_constr : cone_constr_vec) {
    switch (cone_constr.cone) {
      case kConeZero: default: ProxConeZeroCpu(cone_constr, x_out); break;
      case kConeNonNeg: ProxConeNonNegCpu(cone_constr, x_out); break;
      case kConeNonPos: ProxConeNonPosCpu(cone_constr, x_out); break;
      case kConeSoc: ProxConeSocCpu(cone_constr, x_out); break;
      case kConeSdp: ProxConeSdpCpu(cone_constr, x_out); break;
      case kConeExpPrimal: ProxConeExpPrimalCpu(cone_constr, x_out); break;
      case kConeExpDual: ProxConeExpDualCpu(cone_constr, x_out); break;
    }
  }
}

// GPU code.
#ifdef __CUDACC__

namespace {
const CONE_IDX kBlockSize = 256u;
#if __CUDA_ARCH__ >= 300
const CONE_IDX kMaxGridSize = 65535u;  // 2^16 - 1
#else
const CONE_IDX kMaxGridSize = 2147483647u;  // 2^31 - 1
#endif

template <typename F>
__global__
void __Execute(const F& f) {
  f();
}

template <typename T, typename F>
__global__
void __Apply(const F& f, const CONE_IDX *idx, CONE_IDX size, T *v) {
  CONE_IDX tid = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 300
  for (CONE_IDX i = tid; i < size; i += gridDim.x * blockDim.x)
    v[idx[i]] = f(v[idx[i]]);
#else
  v[idx[tid]] = f(v[idx[tid]]);
#endif
}

template <typename T, typename F>
void inline ApplyGpu(const F& f, const ConeConstraintRaw& cone_constr, T *v,
                     int stream) {
  CONE_IDX grid_dim = std::min(kMaxGridSize,
      (cone_constr.size + kBlockSize - 1) / kBlockSize);
  __Apply<<<grid_dim, kBlockSize, 0, stream>>>(f, cone_constr.idx,
      cone_constr.size, v);
}
} // namespace

template <typename T>
inline void ProxConeZeroGpu(const ConeConstraintRaw& cone_constr, T *v,
                            int stream) {
  auto f = [](T) { return static_cast<T>(0); };
  ApplyGpu(f, cone_constr.idx, cone_constr.size, v, stream);
}

template <typename T>
inline void ProxConeNonNegGpu(const ConeConstraintRaw& cone_constr, T *v,
                              int stream) {
  auto f = [](T x) { return Max(static_cast<T>(0), x); };
  ApplyGpu(f, cone_constr.idx, cone_constr.size, v, stream);
}

template <typename T>
inline void ProxConeNonPosGpu(const ConeConstraintRaw& cone_constr, T *v,
                              int stream) {
  auto f = [](T x) { return Min(static_cast<T>(0), x); };
  ApplyGpu(f, cone_constr.idx, cone_constr.size, v, stream);
}

template <typename T>
inline void ProxConeSocGpu(const ConeConstraintRaw& cone_constr, T *v,
                           int stream) {
  // TODO: Use reduce that has stream option.
  // Compute nrm(v[1:end])
  auto square = [v](T i) { return v[i] * v[i]; };
  T nrm = thrust::transform_reduce(thrust::cuda::par.on(stream),
      thrust::device_pointer_cast(cone_constr.idx),
      thrust::device_pointer_cast(cone_constr.idx + cone_constr.size),
      square, static_cast<T>(0.), thrust::plus<T>());

  // Get p from GPU.
  CONE_IDX i;
  T p;
  cudaMemcpyAsync(&i, cone_constr.idx, sizeof(CONE_IDX),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&p, v + i, sizeof(T), cudaMemcpyDeviceToHost,
                  stream);

  // Project if ||x||_2 > p
  if (nrm <= -p) {
    auto f = [](T) { return static_cast<T>(0); };
    ApplyGpu(f, cone_constr, v, stream);
  } else if (nrm >= std::abs(p)) {
    T scale = (static_cast<T>(1) + p / nrm) / 2;
    cudaMemcpyAsync(v + i, &nrm, sizeof(T), cudaMemcpyHostToDevice, stream);
    auto f = [scale](T x) { return scale * x; };
    ApplyGpu(f, cone_constr, v, stream);
  }
}

template <typename T>
inline void ProxConeSdpGpu(const ConeConstraintRaw& cone_constr, T *v,
                           int stream) {
  assert(false && "SDP Not implemented on GPU");
}

template <typename T>
inline void ProxConeExpPrimalGpu(const ConeConstraintRaw& cone_constr, T *v,
                                 int stream) {
  auto f = [idx=cone_constr.idx, =v]() { ProjectExpPrimalCone(idx, v); }
  __Execute<<<1, 1, 0, stream>>>(f);
}

template <typename T>
inline void ProxConeExpDualGpu(const ConeConstraintRaw& cone_constr, T *v,
                               int stream) {
  auto f = [idx=cone_constr.idx, =v]() { ProjectExpDualCone(idx, v); }
  __Execute<<<1, 1, 0, stream>>>(f);
}

template <typename T>
void ProxEvalConeGpu(const std::vector<ConeConstraintRaw>& cone_constr_vec,
                     CONE_IDX size, const T *x_in, T *x_out) {
  cudaMemcpy(x_out, x_in, size * sizeof(T), cudaMemcpyDeviceToDevice);

  CONE_IDX i = 0;
  for (const auto& cone_constr : cone_constr_vec) {
    switch (cone_constr.cone) {
      case kConeZero: default: ProxConeZeroGpu(cone_constr, x_out, i); break;
      case kConeNonNeg: ProxConeNonNegGpu(cone_constr, x_out, i); break;
      case kConeNonPos: ProxConeNonNegGpu(cone_constr, x_out, i); break;
      case kConeSoc: ProxConeSocGpu(cone_constr, x_out, i); break;
      case kConeSdp: ProxConeSdpGpu(cone_constr, x_out, i); break;
      case kConeExpPrimal: ProxConeExpPrimalGpu(cone_constr, x_out, i); break;
      case kConeExpDual: ProxConeExpDualGpu(cone_constr, x_out, i); break;
    }
    ++i;
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();
}


#endif  // __CUDACC__

#endif  // PROX_LIB_CONE_H_

