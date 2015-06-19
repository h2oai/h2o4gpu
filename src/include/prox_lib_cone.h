#ifndef PROX_LIB_CONE_H_
#define PROX_LIB_CONE_H_

#include <assert.h>

#include <set>
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

#include "interface_defs.h"
#include "prox_tools.h"
#include "util.h"

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

inline bool ValidCone(const std::vector<ConeConstraint>& cones, size_t dim) {
  std::set<CONE_IDX> idx;
  for (const auto &cone : cones) {
    for (auto i : cone.idx) {
      if (idx.count(i) > 0) {
        Printf("ERROR: Cone index %d in multiple cones.\n", i);
        return false;
      }
      if (i >= dim) {
        Printf("ERROR: Cone index %d exceeds dimension of cone.\n", i);
        return false;
      }
      idx.insert(i);
    }
  }
  return true;
}

// Shared GPU/CPU code.
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
  if (x_in != x_out)
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

// Helper functions
const CONE_IDX kBlockSize = 256u;
#if __CUDA_ARCH__ >= 300
const CONE_IDX kMaxGridSize = 2147483647u;  // 2^31 - 1
#else
const CONE_IDX kMaxGridSize = 65535u;  // 2^16 - 1
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
  v[idx[tid]] = f(v[idx[tid]]);
#else
  for (CONE_IDX i = tid; i < size; i += gridDim.x * blockDim.x)
    v[idx[i]] = f(v[idx[i]]);
#endif
}

template <typename T, typename F>
void inline ApplyGpu(const F& f, const ConeConstraintRaw& cone_constr, T *v,
                     cudaStream_t stream) {
  CONE_IDX block_size = std::min<CONE_IDX>(kBlockSize, cone_constr.size);
  CONE_IDX grid_dim = std::min(kMaxGridSize,
      (cone_constr.size + block_size - 1) / block_size);
  __Apply<<<grid_dim, block_size, 0, stream>>>(f, cone_constr.idx,
      cone_constr.size, v);
}

// Functors
template <typename T>
struct Zero {
  __DEVICE__ T operator()(T) const { return static_cast<T>(0); }
};

template <typename T>
struct Max0 {
  __DEVICE__ T operator()(T x) const { return Max(static_cast<T>(0), x); }
};

template <typename T>
struct Min0 {
  __DEVICE__ T operator()(T x) const { return Min(static_cast<T>(0), x); }
};

template <typename T>
struct Square {
  __DEVICE__ T operator()(T x) const { return x * x; }
};

template <typename T>
struct Scale {
  T a;
  Scale(T a) : a(a) { }
  __DEVICE__ T operator()(T x) const { return a * x; }
};

// Proximal operators
template <typename T>
inline void ProxConeZeroGpu(const ConeConstraintRaw& cone_constr, T *v,
                            const cudaStream_t &stream) {
  ApplyGpu(Zero<T>(), cone_constr, v, stream);
}

// Prox
template <typename T>
inline void ProxConeNonNegGpu(const ConeConstraintRaw& cone_constr, T *v,
                              const cudaStream_t &stream) {
  ApplyGpu(Max0<T>(), cone_constr, v, stream);
}

template <typename T>
inline void ProxConeNonPosGpu(const ConeConstraintRaw& cone_constr, T *v,
                              const cudaStream_t &stream) {
  ApplyGpu(Min0<T>(), cone_constr, v, stream);
}

template <typename T>
inline void ProxConeSocGpu(const ConeConstraintRaw& cone_constr, T *v,
                           const cudaStream_t &stream) {
  // Compute nrm(v[1:end])
  T nrm = thrust::transform_reduce(thrust::cuda::par.on(stream),
      thrust::device_pointer_cast(cone_constr.idx),
      thrust::device_pointer_cast(cone_constr.idx + cone_constr.size),
      Square<T>(), static_cast<T>(0), thrust::plus<T>());

  // Get p from GPU.
  CONE_IDX i;
  T p;
  cudaMemcpyAsync(&i, cone_constr.idx, sizeof(CONE_IDX),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(v + i, &p, sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Project if ||x||_2 > p
  if (nrm <= -p) {
    ApplyGpu(Zero<T>(), cone_constr, v, stream);
  } else if (nrm >= std::abs(p)) {
    T scale = (static_cast<T>(1) + p / nrm) / 2;
    cudaMemcpyAsync(v + i, &nrm, sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    ApplyGpu(Scale<T>(scale), cone_constr, v, stream);
  }
}

template <typename T>
inline void ProxConeSdpGpu(const ConeConstraintRaw& cone_constr, T *v,
                           const cudaStream_t &stream) {
  assert(false && "SDP Not implemented on GPU");
}

template <typename T>
inline void ProxConeExpPrimalGpu(const ConeConstraintRaw& cone_constr, T *v,
                                 const cudaStream_t &stream) {
  // TODO
  // CONE_IDX *idx = cone_constr.idx;
  // auto f = [idx, v](){ ProjectExpPrimalCone(idx, v); };
  // __Execute<<<1, 1, 0, stream>>>(f);
}

template <typename T>
inline void ProxConeExpDualGpu(const ConeConstraintRaw& cone_constr, T *v,
                               const cudaStream_t &stream) {
  // TODO
  // CONE_IDX *idx = cone_constr.idx;
  // auto f = [idx, v]() { ProjectExpDualCone(idx, v); };
  // __Execute<<<1, 1, 0, stream>>>(f);
}

template <typename T>
void ProxEvalConeGpu(const std::vector<ConeConstraintRaw>& cone_constr_vec,
                     const std::vector<cudaStream_t> streams,
                     CONE_IDX size, const T *x_in, T *x_out) {
  cudaMemcpy(x_out, x_in, size * sizeof(T), cudaMemcpyDeviceToDevice);

  size_t idx = 0;
  for (const auto& cone_constr : cone_constr_vec) {
    const cudaStream_t& s = streams[idx++];
    switch (cone_constr.cone) {
      case kConeZero: default: ProxConeZeroGpu(cone_constr, x_out, s); break;
      case kConeNonNeg: ProxConeNonNegGpu(cone_constr, x_out, s); break;
      case kConeNonPos: ProxConeNonPosGpu(cone_constr, x_out, s); break;
      case kConeSoc: ProxConeSocGpu(cone_constr, x_out, s); break;
      case kConeSdp: ProxConeSdpGpu(cone_constr, x_out, s); break;
      case kConeExpPrimal: ProxConeExpPrimalGpu(cone_constr, x_out, s); break;
      case kConeExpDual: ProxConeExpDualGpu(cone_constr, x_out, s); break;
    }
  }
}


#endif  // __CUDACC__

#endif  // PROX_LIB_CONE_H_

