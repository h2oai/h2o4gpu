#include "Arith.hpp"
#include "../../utils/GpuInfo.cuh"

namespace h2o4gpu {
namespace Matrix {

namespace kernel {

/*
 * Compute min value for each row.
 * @tparam T Numeric type of the data
 * @param _res The output matrix with shape m x 1
 * @param _val The input matrix with shape m x n
 */
template <typename T>
__global__ void row_min_sequential(kParam<T> _res, kParam<T> _val) {

  size_t idx = global_thread_idx();
  if (idx < _val.rows) {
    T min = std::numeric_limits<T>::max();
    for (size_t i = 0; i < _val.cols; ++i) {
      T value = _val.ptr[idx * _val.cols + i];
      if (value < min) {
        min = value;
      }
    }
    _res.ptr[idx] = min;
  }
}

template <typename T>
__global__ void row_argmin_sequential(kParam<int> _res, kParam<T> _val) {

  size_t idx = global_thread_idx();
  if (idx < _val.rows) {
    T min = std::numeric_limits<T>::max();
    int min_idx = -1;
    for (size_t i = 0; i < _val.cols; ++i) {
      T value = _val.ptr[idx * _val.cols + i];
      if (value < min) {
        min = value;
        min_idx = i;
      }
    }
    _res.ptr[idx] = min_idx;
  }
}

}  // namespace kernel

template <typename T>
void DotOp<T>::dot(KmMatrix<T>& _res, KmMatrix<T>& _val) {
  this->dot(_res, _val, _val);
}
template <typename T>
void DotOp<T>::dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs,
                   KmMatrix<T>& _rhs) {
  constexpr T alpha = 1.0;
  constexpr T beta = 1.0;
  cublasHandle_t handle = GpuInfo::ins().cublas_handle();
  Blas::gemm(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,  // FIXME
             _lhs.rows(), _rhs.cols(), _lhs.cols(),
             &alpha,
             _lhs.dev_ptr(), _lhs.cols(),
             _rhs.dev_ptr(), _rhs.cols(),
             &beta,
             _res.dev_ptr(), _res.cols());
}

template <typename T>
void VecBatchDotOp<T>::dot(KmMatrix<T>& _res, KmMatrix<T>& _val) {
  this->dot(_res, _val, _val);
}
template <typename T>
void VecBatchDotOp<T>::dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs, KmMatrix<T>& _rhs) {
  constexpr T alpha = 1.0;
  constexpr T beta = 1.0;
  cublasHandle_t handle = GpuInfo::ins().cublas_handle();
  Blas::gemm_strided_batched(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      1, 1, _rhs.cols(),  // m, n, k
      &alpha,
      _lhs.dev_ptr(), 1, _lhs.cols(),
      _rhs.dev_ptr(), 1, _rhs.cols(),
      &beta,
      _res.dev_ptr(), _res.cols(), 1,  // c should be columun vector
      _lhs.rows());
}

template <typename T>
T SumOp<T>::sum(KmMatrix<T>& _val) {
  T* raw_ptr = _val.dev_ptr();
  thrust::device_ptr<T> ptr (raw_ptr);
  T res = thrust::reduce(ptr, ptr + _val.size(), (T)0, thrust::plus<T>());
  return res;
}

template <typename T>
void MulOp<T>::mul(KmMatrix<T>& _res, KmMatrix<T>& _lhs, T _rhs) {
  cublasHandle_t handle = GpuInfo::ins().cublas_handle();
  Blas::axpy(
      handle, _lhs.size(),  // handle, n
      &_rhs,                // alpha
      _lhs.dev_ptr(), 1,
      _res.dev_ptr(), 1);
}

template <typename T>
T MeanOp<T>::mean(KmMatrix<T>& _val) {
  T res = SumOp<T>().sum(_val);
  res = res / _val.size();
  return res;
}

template <typename T>
KmMatrix<int> ArgMinOp<T>::argmin(KmMatrix<T>& _val, KmMatrixDim _dim) {
  if (_dim == KmMatrixDim::ROW) {
    KmMatrix<int> _res(_val.rows(), 1);
    kernel::row_argmin_sequential<<<div_roundup(_val.rows(), 256), 256>>>(
        _res.k_param(), _val.k_param());
    return _res;
  } else {
    // FIXME
    h2o4gpu_error("Not implemented");
    KmMatrix<int> res;
    return res;
  }
}

template <typename T>
KmMatrix<T> MinOp<T>::min(KmMatrix<T>& _val, KmMatrixDim _dim) {
  size_t blocks = GpuInfo::ins().blocks(32);
  if (_dim == KmMatrixDim::ROW) {
    KmMatrix<T> _res(_val.rows(), 1);
    kernel::row_min_sequential<<<div_roundup(_val.rows(), 256), 256>>>(
        _res.k_param(), _val.k_param());
    return _res;
  } else {
    // FIXME
    h2o4gpu_error("Not implemented");
    KmMatrix<T> res;
    return res;
  }
}

#define INSTANTIATE(T)                                                  \
  template void DotOp<T>::dot(KmMatrix<T>& _res, KmMatrix<T>& _val);    \
  template void DotOp<T>::dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs,     \
                              KmMatrix<T>& _rhs);                       \
  template void VecBatchDotOp<T>::dot(                                  \
      KmMatrix<T>& _res, KmMatrix<T>& _val);                            \
  template void VecBatchDotOp<T>::dot(                                  \
      KmMatrix<T>& _res, KmMatrix<T>& _lhs, KmMatrix<T>& _rhs);         \
  template T SumOp<T>::sum(KmMatrix<T>& _val);                          \
  template void MulOp<T>::mul(KmMatrix<T>& _res, KmMatrix<T>& _lhs, T _rhs); \
  template T MeanOp<T>::mean(KmMatrix<T>& _val);                        \
  template KmMatrix<int> ArgMinOp<T>::argmin(                           \
      KmMatrix<T>& _val, KmMatrixDim _dim);                             \
  template KmMatrix<T> MinOp<T>::min(KmMatrix<T>& _val, KmMatrixDim _dim); \


INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(int)

}      // namespace Matrix
}      // namespace h2o4gpu