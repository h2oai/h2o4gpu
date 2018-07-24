#ifndef M_ARITH_HPP_
#define M_ARITH_HPP_

#include "KmMatrix.hpp"
#include "blas.cuh"
#include "utils.cuh"

namespace H2O4GPU {
namespace KMeans {

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
  size_t stride = grid_stride_x();

  size_t cols = _val.cols;

  for (size_t i = idx; i < _val.rows; i += stride) {
    T min = std::numeric_limits<T>::max();
    printf("cols outer: %u\n", cols);
    for (size_t j = 0; j < cols; ++j) {
      T tmp = _val.ptr[i+j];
      printf("i: %u, j: %u, tmp: %f, cols: %u\n", i, j, tmp, cols);
      if (tmp < min)
        min = tmp;
    }
    printf("i: %u, min: %f\n", i, min);
    _res.ptr[idx] = min;
  }
}

template <typename T>
__global__ void row_argmin_sequential(kParam<int> _res, kParam<T> _val) {

  size_t idx = global_thread_idx();
  size_t stride = grid_stride_x () * _val.cols;

  for (size_t i = idx; i < _val.size(); i += stride) {
    T min = std::numeric_limits<T>::max();
    int min_idx = -1;

    for (size_t j = 0; j < _val.cols; ++j) {
      T tmp = _val.ptr[i+j];
      if (tmp < min) {
        min_idx = i;
        min = tmp;
      }
    }

    _res.ptr[idx] = min_idx;
  }
}

}  // namespace kernel

// FIXME: Using struct for operations is just keeping the possibility of
// creating an unified operations for KmMatrix. For example, let KmMatrix
// inherit those left associative ops, or create an inferface for elementwise
// operations.

// FIXME: Use return value instead.
template <typename T>
struct DotOp {
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _val) {
    this->dot(_res, _val, _val);
  }
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs,
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
};

template <typename T>
struct VecBatchDotOp {
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _val) {
    this->dot(_res, _val, _val);
  }
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs, KmMatrix<T>& _rhs) {
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
};

template <typename T>
struct SumOp {
  T sum(KmMatrix<T>& _val) {
    T* raw_ptr = _val.dev_ptr();
    thrust::device_ptr<T> ptr (raw_ptr);
    T res = thrust::reduce(ptr, ptr + _val.size(), (T)0, thrust::plus<T>());
    return res;
  }
};

template <typename T>
struct MulOp {
  void mul(KmMatrix<T>& _res, KmMatrix<T>& _lhs, T _rhs) {
    cublasHandle_t handle = GpuInfo::ins().cublas_handle();
    Blas::axpy(
        handle, _lhs.size(),  // handle, n
        &_rhs,                // alpha
        _lhs.dev_ptr(), 1,
        _res.dev_ptr(), 1);
  }
};


template <typename T>
struct MeanOp {
  T mean(KmMatrix<T>& _val) {
    T res = SumOp<T>().sum(_val);
    res = res / _val.size();
    return res;
  }
};

template <typename T>
struct ArgMinOp {
  KmMatrix<int> argmin(KmMatrix<T>& _val, KmMatrixDim _dim) {
    size_t blocks = GpuInfo::ins().blocks(32);
    if (_dim == KmMatrixDim::ROW) {
      KmMatrix<int> _res(_val.rows(), 1);
      kernel::row_argmin_sequential<<<blocks, 256>>>(
          _res.k_param(), _val.k_param());
      return _res;
    } else {
      // FIXME
      M_ERROR("Not implemented");
    }
  }
};

template <typename T>
struct MinOp {

  KmMatrix<T> min(KmMatrix<T>& _val, KmMatrixDim _dim) {
    size_t blocks = GpuInfo::ins().blocks(32);
    if (_dim == KmMatrixDim::ROW) {
      KmMatrix<T> _res(_val.rows(), 1);
      kernel::row_min_sequential<<<blocks, 256>>>(_res.k_param(),
                                                  _val.k_param());
      return _res;
    } else {
      // FIXME
      M_ERROR("Not implemented");
    }
  }
};

}      // namespace KMenas
}      // namespace H204GPU

#endif  // M_ARITH_HPP_
