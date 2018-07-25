#ifndef M_ARITH_HPP_
#define M_ARITH_HPP_

#include "KmMatrix.hpp"
#include "blas.cuh"
#include "utils.cuh"

namespace H2O4GPU {
namespace KMeans {

// FIXME: Using struct for operations is just keeping the possibility of
// creating an unified operations for KmMatrix. For example, let KmMatrix
// inherit those left associative ops, or create an inferface for elementwise
// operations.

// FIXME: Use return value instead.
template <typename T>
struct DotOp {
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _val);
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs, KmMatrix<T>& _rhs);
};

template <typename T>
struct VecBatchDotOp {
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _val);
  void dot(KmMatrix<T>& _res, KmMatrix<T>& _lhs, KmMatrix<T>& _rhs);
};

template <typename T>
struct SumOp {
  T sum(KmMatrix<T>& _val);
};

template <typename T>
struct MulOp {
  void mul(KmMatrix<T>& _res, KmMatrix<T>& _lhs, T _rhs);
};


template <typename T>
struct MeanOp {
  T mean(KmMatrix<T>& _val);
};

template <typename T>
struct ArgMinOp {
  KmMatrix<int> argmin(KmMatrix<T>& _val, KmMatrixDim _dim);
};

template <typename T>
struct MinOp {
  KmMatrix<T> min(KmMatrix<T>& _val, KmMatrixDim _dim);
};

}      // namespace KMenas
}      // namespace H204GPU

#endif  // M_ARITH_HPP_
