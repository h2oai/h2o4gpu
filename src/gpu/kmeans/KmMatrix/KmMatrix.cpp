/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include "KmMatrix.hpp"
#if defined (USE_CUDA)
#include "KmMatrixCuda.cuh"
#endif

namespace H2O4GPU {
namespace KMeans {

// ==============================
// KmMatrixImpl implementation
// ==============================

template <typename T>
KmMatrixImpl<T>::KmMatrixImpl(KmMatrix<T> *_matrix)
    : matrix_(_matrix){}


// ==============================
// KmMatrix implementation
// ==============================

template <typename T>
KmMatrix<T>::KmMatrix() :
    param_ (0, 0, nullptr) {
  init_impls();
#if defined (USE_CUDA)
  use_cuda = true;
  impls[0].reset(new CudaKmMatrixImpl<T>(this));
#elif
  use_cuda = false;
  impls[0] = nullptr;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(size_t _rows, size_t _cols) :
    param_ (_rows, _cols, nullptr) {
  init_impls();
#if defined (USE_CUDA)
  use_cuda = true;
  impls[0].reset(new CudaKmMatrixImpl<T>(this));
#elif
  use_cuda = false;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(thrust::host_vector<T> _other,
                      size_t _rows, size_t _cols) :
    param_ (_rows, _cols, nullptr) {
  init_impls();
#if defined (USE_CUDA)
  use_cuda = true;
  impls[0].reset(new CudaKmMatrixImpl<T>(_other, this));
#elif
  use_cuda = false;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(const KmMatrix<T>& _other) :
    param_(_other.param_) {
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = _other.impls[i];
  }
  use_cuda = _other.use_cuda;
  name_ = _other.name_ + "(copied)";
}

template <typename T>
KmMatrix<T>::KmMatrix(KmMatrix<T>&& _other) :
    param_(_other.param_){
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = std::move(_other.impls[i]);
  }
  use_cuda = _other.use_cuda;
  name_ = std::move(_other.name_);
}

template <typename T>
void KmMatrix<T>::operator=(const KmMatrix<T>& _other) {
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = _other.impls[i];
  }
  param_ = _other.param_;
  use_cuda = _other.use_cuda;
  name_ = _other.name_ + "(copied)";
}

template <typename T>
void KmMatrix<T>::operator=(KmMatrix<T>&& _other) {
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = std::move(_other.impls[i]);
  }
  param_ = _other.param_;
  use_cuda = _other.use_cuda;
  name_ = std::move(_other.name_);
}

template <typename T>
KmMatrix<T>::KmMatrix(const KmMatrixProxy<T>& _other) :
    param_ (_other.param()){
  init_impls();
#if defined (USE_CUDA)
  use_cuda = true;
  impls[0].reset(new CudaKmMatrixImpl<T>(_other, this));
#elif
  use_cuda = false;
#endif
}

template <typename T>
void KmMatrix<T>::init_impls() {
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = nullptr;
  }
}

template <typename T>
KmMatrix<T>::~KmMatrix() {
  // std::cout << "name: " << name_ << std::endl;
  // for (size_t i = 0; i < 4; ++i) {
  //   if (impls[i] != nullptr)
  //     delete impls[i];
  // }
}


template <typename T>
size_t KmMatrix<T>::size() const {
  return param_.rows * param_.cols;
}

template <typename T>
size_t KmMatrix<T>::rows() const {
  return param_.rows;
}

template <typename T>
size_t KmMatrix<T>::cols() const {
  return param_.cols;
}

template <typename T>
kParam<T> KmMatrix<T>::k_param () const {
  return param_;
}

template <typename T>
T* KmMatrix<T>::host_ptr() {
  if (use_cuda) {
    return impls[0]->host_ptr();
  } else {
    // FIXME
    return nullptr;
  }
}

template <typename T>
T* KmMatrix<T>::dev_ptr() {
  if (use_cuda) {
    return impls[CUDADense]->dev_ptr();
  } else {
    return nullptr;
  }
}

template <typename T>
KmMatrixProxy<T> KmMatrix<T>::row(size_t idx, bool dev_mem) {
  size_t start = param_.cols * idx;
  size_t stride = 1;
  size_t end = param_.cols * (idx + 1);

  if (impls[0] != nullptr) {
    if (dev_mem) {
      std::cout << "row dev_mem" << std::endl;
      return KmMatrixProxy<T>(thrust::device_ptr<T>(impls[0]->dev_ptr()),
                              start, end, stride, param_, true);
    } else {
      std::cout << "row host_mem" << std::endl;
      return KmMatrixProxy<T>(thrust::device_ptr<T>(impls[0]->host_ptr()),
                              start, end, stride, param_, false);
    }
  }
  std::cerr << "no cuda" << std::endl;
  // FIXME
  assert(false);
  // return KmMatrixProxy<T>(thrust::device_ptr<T>(NULL), 0, 0, 0, param_, false);
}

template <typename T>
KmMatrixProxy<T> KmMatrix<T>::col(size_t idx) {
  // FIXME
  assert (false);
  return KmMatrixProxy<T>(nullptr, 0, 0, 0, param_, false);
}

#define INSTANTIATE(T)                                                  \
  template KmMatrixImpl<T>::KmMatrixImpl(KmMatrix<T> *_matrix);         \
  template KmMatrix<T>::KmMatrix();                                     \
  template KmMatrix<T>::KmMatrix(size_t _rows, size_t _cols);           \
  template KmMatrix<T>::KmMatrix(thrust::host_vector<T> _other,         \
                                 size_t _rows, size_t _cols);           \
  template KmMatrix<T>::KmMatrix(const KmMatrix<T>& _other);            \
  template KmMatrix<T>::KmMatrix(KmMatrix<T>&& _other);                 \
  template void KmMatrix<T>::operator=(const KmMatrix<T>& _other);      \
  template void KmMatrix<T>::operator=(KmMatrix<T>&& _other);           \
  template KmMatrix<T>::KmMatrix(const KmMatrixProxy<T>& _other);       \
  template KmMatrix<T>::~KmMatrix();                                    \
  template size_t KmMatrix<T>::size() const;                            \
  template size_t KmMatrix<T>::rows() const;                            \
  template size_t KmMatrix<T>::cols() const;                            \
  template kParam<T> KmMatrix<T>::k_param () const;                     \
  template T * KmMatrix<T>::host_ptr();                                 \
  template T * KmMatrix<T>::dev_ptr();                                  \
  template KmMatrixProxy<T> KmMatrix<T>::row(size_t idx, bool dev_mem=true);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

#undef INSTANTIATE
}
}  // H2O4GPU
