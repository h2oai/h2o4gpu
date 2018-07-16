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
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(this);
  impls[0].reset(ptr);
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
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(this);
  impls[0].reset(ptr);
#elif
  use_cuda = false;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(thrust::host_vector<T> _vec,
                      size_t _rows, size_t _cols) :
    param_ (_rows, _cols, nullptr) {
  init_impls();
#if defined (USE_CUDA)
  use_cuda = true;
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(_vec, this);
  impls[0].reset(ptr);
#elif
  use_cuda = false;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(const KmMatrixProxy<T>& _other) :
    param_ (_other.param_){
  init_impls();
#if defined (USE_CUDA)
  use_cuda = true;
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(
      _other.orgi_, _other.start(), _other.size(), _other.stride(), this);
  impls[0].reset(ptr);
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
  name_ = _other.name_ + "(copied [in move])";
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
  name_ = _other.name_ + "(copied [in move])";
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
bool KmMatrix<T>::on_device() const {
  if (use_cuda) {
    return impls[CUDADense]->on_device();
  } else {
    return false;
  }
}

template <typename T>
KmMatrixProxy<T> KmMatrix<T>::row(size_t idx, bool dev_mem) {
  size_t start = param_.cols * idx;
  size_t stride = 1;
  size_t end = param_.cols * (idx + 1);
  kParam<T> param(1, param_.cols, nullptr);
  return KmMatrixProxy<T>(*this, start, end, stride, param);
}

template <typename T>
KmMatrixProxy<T> KmMatrix<T>::col(size_t idx) {
  // FIXME
  assert (false);
  return KmMatrixProxy<T>(*this, 0, 0, 0);
}

template <typename T>
bool KmMatrix<T>::operator==(const KmMatrix<T> &_rhs) {
  if (_rhs.use_cuda && use_cuda) {
    std::shared_ptr<CudaKmMatrixImpl<T>> tmp =
        std::dynamic_pointer_cast<CudaKmMatrixImpl<T>>(impls[CUDADense]);
    bool res = std::dynamic_pointer_cast<CudaKmMatrixImpl<T>>(
        impls[CUDADense])->equal(tmp);
    // return std::dynamic_pointer_cast<CudaKmMatrixImpl<T>>(impls[CUDADense])->equal(
    //     _rhs.impls[CUDADense]);
    return res;
  } else {
    // FIXME
    assert(false);
    return false;
  }
}

// ==============================
// Helper functions
// ==============================
template <typename T>
std::ostream& operator<<(std::ostream& os, KmMatrix<T>& m) {
  std::cout << "matrix: " << m.name() << std::endl << "---" << std::endl;
  T * ptr = m.host_ptr();
  kParam<T> param = m.k_param();
  for (size_t i = 0; i < param.rows; ++i) {
    for (size_t j = 0; j < param.cols; ++j) {
      std::cout << std::setw(5) << ptr[i*param.cols + j] << ',';
    }
    std::cout << std::endl;
  }
  std::cout << "---" << std::endl;
  return os;
}

#define INSTANTIATE(T)                                                  \
  /* Standard con(de)structors*/                                        \
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
  /* Methods */                                                         \
  template size_t KmMatrix<T>::size() const;                            \
  template size_t KmMatrix<T>::rows() const;                            \
  template size_t KmMatrix<T>::cols() const;                            \
  template kParam<T> KmMatrix<T>::k_param () const;                     \
  template T * KmMatrix<T>::host_ptr();                                 \
  template T * KmMatrix<T>::dev_ptr();                                  \
  template bool KmMatrix<T>::on_device() const;                         \
  template KmMatrixProxy<T> KmMatrix<T>::row(size_t idx, bool dev_mem=true); \
  template bool KmMatrix<T>::operator==(const KmMatrix<T> &_rhs);       \
  /* Helper functions */                                                \
  template std::ostream& operator<<(std::ostream& os, KmMatrix<T>& m);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

#undef INSTANTIATE
}
}  // H2O4GPU
