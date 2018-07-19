/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <stdexcept>
#include <iostream>
#include <memory>

#include <thrust/device_vector.h>

#include "KmMatrix.hpp"
#include "KmMatrixCuda.cuh"

namespace H2O4GPU {
namespace KMeans {

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par) :
    KmMatrixImpl<T>(_par){}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(const thrust::host_vector<T>& _h_vec,
                                      KmMatrix<T>* _par)
    : on_device_(false), KmMatrixImpl<T>(_par) {
  h_vector_.resize(_h_vec.size());
  thrust::copy(_h_vec.begin(), _h_vec.end(), h_vector_.begin());
}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par, size_t _size) :
    KmMatrixImpl<T>(_par) {
  if (_size == 0) return;

  d_vector_.resize(_size);
  on_device_ = true;
}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(
    KmMatrix<T>& _other, size_t _start, size_t _size, size_t _stride,
    KmMatrix<T> * _par) :
    KmMatrixImpl<T>(_par) {
  assert (_size > 0);

  if (_size == 0)
    return;

  T* raw_ptr;

  assert (raw_ptr != nullptr && raw_ptr != NULL);

  std::cerr << "Warning: Copying data from " << _other.name()
            << "." << std::endl;
  if (_other.on_device()) {
    raw_ptr = _other.dev_ptr();
    thrust::device_ptr<T> ptr (raw_ptr);
    ptr += _start;
    d_vector_.resize(_size);
    on_device_ = true;
    thrust::copy(ptr, ptr + _size, d_vector_.begin());
  } else {
    raw_ptr = _other.host_ptr();
    raw_ptr += _start;
    h_vector_.resize(_size);
    on_device_ = false;
    thrust::copy(raw_ptr, raw_ptr + _size, h_vector_.begin());
  }
}

template <typename T>
CudaKmMatrixImpl<T>::~CudaKmMatrixImpl() {}

template <typename T>
T* CudaKmMatrixImpl<T>::host_ptr() {
  device_to_host();
  return thrust::raw_pointer_cast(h_vector_.data());
}

template <typename T>
T* CudaKmMatrixImpl<T>::dev_ptr() {
  host_to_device();
  T* ptr = thrust::raw_pointer_cast(d_vector_.data());
  return ptr;
}

template <typename T>
void CudaKmMatrixImpl<T>::host_to_device() {
  if (on_device_)
    return;
  d_vector_.resize(h_vector_.size());
  thrust::copy(h_vector_.begin(), h_vector_.end(), d_vector_.begin());
  on_device_ = true;
}

template <typename T>
void CudaKmMatrixImpl<T>::device_to_host() {
  if (!on_device_)
    return;
  h_vector_.resize(d_vector_.size());
  thrust::copy(d_vector_.begin(), d_vector_.end(), h_vector_.begin());
  on_device_ = false;
}

template <typename T>
bool CudaKmMatrixImpl<T>::on_device() const {
  return on_device_;
}

template <typename T>
size_t CudaKmMatrixImpl<T>::size() const {
  if (on_device_) {
    return d_vector_.size();
  } else {
    return h_vector_.size();
  }
}

template <typename T>
bool CudaKmMatrixImpl<T>::equal(std::shared_ptr<CudaKmMatrixImpl<T>>& _rhs) {
  // FIXME, Is it floating compatible?
  _rhs->host_to_device();
  host_to_device();
  bool res = thrust::equal(d_vector_.begin(), d_vector_.end(),
                           _rhs->d_vector_.begin());
  return res;
}

#define INSTANTIATE(T)                                                  \
  /* Standard con(de)structors*/                                        \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(                       \
      KmMatrix<T>& _other, size_t _start, size_t _size, size_t _stride, \
      KmMatrix<T> * _par);                                              \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(                       \
      const thrust::host_vector<T>& _h_vec, KmMatrix<T>* _par);         \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par);   \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par,    \
                                                 size_t _size);         \
  template CudaKmMatrixImpl<T>::~CudaKmMatrixImpl();                    \
  /* Member functions */                                                \
  template bool CudaKmMatrixImpl<T>::on_device() const;                 \
  template void CudaKmMatrixImpl<T>::device_to_host();                  \
  template void CudaKmMatrixImpl<T>::host_to_device();                  \
  template T* CudaKmMatrixImpl<T>::dev_ptr();                           \
  template T* CudaKmMatrixImpl<T>::host_ptr();                          \
  template size_t CudaKmMatrixImpl<T>::size() const;                    \
  template bool CudaKmMatrixImpl<T>::equal(                             \
      std::shared_ptr<CudaKmMatrixImpl<T>>& _rhs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

#undef INSTANTIATE
}  // namespace H204GPU
}  // namespace Array
