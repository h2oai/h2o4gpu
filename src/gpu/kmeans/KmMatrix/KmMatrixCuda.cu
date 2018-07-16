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
    : _on_device(false), KmMatrixImpl<T>(_par) {
  _h_vector.resize(_h_vec.size());
  thrust::copy(_h_vec.begin(), _h_vec.end(), _h_vector.begin());
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

  if (_other.on_device()) {
    raw_ptr = _other.dev_ptr();
    thrust::device_ptr<T> ptr (raw_ptr);
    ptr += _start;
    _d_vector.resize(_size);
    _on_device = true;
    thrust::copy(ptr, ptr + _size, _d_vector.begin());
  } else {
    raw_ptr = _other.host_ptr();
    raw_ptr += _start;
    _h_vector.resize(_size);
    _on_device = false;
    thrust::copy(raw_ptr, raw_ptr + _size, _h_vector.begin());
  }
}

template <typename T>
CudaKmMatrixImpl<T>::~CudaKmMatrixImpl() {}

template <typename T>
T* CudaKmMatrixImpl<T>::host_ptr() {
  device_to_host();
  return thrust::raw_pointer_cast(_h_vector.data());
}

template <typename T>
T* CudaKmMatrixImpl<T>::dev_ptr() {
  host_to_device();
  T* ptr = thrust::raw_pointer_cast(_d_vector.data());
  return ptr;
}

template <typename T>
void CudaKmMatrixImpl<T>::host_to_device() {
  if (_on_device)
    return;
  _d_vector.resize(_h_vector.size());
  thrust::copy(_h_vector.begin(), _h_vector.end(), _d_vector.begin());
  _on_device = true;
}

template <typename T>
void CudaKmMatrixImpl<T>::device_to_host() {
  if (!_on_device)
    return;
  _h_vector.resize(_d_vector.size());
  thrust::copy(_d_vector.begin(), _d_vector.end(), _h_vector.begin());
  _on_device = false;
}

template <typename T>
bool CudaKmMatrixImpl<T>::on_device() const {
  return _on_device;
}

template <typename T>
size_t CudaKmMatrixImpl<T>::size() const {
  if (_on_device) {
    return _d_vector.size();
  } else {
    return _h_vector.size();
  }
}

template <typename T>
bool CudaKmMatrixImpl<T>::equal(std::shared_ptr<CudaKmMatrixImpl<T>>& _rhs) {
  // FIXME, Is it floating compatible?
  _rhs->host_to_device();
  host_to_device();
  bool res = thrust::equal(_d_vector.begin(), _d_vector.end(),
                           _rhs->_d_vector.begin());
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
