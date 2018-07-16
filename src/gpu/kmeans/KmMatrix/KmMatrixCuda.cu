/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <stdexcept>
#include <iostream>

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
  std::cout << "Copy host vector: " << _h_vector.size() << std::endl;
  for (size_t i = 0; i < _h_vector.size(); ++i) {
    std::cout << _h_vector[i] << ' ';
  }
  std::cout << std::endl;
}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(const KmMatrixProxy<T>& _other,
                                      KmMatrix<T>* _par)
    : KmMatrixImpl<T>(_par) {
  thrust::device_ptr<T> ptr = _other.data();
  if (_other.on_device()) {
    std::cout << "proxy to dev" << std::endl;
    if (_other.stride() == 1) {
      _d_vector.resize(_other.size());
      thrust::copy(ptr, ptr + _other.size(), _d_vector.begin());
      std::cout << "copied" << std::endl;
      for (size_t i = 0; i < _d_vector.size(); ++i) {
        std::cout << _d_vector[i] << ' ';
      }
      std::cout << std::endl;
    } else {
      // FIXME
      assert(false);
    }
    _on_device = true;
  } else {
    if (_other.stride() == 1) {
      _h_vector.resize(_other.size());
      std::cout << "_other.size(): " << _other.size() << std::endl;
      for (size_t i = 0; i < _other.size(); ++i) {
        std::cout << ptr[i] << ' ';
      }
      std::cout << std::endl;
      thrust::copy(ptr, ptr + _other.size(), _h_vector.begin());

    } else {
      // FIXME
      assert(false);
    }
    _on_device = false;
  }
}

template <typename T>
CudaKmMatrixImpl<T>::~CudaKmMatrixImpl() {}

template <typename T>
T* CudaKmMatrixImpl<T>::host_ptr() {
  device_to_host();
  std::cout << "host ptr: " << _h_vector.size() << std::endl;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      std::cout << std::setw(6) << _h_vector[i*4+j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  return _h_vector.data();
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
  _h_vector.resize(_d_vector.size());
  thrust::copy(_h_vector.begin(), _h_vector.end(), _d_vector.begin());
  _on_device = true;
}

template <typename T>
void CudaKmMatrixImpl<T>::device_to_host() {
  if (!_on_device)
    return;
  std::cout << "bring back to host" << std::endl;
  std::cout << "_d_.size()" << _d_vector.size() << std::endl;
  _h_vector.resize(_d_vector.size());
  thrust::copy(_d_vector.begin(), _d_vector.end(), _h_vector.begin());
  _on_device = false;
}

template <typename T>
bool CudaKmMatrixImpl<T>::on_device() const {
  return _on_device;
}

#define INSTANTIATE(T)                                                  \
  template bool CudaKmMatrixImpl<T>::on_device() const;                 \
  template void CudaKmMatrixImpl<T>::device_to_host();                  \
  template void CudaKmMatrixImpl<T>::host_to_device();                  \
  template T* CudaKmMatrixImpl<T>::dev_ptr();                           \
  template T* CudaKmMatrixImpl<T>::host_ptr();                          \
  template CudaKmMatrixImpl<T>::~CudaKmMatrixImpl();                    \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(                       \
      const KmMatrixProxy<T>& _other, KmMatrix<T>* _par);               \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(                       \
      const thrust::host_vector<T>& _h_vec, KmMatrix<T>* _par);         \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par);   \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

#undef INSTANTIATE
}  // namespace H204GPU
}  // namespace Array
