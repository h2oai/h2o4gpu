/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <stdexcept>
#include <iostream>

#include "array.cuh"
#include "kmeans_general.h"

namespace H2O4GPU {
namespace Array {

template <typename T>
CUDAArray<T>::CUDAArray() {
  CUBLAS_CHECK(cublasCreate(&blas_handle));
}

template <typename T>
CUDAArray<T>::CUDAArray(size_t _size) {
  this->_d_vector.resize(_size);
  CUBLAS_CHECK(cublasCreate(&blas_handle));
}

template <typename T>
CUDAArray<T>::CUDAArray(Dims _other) {
  _dims = _other;
  _d_vector.resize(_dims[0] * _dims[1]);
  CUBLAS_CHECK(cublasCreate(&blas_handle));
}

template <typename T>
CUDAArray<T>::CUDAArray(const thrust::device_vector<T>& _d_vec,
                        const Dims _dims) {
  this->_d_vector = _d_vec;
  this->_dims = _dims;
  CUBLAS_CHECK(cublasCreate(&blas_handle));
}

template <typename T>
CUDAArray<T>::~CUDAArray() {
  // if (blas_handle != NULL)
  //   CUBLAS_CHECK(cublasDestroy(blas_handle));
}

template <typename T>
void CUDAArray<T>::operator=(const CUDAArray<T>& _other) {
  _dims = _other._dims;
  _d_vector = _other._d_vector;

  for (size_t i = 0; i < _d_vector.size(); ++i) {
    std::cout << _d_vector[i] << ' ';
  }
  std::cout << std::endl;
}

template <typename T>
void CUDAArray<T>::print() const {
  std::cout << "Array: [";
  for (size_t i = 0; i < 4; ++i) {
    std::cout << _dims[i] << ", ";
  }
  std::cout << "\b\b]" << std::endl;
  for (size_t i = 0; i < _dims[0]; ++i) {
    for (size_t j = 0; j < _dims[1]; ++j) {
      std::cout << _d_vector[i*_dims[0]+j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// return 1 row
template <typename T>
CUDAArray<T> CUDAArray<T>::index(size_t _idx) {

  Dims new_dim (1, _dims[1], 0, 0);
  CUDAArray<T> result (new_dim);
  thrust::device_vector<T> _row (_dims[1]);

  thrust::copy(_d_vector.begin() + _idx * _dims[1],
               _d_vector.begin() + (_idx+1) * _dims[1],
               result._d_vector.begin());
  return result;
}

template <typename T>
T* CUDAArray<T>::get() {
  return _d_vector.data().get();
}

template <typename T>
thrust::device_vector<T>& CUDAArray<T>::device_vector() {
  return _d_vector;
}

template <typename T>
size_t CUDAArray<T>::stride() {
  return _stride;
}

template <typename T>
size_t CUDAArray<T>::size () const {
  return _h_vector.size();
}

template <typename T>
size_t CUDAArray<T>::n_gpu() const {
  return _n_gpu;
}

template <typename T>
Dims CUDAArray<T>::dims() const {
  return _dims;
}


#define INSTANTIATE(T)                                                  \
  template CUDAArray<T>::CUDAArray();                                   \
  template CUDAArray<T>::CUDAArray(size_t _size);                       \
  template CUDAArray<T>::CUDAArray(const thrust::device_vector<T>& _d_vec, \
                                   const Dims _dims);                   \
  template CUDAArray<T>::~CUDAArray();                                  \
  template void CUDAArray<T>::operator=(const CUDAArray<T>& _other);    \
  template void CUDAArray<T>::print() const;                            \
  template CUDAArray<T> CUDAArray<T>::index(size_t dim0);               \
  template T * CUDAArray<T>::get();                                     \
  template thrust::device_vector<T>& CUDAArray<T>::device_vector();     \
  template size_t CUDAArray<T>::stride();                               \
  template size_t CUDAArray<T>::size () const;                          \
  template size_t CUDAArray<T>::n_gpu() const;                          \
  template Dims CUDAArray<T>::dims() const;                             \

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace H204GPU
}  // namespace Array
