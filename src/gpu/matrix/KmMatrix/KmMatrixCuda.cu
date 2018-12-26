/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include <stdexcept>
#include <iostream>
#include <memory>

#include <thrust/device_vector.h>

#include "KmMatrixCuda.cuh"
#include "KmMatrix.hpp"

namespace h2o4gpu {
namespace Matrix {

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par) :
    KmMatrixImpl<T>(_par){
  assert(_par);
}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(const thrust::host_vector<T>& _h_vec,
                                      KmMatrix<T>* _par)
    : on_device_(false), KmMatrixImpl<T>(_par) {
  assert(_par);
  h_vector_.resize(_h_vec.size());
  thrust::copy(_h_vec.begin(), _h_vec.end(), h_vector_.begin());
}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(size_t _size, KmMatrix<T> * _par) :
    KmMatrixImpl<T>(_par) {
  assert(_par);
  if (_size == 0) return;

  d_vector_.resize(_size);
  on_device_ = true;
}

template <typename T>
CudaKmMatrixImpl<T>::CudaKmMatrixImpl(
    KmMatrix<T>& _other, size_t _start, size_t _size, size_t _stride,
    KmMatrix<T> * _par) :
    KmMatrixImpl<T>(_par) {
  assert(_par);
  assert (_size > 0);

  if (_size == 0)
    return;

  T* raw_ptr;

  if (_other.on_device()) {
    raw_ptr = _other.dev_ptr();
    assert (raw_ptr);
    thrust::device_ptr<T> ptr (raw_ptr);
    ptr += _start;
    d_vector_.resize(_size);
    on_device_ = true;
    thrust::copy(ptr, ptr + _size, d_vector_.begin());
  } else {
    raw_ptr = _other.host_ptr();
    assert (raw_ptr);
    raw_ptr += _start;
    h_vector_.resize(_size);
    on_device_ = false;
    thrust::copy(raw_ptr, raw_ptr + _size, h_vector_.begin());
  }
}

template <typename T>
CudaKmMatrixImpl<T>::~CudaKmMatrixImpl() {}

template <typename T>
void CudaKmMatrixImpl<T>::set_interface(KmMatrix<T>* _par) {
  assert(_par);
  KmMatrixImpl<T>::matrix_ = _par;
}

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
bool CudaKmMatrixImpl<T>::equal(KmMatrix<T>& _rhs) {
  T* rhs_raw_ptr = _rhs.dev_ptr();
  host_to_device();
  thrust::device_ptr<T> rhs_ptr (rhs_raw_ptr);
  // FIXME, Is it floating compatible?
  bool res = thrust::equal(d_vector_.begin(), d_vector_.end(),
                           rhs_ptr);
  return res;
}

template <typename T>
KmMatrix<T> CudaKmMatrixImpl<T>::rows(KmMatrix<T>& _index) {

  KmMatrix<T> out (_index.rows(), KmMatrixImpl<T>::matrix_->cols());

  T * index_ptr = _index.dev_ptr();
  T * in_ptr = KmMatrixImpl<T>::matrix_->dev_ptr();
  T * out_ptr = out.dev_ptr();

  auto iter = thrust::make_counting_iterator(0);

  size_t cols = KmMatrixImpl<T>::matrix_->cols();

  thrust::for_each(
      thrust::device,
      iter, iter + _index.rows(),
      [=] __device__ (int idx) {
        size_t index = index_ptr[idx];

        size_t in_begin = index * cols;
        size_t in_end = (index + 1) * cols;

        size_t out_begin = idx * cols;

        for (size_t i = in_begin, j = out_begin; i != in_end; ++i, ++j) {
          out_ptr[j] = in_ptr[i];
        }
      });

  return out;
}

template <typename T>
KmMatrix<T> CudaKmMatrixImpl<T>::stack(KmMatrix<T>& _second,
                                       KmMatrixDim _dim) {
  if (_dim == KmMatrixDim::ROW) {
    if (KmMatrixImpl<T>::matrix_->cols() != _second.cols()) {
      h2o4gpu_error("Columns of first is not equal to second.");
    }
    host_to_device();

    T * sec_raw_ptr = _second.dev_ptr();
    thrust::device_ptr<T> self_ptr = d_vector_.data();

    thrust::device_ptr<T> sec_ptr (sec_raw_ptr);

    KmMatrix<T> res (KmMatrixImpl<T>::matrix_->rows() + _second.rows(),
                     KmMatrixImpl<T>::matrix_->cols());

    T * res_raw_ptr = res.dev_ptr();
    thrust::device_ptr<T> res_ptr (res_raw_ptr);

    thrust::copy(self_ptr, self_ptr + size(), res_ptr);
    res_ptr = thrust::device_ptr<T>(res_raw_ptr) + size();
    thrust::copy(sec_ptr, sec_ptr + _second.size(), res_ptr);

    return res;
  } else {
    // FIXME
    h2o4gpu_error("Not implemented.");
    KmMatrix<T> res;
    return res;
  }
}


#define INSTANTIATE(T)                                                  \
  /* Standard con(de)structors*/                                        \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(                       \
      KmMatrix<T>& _other, size_t _start, size_t _size, size_t _stride, \
      KmMatrix<T> * _par);                                              \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(                       \
      const thrust::host_vector<T>& _h_vec, KmMatrix<T>* _par);         \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(KmMatrix<T> * _par);   \
  template CudaKmMatrixImpl<T>::CudaKmMatrixImpl(size_t _size,          \
                                                 KmMatrix<T> * _par);   \
  template CudaKmMatrixImpl<T>::~CudaKmMatrixImpl();                    \
  template void CudaKmMatrixImpl<T>::set_interface(KmMatrix<T>* _par);  \
  /* Member functions */                                                \
  template bool CudaKmMatrixImpl<T>::on_device() const;                 \
  template void CudaKmMatrixImpl<T>::device_to_host();                  \
  template void CudaKmMatrixImpl<T>::host_to_device();                  \
  template T* CudaKmMatrixImpl<T>::dev_ptr();                           \
  template T* CudaKmMatrixImpl<T>::host_ptr();                          \
  template size_t CudaKmMatrixImpl<T>::size() const;                    \
  template bool CudaKmMatrixImpl<T>::equal(KmMatrix<T>& _rhs);          \
  template KmMatrix<T> CudaKmMatrixImpl<T>::stack(KmMatrix<T>& _second, \
                                                  KmMatrixDim _dim);    \
  template KmMatrix<T> CudaKmMatrixImpl<T>::rows(KmMatrix<T>& _index);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

#undef INSTANTIATE

}  // namespace Matrix
}  // namespace h2o4gpu
