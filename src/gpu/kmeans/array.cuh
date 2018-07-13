/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef CUDA_ARRAY_H_
#define CUDA_ARRAY_H_

#include <thrust/device_vector.h>
#include <cublas_v2.h>

namespace H2O4GPU {
namespace Array {

constexpr float esp = 1e-16f;

struct Dims {
  size_t dims[4];
  Dims() {
    for (size_t i = 0; i < 4; ++i) {
      dims[i] = 0;
    }
  }
  Dims(size_t _dims[4]) {
    for (size_t i = 0; i < 4; ++i) {
      dims[i] = _dims[i];
    }
  }
  Dims (size_t d0, size_t d1, size_t d2, size_t d3) {
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
  }
  size_t operator[](size_t _idx) const {
    return dims[_idx];
  }
  void operator=(const Dims& _other) {
    for (size_t i = 0; i < 4; ++i) {
      dims[i] = _other.dims[i];
    }
  }
};

template <typename T>
class CUDAArray {
 private:
  thrust::host_vector<T> _h_vector;
  thrust::device_vector<T> _d_vector;

  Dims _dims;

  bool _is_synced;

  size_t _stride;

  size_t _n_gpu;

  cublasHandle_t blas_handle;

 public:
  CUDAArray();
  CUDAArray(size_t _size);
  CUDAArray(Dims _dims);
  CUDAArray(const thrust::device_vector<T>& _d_vec, const Dims _dims);

  virtual ~CUDAArray();

  void operator=(const CUDAArray<T>& _other);

  void print() const;

  CUDAArray<T> index(size_t dim0);

  thrust::device_vector<T>& device_vector();

  size_t stride();

  size_t size () const;

  size_t n_gpu() const;

  T* get();

  Dims dims () const;
};


template <typename T>
CUDAArray<T> div(CUDAArray<T> _lhs, T _rhs) {
  if (_rhs < esp) {
    throw std::runtime_error("Value under flow");
  }
  // cublasScal(blas_handle, lhs);
}

template <typename T>
CUDAArray<T> min_element(CUDAArray<T>& _value) {
  T result = thrust::min_element
             (_value._d_vector.begin(), _value._d_vector.end());
  return result;
}

}  // namespace Array
}  // namespace H2O4GPU

#endif
