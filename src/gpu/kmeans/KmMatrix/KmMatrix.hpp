/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef KM_MATRIX_HPP_
#define KM_MATRIX_HPP_

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <string>
#include <memory>
#include <iomanip>

// FIXME
#define USE_CUDA 1

#if defined (USE_CUDA)
#include "KmMatrixCuda.cuh"
#endif

namespace H2O4GPU {
namespace KMeans {

template <typename T>
class KmMatrixProxy;

template <typename T>
class KmMatrix;

// Kernel parameter
template <typename T>
struct kParam {
  size_t rows;
  size_t cols;
  T *ptr;

  kParam(size_t _rows, size_t _cols, T *_ptr)
      : rows (_rows), cols(_cols), ptr (_ptr) {}
  kParam(const kParam<T>& _other) {
    rows = _other.rows;
    cols = _other.cols;
    ptr = _other.ptr;
  }
  kParam operator=(const kParam& _other) {
    rows = _other.rows;
    cols = _other.cols;
    ptr = _other.ptr;
  }
};

template <typename T>
class KmMatrixImpl {
 private:
  KmMatrix<T> * matrix_;
 public:
  KmMatrixImpl(KmMatrix<T> *_matrix);
  virtual ~KmMatrixImpl () {}

  virtual T* host_ptr() {}
  virtual T* dev_ptr() {}
  virtual bool on_device() const {}
};

template <typename T>
class KmMatrix {
 private:

  enum Backend {
    CUDADense = 0,
    CUDASparse = 1,
    CPUDense = 2,
    CPUSparse = 3
  };

  std::shared_ptr<KmMatrixImpl<T>> impls[4];
  kParam<T> param_;

  bool use_cuda;

  void init_impls();

  std::string name_;

 public:
  explicit KmMatrix();
  KmMatrix(size_t _rows, size_t _cols);
  KmMatrix(thrust::host_vector<T> _other, size_t _rows, size_t _cols);
  KmMatrix(const KmMatrixProxy<T>& _other);

  KmMatrix(const KmMatrix<T>& _other);
  KmMatrix(KmMatrix<T>&& _other);

  void operator=(const KmMatrix<T>& _other);
  void operator=(KmMatrix<T>&& _other);

  virtual ~KmMatrix();

  size_t size () const;
  size_t rows () const;
  size_t cols () const;

  T* host_ptr();
  T* dev_ptr();

  kParam<T> k_param () const;

  std::string name() const { return name_; }
  void set_name (std::string _name) {name_ = _name;}

  KmMatrixProxy<T> row(size_t idx, bool dev_mem=true);
  KmMatrixProxy<T> col(size_t idx);
};

template <typename T>
std::ostream& operator<<(std::ostream& os, KmMatrix<T>& m) {
  std::cout << "matrix: " << m.name() << std::endl;
  T * ptr = m.host_ptr();
  kParam<T> param = m.k_param();
  for (size_t i = 0; i < param.rows; ++i) {
    for (size_t j = 0; j < param.cols; ++j) {
      std::cout << "(" << i << ", "<< j << ", " << i*param.cols + j << ")" << std::setw(6) << ptr[i*param.cols + j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << "---" << std::endl;
}

template <typename T>
class KmMatrixProxy {
 private:
  thrust::device_ptr<T> ptr_;
  size_t start_;
  size_t end_;
  size_t stride_;

  bool on_device_;

  kParam<T> param_;

 public:
  size_t start() const {
    return start_;
  }
  size_t end() const {
    return end_;
  }
  size_t stride() const {
    return stride_;
  }
  size_t size() const {
    return (end_ - start_) / stride_;
  }
  bool on_device() const {
    return on_device_;
  }
  thrust::device_ptr<T> data() const {
    return ptr_ + start_;
  }
  kParam<T> param() const {
    return param_;
  }

  KmMatrixProxy(thrust::device_ptr<T> _ptr,
                size_t _start, size_t _end, size_t _stride,
                kParam<T> _param, bool _on_device)
      : ptr_(_ptr), start_(_start), end_(_end), stride_(_stride),
        param_(_param), on_device_(_on_device) {}

  void operator=(KmMatrix<T>& _other) {
    assert(size() == _other.size);

    assert (_other.size() == size());
    // FIXME
    assert(stride_ == 1);

    if (on_device_) {
      auto _other_dev_ptr = thrust::device_ptr<T>(_other.dev_ptr());
      
      thrust::copy(_other_dev_ptr, _other_dev_ptr + size(), ptr_);
    } else {
      thrust::copy(_other.host_ptr(), _other.host_ptr() + size(),
                   ptr_ + start_);
    }
  }
};

}  // namespace KMeans
}  // namespace H2O4GPU

#endif
