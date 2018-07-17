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

#include "KmConfig.h"

#if USE_CUDA()
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

  M_HOSTDEV size_t size() const {
    return rows * cols;
  }
};

template <typename T>
class KmMatrixImpl {
 protected:
  KmMatrix<T> * matrix_;
 public:
  KmMatrixImpl(KmMatrix<T> *_matrix);
  virtual ~KmMatrixImpl () {}

  virtual T* host_ptr() {}
  virtual T* dev_ptr() {}
  virtual size_t size() const {}
  virtual bool on_device() const {}
};

template <typename T>
class KmMatrix {
 private:

  bool use_cuda;

  enum Backend {
    CUDADense = 0,
    CUDASparse = 1,
    CPUDense = 2,
    CPUSparse = 3
  };

  std::shared_ptr<KmMatrixImpl<T>> impls[4];
  kParam<T> param_;

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

  bool operator==(KmMatrix<T>& _rhs);

  virtual ~KmMatrix();

  size_t size () const;
  size_t rows () const;
  size_t cols () const;

  T* host_ptr();
  T* dev_ptr();

  bool on_device() const;

  kParam<T> k_param ();

  std::string name() const { return name_; }
  void set_name (std::string _name) {name_ = _name;}

  KmMatrixProxy<T> row(size_t idx, bool dev_mem=true);
  KmMatrixProxy<T> col(size_t idx);
};

template <typename T>
std::ostream& operator<<(std::ostream& os, KmMatrix<T>& m);

template <typename T>
class KmMatrixProxy {
 private:
  KmMatrix<T>& orgi_;

  kParam<T> param_;

  size_t start_;
  size_t end_;
  size_t stride_;

  size_t start() const;
  size_t end() const;
  size_t stride() const;

 public:

  size_t size() const;
  bool on_device() const;

  KmMatrixProxy(KmMatrix<T>& _other,
                size_t _start, size_t _end, size_t _stride, kParam<T>& _param);

  bool operator==(const KmMatrix<T>& _rhs);
  bool operator==(const KmMatrixProxy<T>& _rhs);

  void operator=(KmMatrix<T>& _other);
  friend KmMatrix<T>;
};

}  // namespace KMeans
}  // namespace H2O4GPU

#endif
