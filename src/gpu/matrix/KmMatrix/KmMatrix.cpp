/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include "KmMatrix.hpp"

#if USE_CUDA()
#include "KmMatrixCuda.cuh"
#endif

namespace h2o4gpu {
namespace Matrix {

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
#if USE_CUDA()
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(this);
  impls[(int)Backend::CUDADense].reset(ptr);
  backend_ = Backend::CUDADense;
#elif
  backend_ = Backend::CPUDense;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(size_t _rows, size_t _cols) :
    param_ (_rows, _cols, nullptr) {
  init_impls();
#if USE_CUDA()
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(_rows * _cols, this);
  impls[(int)Backend::CUDADense].reset(ptr);
  backend_ = Backend::CUDADense;
#elif
  backend_ = Backend::CPUDense;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(thrust::host_vector<T> _vec,
                      size_t _rows, size_t _cols) :
    param_ (_rows, _cols, nullptr) {
  init_impls();
  if (_vec.size() != _rows * _cols) {
    throw KmMatrixSizeError("Expecting hv::size() == rows * cols. " +
                            std::string("Got hv::size(): ") +
                            std::to_string(_vec.size()) +
                            ", rows: " + std::to_string(_rows) +
                            ", cols: " + std::to_string(_cols));
  }
#if USE_CUDA()
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(_vec, this);
  impls[(int)Backend::CUDADense].reset(ptr);
  backend_ = Backend::CUDADense;
#elif
  backend_ = Backend::CPUDense;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(const KmMatrixProxy<T>& _other) :
    param_ (_other.param_){
  init_impls();
  name_ = _other.orgi_.name_ + "(" + std::to_string(_other.start()) + "," +
          std::to_string(_other.end()) + ")";
#if USE_CUDA()
  KmMatrixImpl<T> * ptr = new CudaKmMatrixImpl<T>(
      _other.orgi_, _other.start(), _other.size(), _other.stride(), this);
  impls[(int)Backend::CUDADense].reset(ptr);
  backend_ = Backend::CUDADense;
#elif
  backend_ = Backend::CPUDense;
#endif
}

template <typename T>
KmMatrix<T>::KmMatrix(const KmMatrix<T>& _other) :
    param_(_other.param_) {
  copy_impls(_other.impls);
  backend_ = _other.backend_;
  name_ = _other.name_ + "(copied)";
}

template <typename T>
KmMatrix<T>::KmMatrix(KmMatrix<T>&& _other) :
    param_(_other.param_) {
  copy_impls(_other.impls);
  backend_ = _other.backend_;
  name_ = _other.name_ + "(copied [in move])";
}

template<typename T>
void KmMatrix<T>::copy_impls(const std::shared_ptr<KmMatrixImpl<T>>* _impls) {
  for (size_t i = 0; i < 4; ++i) {
    if (_impls[i].get() != nullptr) {
      impls[i] = _impls[i];
      impls[i]->set_interface(this);
    }
  }
}

template <typename T>
void KmMatrix<T>::operator=(const KmMatrix<T>& _other) {
  copy_impls(_other.impls);
  param_ = _other.param_;
  backend_ = _other.backend_;
  name_ = _other.name_ + "(copied)";
}

template <typename T>
void KmMatrix<T>::operator=(KmMatrix<T>&& _other) {
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = std::move(_other.impls[i]);
    if (impls[i] != nullptr)
      impls[i]->set_interface(this);
  }
  param_ = _other.param_;
  backend_ = _other.backend_;
  name_ = _other.name_ + "(copied [in move])";
}

template <typename T>
void KmMatrix<T>::init_impls() {
  for (size_t i = 0; i < 4; ++i) {
    impls[i] = nullptr;
  }
}

template <typename T>
KmMatrix<T>::~KmMatrix() {}


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
KmMatrix<T> KmMatrix<T>::reshape(size_t _rows, size_t _cols) const {
  if (_rows * _cols != param_.rows * param_.cols) {
    throw KmMatrixSizeError("Expecting rows * cols == " +
                            std::to_string(param_.rows * param_.cols) +
                            ", get " +
                            "rows: " + std::to_string(_rows) +
                            ", cols: " + std::to_string(_cols));
  }
  KmMatrix<T> res (*this);
  res.param_.rows = _rows;
  res.param_.cols = _cols;
  return res;
}

template <typename T>
kParam<T> KmMatrix<T>::k_param () {
  T * ptr = dev_ptr();
  kParam<T> param (param_);
  param.ptr = ptr;
  return param;
}

template <typename T>
T* KmMatrix<T>::host_ptr() {
  if (backend_ == Backend::CUDADense) {
    return impls[0]->host_ptr();
  } else {
    // FIXME
    return nullptr;
  }
}

template <typename T>
T* KmMatrix<T>::dev_ptr() {
  if (backend_ == Backend::CUDADense) {
    return impls[(int)Backend::CUDADense]->dev_ptr();
  } else {
    return nullptr;
  }
}

template <typename T>
bool KmMatrix<T>::on_device() const {
  if (backend_ == Backend::CUDADense) {
    return impls[(int)Backend::CUDADense]->on_device();
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
  h2o4gpu_error("Not implemented.");
  return KmMatrixProxy<T>(*this, 0, 0, 0);
}

template <typename T>
KmMatrix<T> KmMatrix<T>::rows(KmMatrix<T>& _index) {
  KmMatrix<T> res;
  if (backend_ == Backend::CUDADense &&
      _index.backend_ == Backend::CUDADense) {
    if (! _index.on_device()) {
      _index.dev_ptr();
    }
    res = impls[(int)Backend::CUDADense]->rows(_index);
  } else {
    h2o4gpu_error("Not implemented.");
  }
  return res;
}

template <typename T>
KmMatrix<T> KmMatrix<T>::cols(KmMatrix<T>& _index) {
  h2o4gpu_error("Not implemented.");
  KmMatrix<T> res;
  return res;
}

template <typename T>
bool KmMatrix<T>::operator==(KmMatrix<T>& _rhs) {
  if (_rhs.backend_ == Backend::CUDADense && backend_ == Backend::CUDADense) {
    bool res = impls[(int)Backend::CUDADense]->equal(_rhs);
    return res;
  } else {
    h2o4gpu_error("Not implemented.");
    return false;
  }
}

template <typename T>
KmMatrix<T> KmMatrix<T>::stack(KmMatrix<T> &_second,
                               KmMatrixDim _dim) {
  KmMatrix<T> res;

  if (_dim == KmMatrixDim::ROW) {
    if (cols() != _second.cols()) {
      h2o4gpu_error("Columns of first is not equal to second.");
    }

    if (backend_ == Backend::CUDADense) {
      res = impls[(int)Backend::CUDADense]->stack(_second, _dim);
    } else {
      h2o4gpu_error("Not implemented.");
    }

  } else {
    h2o4gpu_error("Not implemented.");
  }

  return res;
}


// ==============================
// Helper functions
// ==============================

template <typename T>
std::ostream& operator<<(std::ostream& os, KmMatrix<T>& m) {
  std::cout << "\nmatrix: " << m.name() << std::endl;
  std::cout << "shape: (" << m.rows() << ", " << m.cols() << ")\n" << "[";
  T * ptr = m.host_ptr();
  kParam<T> param = m.k_param();
  for (size_t i = 0; i < param.rows; ++i) {
    if (i == 0) std::cout << "[";
    else std::cout << " [";
    for (size_t j = 0; j < param.cols; ++j) {
      std::cout << std::setw(5) << ptr[i*param.cols + j] << ',';
    }
    std::cout << " ]";
    if (i != param.rows - 1) std::cout << "," << std::endl;
  }
  std::cout << "]\n" << std::endl;
  return os;
}

template <typename T>
KmMatrix<T> stack(KmMatrix<T>& _first, KmMatrix<T>& _second,
                  KmMatrixDim _dim) {
  return _first.stack(_second, _dim);
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
  template void KmMatrix<T>::copy_impls(                                \
      const std::shared_ptr<KmMatrixImpl<T>>* _impls);                  \
  template void KmMatrix<T>::operator=(const KmMatrix<T>& _other);      \
  template void KmMatrix<T>::operator=(KmMatrix<T>&& _other);           \
  template KmMatrix<T>::KmMatrix(const KmMatrixProxy<T>& _other);       \
  template KmMatrix<T>::~KmMatrix();                                    \
  /* Methods */                                                         \
  template size_t KmMatrix<T>::size() const;                            \
  template size_t KmMatrix<T>::rows() const;                            \
  template size_t KmMatrix<T>::cols() const;                            \
  template KmMatrix<T> KmMatrix<T>::reshape(                            \
      size_t _rows, size_t _cols) const;                                \
  template kParam<T> KmMatrix<T>::k_param ();                           \
  template T * KmMatrix<T>::host_ptr();                                 \
  template T * KmMatrix<T>::dev_ptr();                                  \
  template bool KmMatrix<T>::on_device() const;                         \
  template KmMatrixProxy<T> KmMatrix<T>::row(size_t idx, bool dev_mem=true); \
  template KmMatrix<T> KmMatrix<T>::rows(KmMatrix<T>& _index);          \
  template KmMatrix<T> KmMatrix<T>::cols(KmMatrix<T>& _index);          \
  template bool KmMatrix<T>::operator==(KmMatrix<T> &_rhs);             \
  template KmMatrix<T> KmMatrix<T>::stack(KmMatrix<T> &_second,         \
                                          KmMatrixDim _dim);            \
  /* Helper functions */                                                \
  template std::ostream& operator<<(std::ostream& os, KmMatrix<T>& m);  \
  template KmMatrix<T> stack(KmMatrix<T>& _first, KmMatrix<T>& _second, \
      KmMatrixDim _dim);


INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

#undef INSTANTIATE
}  // namespace Matrix
}  // namepsace h2o4gpu
