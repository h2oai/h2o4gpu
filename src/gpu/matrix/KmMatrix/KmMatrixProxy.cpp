/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include "KmMatrix.hpp"

namespace h2o4gpu {
namespace Matrix {

template <typename T>
KmMatrixProxy<T>::KmMatrixProxy(KmMatrix<T>& _other,
                                size_t _start, size_t _end, size_t _stride,
                                kParam<T>& _param)
    : orgi_ (_other), param_(_param), start_(_start), end_(_end),
      stride_(_stride) {
  assert(size() > 0);
}

template <typename T>
bool KmMatrixProxy<T>::on_device() const {
    return orgi_.on_device();
}

template <typename T>
size_t KmMatrixProxy<T>::start() const {
  return start_;
}

template <typename T>
size_t KmMatrixProxy<T>::end() const {
  return end_;
}

template <typename T>
size_t KmMatrixProxy<T>::stride() const {
  return stride_;
}

template <typename T>
size_t KmMatrixProxy<T>::size() const {
    return (end_ - start_) / stride_;
}

template <typename T>
void KmMatrixProxy<T>::operator=(KmMatrix<T> &_other) {
  // FIXME
  assert(false);
}

#define INSTANTIATE(T)                                                  \
  template KmMatrixProxy<T>::KmMatrixProxy(KmMatrix<T>& _other,         \
                                           size_t _start, size_t _end,  \
                                           size_t _stride,              \
                                           kParam<T>& _param);          \
  template bool KmMatrixProxy<T>::on_device() const;                    \
  template size_t KmMatrixProxy<T>::start() const;                      \
  template size_t KmMatrixProxy<T>::end() const;                        \
  template size_t KmMatrixProxy<T>::stride() const;                     \
  template size_t KmMatrixProxy<T>::size() const;                       \
  template void KmMatrixProxy<T>::operator=(KmMatrix<T> &_other);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)

}  // namespace Matrix
}  // namespace h2o4gpu
