#include "matrix_dense.h"

namespace pogs {

template <typename T>
int MatrixDense<T>::Mul(Trans op, T alpha, T *x, T beta, T *y) {
  if (ord == ROW) {
    cml::matrix<T, CblasRowMajor> A = matrix_view_array(data, m, n);
    cml::blas_gemv(
  } else {

  }

}


}  // namespace pogs
