#include <algorithm>
#include <limits>

#include "cgls.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector_cgls.h"
#include "projector_helper.h"
#include "util.h"

namespace pogs {

namespace {

int kMaxIter = 100;
bool kCglsQuiet = true;

// CGLS Gemv struct for matrix multiplication.
template <typename T, typename M>
struct Gemv : cgls::Gemv<T> {
  const M& A;
  Gemv(const M& A) : A(A) { }
  int operator()(char op, const T alpha, const T *x, const T beta, T *y)
      const {
    return A.Mul(op, alpha, x, beta, y);
  }
};

}  // namespace

template <typename T, typename M>
ProjectorCgls<T, M>::ProjectorCgls(int ignored, const M& A)
    : _A(A) { }

template <typename T, typename M>
ProjectorCgls<T, M>::~ProjectorCgls() { }

template <typename T, typename M>
int ProjectorCgls<T, M>::Init() {
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  ASSERT(_A.IsInit());

  return 0;
}

template <typename T, typename M>
int ProjectorCgls<T, M>::Project(const T *x0, const T *y0, T s, T *x, T *y,
                                 T tol) {
  DEBUG_EXPECT(this->_done_init);
  DEBUG_EXPECT(s >= static_cast<T>(0.));
  if (!this->_done_init || s < static_cast<T>(0.))
    return 1;

  // Set initial x and y.
  memset(x, 0, _A.Cols() * sizeof(T));
  memcpy(y, y0, _A.Rows() * sizeof(T));

  // y := y0 - Ax0;
  _A.Mul('n', static_cast<T>(-1.), x0, static_cast<T>(1.), y);

  // Minimize ||Ax - b||_2^2 + s||x||_2^2
  cgls::Solve(Gemv<T, M>(_A), static_cast<cgls::INT>(_A.Rows()),
      static_cast<cgls::INT>(_A.Cols()), y, x, s, tol, kMaxIter, kCglsQuiet);
 
  // x := x + x0
  gsl::vector<T> x_vec = gsl::vector_view_array(x, _A.Cols());
  const gsl::vector<T> x0_vec = gsl::vector_view_array(x0, _A.Cols());
  gsl::blas_axpy(static_cast<T>(1.), &x0_vec, &x_vec);

  // y := Ax
  _A.Mul('n', static_cast<T>(1.), x, static_cast<T>(0.), y);

#ifdef DEBUG
  // Verify that projection was successful.
  CheckProjection(&_A, x0, y0, x, y, s, static_cast<T>(1e1 * 1E-3));
#endif

  return 0;
}

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class ProjectorCgls<double, MatrixDense<double> >;
template class ProjectorCgls<double, MatrixSparse<double> >;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class ProjectorCgls<float, MatrixDense<float> >;
template class ProjectorCgls<float, MatrixSparse<float> >;
#endif

}  // namespace pogs

