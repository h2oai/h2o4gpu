#include <algorithm>
#include <cstring>

#include "gsl/cblas.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "matrix/matrix_dense.h"
#include "projector/projector_direct.h"
#include "projector_helper.h"
#include "util.h"

namespace pogs {

namespace {

template<typename T>
struct CpuData {
  T *AA, *L, s;
  CpuData() : AA(0), L(0), s(static_cast<T>(-1.)) { }
};

}  // namespace

template <typename T, typename M>
ProjectorDirect<T, M>::ProjectorDirect(int ignored, const M& A)
    : _A(A) {
  // Set CPU specific this->_info.
  CpuData<T> *info = new CpuData<T>();
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T, typename M>
ProjectorDirect<T, M>::~ProjectorDirect() {
  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);

  if (info->AA) {
    delete [] info->AA;
    info->AA = 0;
  }

  if (info->L) {
    delete [] info->L;
    info->L = 0;
  }

  delete info;
  this->_info = 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Init() {
  if (this->_done_init)
    return 1;
  this->_done_init = true;
  ASSERT(_A.IsInit());

  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);

  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  info->AA = new T[min_dim * min_dim];
  ASSERT(info->AA != 0);
  info->L = new T[min_dim * min_dim];
  ASSERT(info->L != 0);
  memset(info->AA, 0, min_dim * min_dim * sizeof(T));
  memset(info->L, 0, min_dim * min_dim * sizeof(T));

  CBLAS_TRANSPOSE_t op_type = _A.Rows() > _A.Cols() ? CblasTrans : CblasNoTrans;

  // Compute AA
  if (_A.Order() == MatrixDense<T>::ROW) {
    const gsl::matrix<T, CblasRowMajor> A =
        gsl::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    gsl::matrix<T, CblasRowMajor> AA = gsl::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    //C := alpha*A*A' + beta*C
    gsl::blas_syrk(CblasLower, op_type,
        static_cast<T>(1.), &A, static_cast<T>(0.), &AA);
  } else {
    const gsl::matrix<T, CblasColMajor> A =
        gsl::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    gsl::matrix<T, CblasColMajor> AA = gsl::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    gsl::blas_syrk(CblasLower, op_type,
        static_cast<T>(1.), &A, static_cast<T>(0.), &AA);
  }

  return 0;
}

template <typename T, typename M>
int ProjectorDirect<T, M>::Project(const T *x0, const T *y0, T s, T *x, T *y,
                                   T tol) {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init || s < static_cast<T>(0.))
    return 1;

  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);

  size_t min_dim = std::min(_A.Rows(), _A.Cols());

  // Set up views for raw vectors.
  gsl::vector<T> y_vec = gsl::vector_view_array(y, _A.Rows());
  const gsl::vector<T> y0_vec = gsl::vector_view_array(y0, _A.Rows());
  gsl::vector<T> x_vec = gsl::vector_view_array(x, _A.Cols());
  const gsl::vector<T> x0_vec = gsl::vector_view_array(x0, _A.Cols());

  // Set (x, y) = (x0, y0).
  gsl::vector_memcpy(&x_vec, &x0_vec);
  gsl::vector_memcpy(&y_vec, &y0_vec);

  if (_A.Order() == MatrixDense<T>::ROW) {
    const gsl::matrix<T, CblasRowMajor> A =
        gsl::matrix_view_array<T, CblasRowMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    gsl::matrix<T, CblasRowMajor> AA = gsl::matrix_view_array<T, CblasRowMajor>
        (info->AA, min_dim, min_dim);
    gsl::matrix<T, CblasRowMajor> L = gsl::matrix_view_array<T, CblasRowMajor>
        (info->L, min_dim, min_dim);

    if (s != info->s) {
      gsl::matrix_memcpy(&L, &AA); // originally from AA := A*A'
      gsl::vector<T> diagL = gsl::matrix_diagonal(&L);
      gsl::vector_add_constant(&diagL, s);
      gsl::linalg_cholesky_decomp(&L);
    }
    if (_A.Rows() > _A.Cols()) {
      // 1*A*y + 1*x -> x
      gsl::blas_gemv(CblasTrans, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      // Solve L*x = 0 for x -> x
      gsl::linalg_cholesky_svx(&L, &x_vec);
      // 1*A*x+0*y -> y
      gsl::blas_gemv(CblasNoTrans, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
    } else {
      gsl::blas_gemv(CblasNoTrans, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      gsl::linalg_cholesky_svx(&L, &y_vec);
      gsl::blas_gemv(CblasTrans, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      gsl::blas_axpy(static_cast<T>(1.), &y0_vec, &y_vec);
    }
  } else {
    const gsl::matrix<T, CblasColMajor> A =
        gsl::matrix_view_array<T, CblasColMajor>
        (_A.Data(), _A.Rows(), _A.Cols());
    gsl::matrix<T, CblasColMajor> AA = gsl::matrix_view_array<T, CblasColMajor>
        (info->AA, min_dim, min_dim);
    gsl::matrix<T, CblasColMajor> L = gsl::matrix_view_array<T, CblasColMajor>
        (info->L, min_dim, min_dim);

    if (s != info->s) {
      gsl::matrix_memcpy(&L, &AA);
      gsl::vector<T> diagL = gsl::matrix_diagonal(&L);
      gsl::vector_add_constant(&diagL, s);
      gsl::linalg_cholesky_decomp(&L);
    }
    if (_A.Rows() > _A.Cols()) {
      gsl::blas_gemv(CblasTrans, static_cast<T>(1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      gsl::linalg_cholesky_svx(&L, &x_vec);
      gsl::blas_gemv(CblasNoTrans, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(0.), &y_vec);
    } else {
      gsl::blas_gemv(CblasNoTrans, static_cast<T>(1.), &A, &x_vec,
          static_cast<T>(-1.), &y_vec);
      gsl::linalg_cholesky_svx(&L, &y_vec);
      gsl::blas_gemv(CblasTrans, static_cast<T>(-1.), &A, &y_vec,
          static_cast<T>(1.), &x_vec);
      gsl::blas_axpy(static_cast<T>(1.), &y0_vec, &y_vec);
    }
  }

#ifdef DEBUG
  // Verify that projection was successful.
  CheckProjection(&_A, x0, y0, x, y, s,
      static_cast<T>(1e3) * std::numeric_limits<T>::epsilon());
#endif

  info->s = s;
  return 0;
}

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class ProjectorDirect<double, MatrixDense<double> >;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class ProjectorDirect<float, MatrixDense<float> >;
#endif

}  // namespace pogs

