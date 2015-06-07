#include <algorithm>
#include <cstring>
#include <functional>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "equil_helper.h"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "util.h"

namespace pogs {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
const NormTypes kNormEquilibrate = kNorm2; 
const NormTypes kNormNormalize   = kNormFro;

template<typename T>
struct CpuData {
  const T *orig_data;
  CpuData(const T *orig_data) : orig_data(orig_data) { }
};

CBLAS_TRANSPOSE_t OpToCblasOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CblasNoTrans : CblasTrans;
}

template <typename T>
T NormEst(NormTypes norm_type, const MatrixDense<T>& A);

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDense<T>::Ord ord, T *data);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixDense Implementation /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
MatrixDense<T>::MatrixDense(char ord, size_t m, size_t n, const T *data)
    : Matrix<T>(m, n), _data(0) {
  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  // Set GPU specific _info.
  CpuData<T> *info = new CpuData<T>(data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDense<T>::MatrixDense(const MatrixDense<T>& A)
    : Matrix<T>(A._m, A._n), _data(0), _ord(A._ord) {

  CpuData<T> *info_A = reinterpret_cast<CpuData<T>*>(A._info);
  CpuData<T> *info = new CpuData<T>(info_A->orig_data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDense<T>::~MatrixDense() {
  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;
}

template <typename T>
int MatrixDense<T>::Init() {
  DEBUG_EXPECT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);

  // Copy Matrix to GPU.
  _data = new T[this->_m * this->_n];
  ASSERT(_data != 0);
  memcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T));

  return 0;
}

template <typename T>
int MatrixDense<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init)
    return 1;

  const gsl::vector<T> x_vec = gsl::vector_view_array<T>(x, this->_n);
  gsl::vector<T> y_vec = gsl::vector_view_array<T>(y, this->_m);

  if (_ord == ROW) {
    gsl::matrix<T, CblasRowMajor> A =
        gsl::matrix_view_array<T, CblasRowMajor>(_data, this->_m, this->_n);
    gsl::blas_gemv(OpToCblasOp(trans), alpha, &A, &x_vec, beta,
        &y_vec);
  } else {
    gsl::matrix<T, CblasColMajor> A =
        gsl::matrix_view_array<T, CblasColMajor>(_data, this->_m, this->_n);
    gsl::blas_gemv(OpToCblasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }

  return 0;
}

template <typename T>
int MatrixDense<T>::Equil(T *d, T *e,
                          const std::function<void(T*)> &constrain_d,
                          const std::function<void(T*)> &constrain_e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  // Number of elements in matrix.
  size_t num_el = this->_m * this->_n;

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign = 0;
  size_t num_sign_bytes = (num_el + 7) / 8;
  sign = new unsigned char[num_sign_bytes];
  ASSERT(sign != 0);

  // Fill sign bits, assigning each thread a multiple of 8 elements.
  size_t num_chars = num_el / 8;
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    SetSign(_data, sign, num_chars, SquareF<T>());
  } else {
    SetSign(_data, sign, num_chars, AbsF<T>());
  }

  // If numel(A) is not a multiple of 8, then we need to set the last couple
  // of sign bits too.
  if (num_el > num_chars * 8) {
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      SetSignSingle(_data + num_chars * 8, sign + num_chars,
          num_el - num_chars * 8, SquareF<T>());
    } else {
      SetSignSingle(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, AbsF<T>());
    }
  }

  // Perform Sinkhorn-Knopp equilibration.
  SinkhornKnopp(this, d, e, constrain_d, constrain_e);

  // Transform A = sign(A) .* sqrt(A) if 2-norm equilibration was performed,
  // or A = sign(A) .* A if the 1-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    UnSetSign(_data, sign, num_chars, SqrtF<T>());
  } else {
    UnSetSign(_data, sign, num_chars, IdentityF<T>());
  }

  // Deal with last few entries if num_el is not a multiple of 8.
  if (num_el > num_chars * 8) {
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
     UnSetSignSingle(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, SqrtF<T>());
    } else {
      UnSetSignSingle(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, IdentityF<T>());
    }
  }

  // Compute D := sqrt(D), E := sqrt(E), if 2-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    std::transform(d, d + this->_m, d, SqrtF<T>());
    std::transform(e, e + this->_n, e, SqrtF<T>());
  }

  // Compute A := D * A * E.
  MultDiag(d, e, this->_m, this->_n, _ord, _data);

  // Scale A to have norm of 1 (in the kNormNormalize norm).
  T normA = NormEst(kNormNormalize, *this);
  gsl::vector<T> a_vec = gsl::vector_view_array(_data, num_el);
  gsl::vector_scale(&a_vec, 1 / normA);

  // Scale d and e to account for normalization of A.
  gsl::vector<T> d_vec = gsl::vector_view_array<T>(d, this->_m);
  gsl::vector<T> e_vec = gsl::vector_view_array<T>(e, this->_n);
  gsl::vector_scale(&d_vec, 1 / std::sqrt(normA));
  gsl::vector_scale(&e_vec, 1 / std::sqrt(normA));

  DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA,
      gsl::blas_nrm2(&d_vec), gsl::blas_nrm2(&e_vec));

  delete [] sign;

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
template <typename T>
T NormEst(NormTypes norm_type, const MatrixDense<T>& A) {
  switch (norm_type) {
    case kNorm2: {
      return Norm2Est(&A);
    }
    case kNormFro: {
      const gsl::vector<T> a = gsl::vector_view_array(A.Data(),
          A.Rows() * A.Cols());
      return gsl::blas_nrm2(&a) /
          std::sqrt(static_cast<T>(std::min(A.Rows(), A.Cols())));
    }
    case kNorm1:
      // 1-norm normalization doens't make make sense since it treats rows and
      // columns differently.
    default:
      ASSERT(false);
      return static_cast<T>(0.);
  }
}

// Performs A := D * A * E for A in row major
template <typename T>
void MultRow(size_t m, size_t n, const T *d, const T *e, T *data) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t t = 0; t < m * n; ++t)
    data[t] *= d[t / n] * e[t % n];
}

// Performs A := D * A * E for A in col major
template <typename T>
void MultCol(size_t m, size_t n, const T *d, const T *e, T *data) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t t = 0; t < m * n; ++t)
    data[t] *= d[t % m] * e[t / m];
}

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDense<T>::Ord ord, T *data) {
  if (ord == MatrixDense<T>::ROW) {
    MultRow(m, n, d, e, data);
  } else {
    MultCol(m, n, d, e, data);
  }
}

}  // namespace

// Explicit template instantiation.
#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class MatrixDense<double>;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class MatrixDense<float>;
#endif

}  // namespace pogs

