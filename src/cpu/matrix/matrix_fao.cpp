#include "gsl/gsl_vector.h"
#include "util.h"
#include "equil_helper.h"
#include "matrix/matrix.h"
#include "matrix/matrix_fao.h"

namespace pogs {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
const NormTypes kNormEquilibrate = kNorm2;
const NormTypes kNormNormalize   = kNormFro;

// template <typename T>
// struct CpuData {
//   const T *orig_data;
//   const POGS_INT *orig_ptr, *orig_ind;
//   CpuData(const T *data, const POGS_INT *ptr, const POGS_INT *ind)
//       : orig_data(data), orig_ptr(ptr), orig_ind(ind) { }
// };

CBLAS_TRANSPOSE_t OpToCblasOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CblasNoTrans : CblasTrans;
}

template <typename T>
T NormEst(NormTypes norm_type, const MatrixFAO<T>& A);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixDense Implementation /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
MatrixFAO<T>::MatrixFAO(T *dag_output, size_t m, T *dag_input, size_t n,
                        void (*Amul)(void *), void (*ATmul)(void *),
                        void *dag) :  Matrix<T>(m, n) {
  this->_dag_output = gsl::vector_view_array<T>(dag_output, m);
  this->_dag_input = gsl::vector_view_array<T>(dag_input, n);
  this->_m = m;
  this->_n = n;
  this->_Amul = Amul;
  this->_ATmul = ATmul;
  this->_dag = dag;
}

template <typename T>
MatrixFAO<T>::~MatrixFAO() {
  // TODO
}

template <typename T>
int MatrixFAO<T>::Init() {
  DEBUG_ASSERT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;
  return 0;
}

template <typename T>
int MatrixFAO<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  gsl::vector<T> x_vec, y_vec;
  if (trans == 'n' || trans == 'N') {
    // x_vec = gsl::vector_view_array<T>(x, this->_n);
    y_vec = gsl::vector_view_array<T>(y, this->_m);
    gsl::vector_memcpy<T>(&this->_dag_input, x);
    gsl::vector_scale<T>(&this->_dag_input, alpha);
    // Multiply by D.
    if (this->_done_equil) {
      gsl::vector_mul<T>(&this->_dag_input, &this->_d);
    }
    this->_Amul(this->_dag);
    // Multiply by E.
    if (this->_done_equil) {
      gsl::vector_mul<T>(&this->_dag_output, &this->_e);
    }
    gsl::vector_scale(&y_vec, beta);
    gsl::vector_add<T>(&y_vec, &this->_dag_output);
  } else {
    // x_vec = gsl::vector_view_array<T>(x, this->_m);
    y_vec = gsl::vector_view_array<T>(y, this->_n);
    gsl::vector_memcpy<T>(&this->_dag_output, x);
    gsl::vector_scale<T>(&this->_dag_output, alpha);
    // Multiply by E.
    if (this->_done_equil) {
      gsl::vector_mul<T>(&this->_dag_output, this->_e);
    }
    gsl::vector_memcpy<T>(&this->_dag_output, &x_vec);
    this->_ATmul(this->_dag);
    // Multiply by D.
    if (this->_done_equil) {
      gsl::vector_mul<T>(&this->_dag_input, &this->_d);
    }
    gsl::vector_scale<T>(&y_vec, beta);
    gsl::vector_add<T>(&y_vec, &this->_dag_input);
  }

  return 0;
}

template <typename T>
int MatrixFAO<T>::Equil(T *d, T *e,
                        const std::function<void(T*)> &constrain_d,
                        const std::function<void(T*)> &constrain_e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  // // Perform Sinkhorn-Knopp equilibration.
  // SinkhornKnopp(this, d, e, constrain_d, constrain_e);

  // Compute D := sqrt(D), E := sqrt(E), if 2-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    std::transform(d, d + this->_m, d, SqrtF<T>());
    std::transform(e, e + this->_n, e, SqrtF<T>());
  }

  // Scale A to have norm of 1 (in the kNormNormalize norm).
  T normA = NormEst(kNormNormalize, *this);

  // Scale d and e to account for normalization of A.
  gsl::vector<T> d_vec = gsl::vector_view_array<T>(d, this->_m);
  gsl::vector<T> e_vec = gsl::vector_view_array<T>(e, this->_n);
  gsl::vector_scale(&d_vec, 1 / normA);
  gsl::vector_scale(&e_vec, 1 / normA);

  // Save D and E.
  this->_done_equil = 1;
  this->_d = gsl::vector_view_array<T>(d, this->_m);
  this->_e = gsl::vector_view_array<T>(e, this->_m);

  DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA,
      gsl::blas_nrm2(&d_vec), gsl::blas_nrm2(&e_vec));

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
template <typename T>
T NormEst(NormTypes norm_type, const MatrixFAO<T>& A) {
  switch (norm_type) {
    case kNorm2: {
      return 1;
    }
    case kNormFro: {
      return 1;
    }
    case kNorm1:
      // 1-norm normalization doens't make make sense since it treats rows and
      // columns differently.
    default:
      ASSERT(false);
      return static_cast<T>(0.);
  }
}

}  // namespace

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class MatrixFAO<double>;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class MatrixFAO<float>;
#endif

}  // namespace pogs

