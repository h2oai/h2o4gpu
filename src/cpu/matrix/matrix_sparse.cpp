/*!
 * Modifications copyright (C) 2017 H2O.ai
 */
#include "gsl/gsl_spblas.h"
#include "gsl/gsl_spmat.h"
#include "gsl/gsl_vector.h"
#include "equil_helper.h"
#include "matrix/matrix.h"
#include "matrix/matrix_sparse.h"
#include "util.h"

namespace h2o4gpu {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
const NormTypes kNormEquilibrate = kNorm2; 
const NormTypes kNormNormalize   = kNormFro; 

template <typename T>
struct CpuData {
  const T *orig_data;
  const H2O4GPU_INT *orig_ptr, *orig_ind;
  CpuData(const T *data, const H2O4GPU_INT *ptr, const H2O4GPU_INT *ind)
      : orig_data(data), orig_ptr(ptr), orig_ind(ind) { }
};

CBLAS_TRANSPOSE_t OpToCblasOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CblasNoTrans : CblasTrans;
}

template <typename T>
void MultDiag(const T *d, const T *e, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz,
              typename MatrixSparse<T>::Ord ord, T *data, const H2O4GPU_INT *ind,
              const H2O4GPU_INT *ptr);

template <typename T>
T NormEst(NormTypes norm_type, const MatrixSparse<T>& A);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixDense Implementation /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
MatrixSparse<T>::MatrixSparse(int sharedA, int me, int wDev, char ord, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz,
                              const T *data, const H2O4GPU_INT *ptr,
                              const H2O4GPU_INT *ind)
  : Matrix<T>(m, n), _sharedA(sharedA), _me(me),  _wDev(0), _data(0), _de(0), _ptr(0), _ind(0), _nnz(nnz) {
  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;
  _me=0;
  _sharedA=0;
  // Set CPU specific data.
  CpuData<T> *info = new CpuData<T>(data, ptr, ind);
  this->_info = reinterpret_cast<void*>(info);
}

  template <typename T>
  MatrixSparse<T>::MatrixSparse(int wDev, char ord, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz,
                              const T *data, const H2O4GPU_INT *ptr,
                              const H2O4GPU_INT *ind)
    : MatrixSparse<T>(0,0,wDev,ord,m,n,nnz,data,ptr,ind){}
  template <typename T>
  MatrixSparse<T>::MatrixSparse(char ord, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz,
                              const T *data, const H2O4GPU_INT *ptr,
                              const H2O4GPU_INT *ind)
    : MatrixSparse<T>(0,0,0,ord,m,n,nnz,data,ptr,ind){}

template <typename T>
MatrixSparse<T>::MatrixSparse(int sharedA, int me, int wDev, const MatrixSparse<T>& A)
  : Matrix<T>(A._m, A._n), _sharedA(sharedA), _me(me), _wDev(wDev), _data(0), _de(0), _ptr(0), _ind(0), _nnz(A._nnz), 
      _ord(A._ord) {
  _me=0;
  _sharedA=0;

  CpuData<T> *info_A = reinterpret_cast<CpuData<T>*>(A._info);
  CpuData<T> *info = new CpuData<T>(info_A->orig_data, info_A->orig_ptr,
      info_A->orig_ind);
  this->_info = reinterpret_cast<void*>(info);
}

  template <typename T>
MatrixSparse<T>::MatrixSparse(int wDev, const MatrixSparse<T>& A)
  : MatrixSparse<T>(A._sharedA,A._me,wDev,A){}

  template <typename T>
MatrixSparse<T>::MatrixSparse(const MatrixSparse<T>& A)
  : MatrixSparse<T>(A._sharedA,A._me,A._wDev,A){}

template <typename T>
MatrixSparse<T>::~MatrixSparse() {
  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;

  if (this->_done_init) {
    if (_data) {
      delete [] _data;
      _data = 0;
    }

    if (_ptr) {
      delete [] _ptr;
      _ptr = 0;
    }

    if (_ind) {
      delete [] _ind;
      _ind = 0;
    }
  }
  if(_de && !this->_sharedA){ // JONTODO: When sharedA=1, only free if on sourceme
    delete [] _de;
    this->_de=0;
  }
}

template <typename T>
int MatrixSparse<T>::Init() {
  DEBUG_ASSERT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  CpuData<T> *info = reinterpret_cast<CpuData<T>*>(this->_info);
  const T *orig_data = info->orig_data;
  const H2O4GPU_INT *orig_ptr = info->orig_ptr;
  const H2O4GPU_INT *orig_ind = info->orig_ind;

  // Allocate sparse matrix on gpu.
  _data = new T[static_cast<size_t>(2) * _nnz]; ASSERT(_data != 0);
  _de = new T[this->_m + this->_n]; ASSERT(_de != 0);memset(_de, 0, (this->_m + this->_n) * sizeof(T)); // not sparse
  //  Equil(1); // JONTODO: Hack -- for future if make like dense otherwise
  _ind = new H2O4GPU_INT[static_cast<size_t>(2) * _nnz]; ASSERT(_ind != 0);
  _ptr = new H2O4GPU_INT[this->_m + this->_n + 2]; ASSERT(_ptr != 0);

  if (_ord == ROW) {
    gsl::spmat<T, H2O4GPU_INT, CblasRowMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    gsl::spmat_memcpy(&A, orig_data, orig_ind, orig_ptr);
  } else {
    gsl::spmat<T, H2O4GPU_INT, CblasColMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    gsl::spmat_memcpy(&A, orig_data, orig_ind, orig_ptr);
  }

  return 0;
}

template <typename T>
int MatrixSparse<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  gsl::vector<T> x_vec, y_vec;
  if (trans == 'n' || trans == 'N') {
    x_vec = gsl::vector_view_array<T>(x, this->_n);
    y_vec = gsl::vector_view_array<T>(y, this->_m);
  } else {
    x_vec = gsl::vector_view_array<T>(x, this->_m);
    y_vec = gsl::vector_view_array<T>(y, this->_n);
  }

  if (_ord == ROW) {
    gsl::spmat<T, H2O4GPU_INT, CblasRowMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    gsl::spblas_gemv(OpToCblasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  } else {
    gsl::spmat<T, H2O4GPU_INT, CblasColMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    gsl::spblas_gemv(OpToCblasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }

  return 0;
}

template <typename T>
int MatrixSparse<T>::Mulvalid(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  gsl::vector<T> x_vec, y_vec;
  if (trans == 'n' || trans == 'N') {
    x_vec = gsl::vector_view_array<T>(x, this->_n);
    y_vec = gsl::vector_view_array<T>(y, this->_mvalid);
  } else {
    x_vec = gsl::vector_view_array<T>(x, this->_mvalid);
    y_vec = gsl::vector_view_array<T>(y, this->_n);
  }

  if (_ord == ROW) {
    gsl::spmat<T, H2O4GPU_INT, CblasRowMajor> A(_vdata, _ind, _ptr, this->_mvalid,
        this->_n, _nnz);
    gsl::spblas_gemv(OpToCblasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  } else {
    gsl::spmat<T, H2O4GPU_INT, CblasColMajor> A(_vdata, _ind, _ptr, this->_mvalid,
        this->_n, _nnz);
    gsl::spblas_gemv(OpToCblasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }

  return 0;
}

template <typename T>
int MatrixSparse<T>::Equil(bool equillocal) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  if (this->_done_equil) return 0;
  else this->_done_equil=1;

  int m=this->_m;
  int n=this->_n;

  T *d = _de;
  T *e = d+m;

  
  // Number of elements in matrix.
  size_t num_el = static_cast<size_t>(2) * _nnz;

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign;
  size_t num_sign_bytes = (num_el + 7) / 8;
  sign = new unsigned char[num_sign_bytes];

  size_t num_chars = num_el / 8;
  if(equillocal){
    // Fill sign bits, assigning each thread a multiple of 8 elements.
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
  }

  // Perform Sinkhorn-Knopp equilibration.
  SinkhornKnopp(this, d, e, equillocal);

  if(equillocal){
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
  }
  
  // Compute D := sqrt(D), E := sqrt(E), if 2-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    std::transform(d, d + this->_m, d, SqrtF<T>());
    std::transform(e, e + this->_n, e, SqrtF<T>());
  }

  // Compute A := D * A * E.
  MultDiag(d, e, this->_m, this->_n, _nnz, _ord, _data, _ind, _ptr);

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
T NormEst(NormTypes norm_type, const MatrixSparse<T>& A) {
  switch (norm_type) {
    case kNorm2: {
      return Norm2Est(&A);
    }
    case kNormFro: {
      const gsl::vector<T> a = gsl::vector_view_array(A.Data(), A.Nnz());
      return gsl::blas_nrm2(&a) / std::sqrt(std::min(A.Rows(), A.Cols()));
    }
    case kNorm1:
      // 1-norm normalization doens't make make sense since it treats rows and
      // columns differently.
    default:
      ASSERT(false);
      return static_cast<T>(0.);
  }
}

// Performs D * A * E for A in row major
template <typename T>
void MultRow(const T *d, const T *e, T *data, const H2O4GPU_INT *row_ptr,
             const H2O4GPU_INT *col_ind, H2O4GPU_INT size) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (H2O4GPU_INT t = 0; t < size; ++t)
    for (H2O4GPU_INT i = row_ptr[t]; i < row_ptr[t + 1]; ++i)
      data[i] *= d[t] * e[col_ind[i]];
}

// Performs D * A * E for A in col major
template <typename T>
void MultCol(const T *d, const T *e, T *data, const H2O4GPU_INT *col_ptr,
             const H2O4GPU_INT *row_ind, H2O4GPU_INT size) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (H2O4GPU_INT t = 0; t < size; ++t)
    for (H2O4GPU_INT i = col_ptr[t]; i < col_ptr[t + 1]; ++i)
      data[i] *= d[row_ind[i]] * e[t];
}

template <typename T>
void MultDiag(const T *d, const T *e, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz,
              typename MatrixSparse<T>::Ord ord, T *data, const H2O4GPU_INT *ind,
              const H2O4GPU_INT *ptr) {
  if (ord == MatrixSparse<T>::ROW) {
    MultRow(d, e, data, ptr, ind, m);
    MultCol(d, e, data + nnz, ptr + m + 1, ind + nnz, n);
  } else {
    MultCol(d, e, data, ptr, ind, n);
    MultRow(d, e, data + nnz, ptr + n + 1, ind + nnz, m);
  }
}

}  // namespace

#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE==1
template class MatrixSparse<double>;
#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE==1
template class MatrixSparse<float>;
#endif

}  // namespace h2o4gpu

