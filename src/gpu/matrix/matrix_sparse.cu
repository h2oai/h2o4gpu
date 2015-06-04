#include <cublas_v2.h>
#include <cusparse.h>

#include "cml/cml_spblas.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_vector.cuh"
#include "equil_helper.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_sparse.h"
#include "util.h"

namespace pogs {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
const NormTypes kNormEquilibrate = kNorm2; 
const NormTypes kNormNormalize   = kNormFro; 

template <typename T>
struct GpuData {
  const T *orig_data;
  const POGS_INT *orig_ptr, *orig_ind;
  cublasHandle_t d_hdl;
  cusparseHandle_t s_hdl;
  cusparseMatDescr_t descr;
  GpuData(const T *data, const POGS_INT *ptr, const POGS_INT *ind)
      : orig_data(data), orig_ptr(ptr), orig_ind(ind) {
    cublasCreate(&d_hdl);
    cusparseCreate(&s_hdl);
    cusparseCreateMatDescr(&descr);
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(d_hdl);
    cusparseDestroy(s_hdl);
    cusparseDestroyMatDescr(descr);
    DEBUG_CUDA_CHECK_ERR();
  }
};

cusparseOperation_t OpToCusparseOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return (trans == 'n' || trans == 'N')
      ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
}

template <typename T>
void MultDiag(const T *d, const T *e, POGS_INT m, POGS_INT n, POGS_INT nnz,
              typename MatrixSparse<T>::Ord ord, T *data, const POGS_INT *ind,
              const POGS_INT *ptr);

template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, const MatrixSparse<T>& A);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixDense Implementation /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
MatrixSparse<T>::MatrixSparse(char ord, POGS_INT m, POGS_INT n, POGS_INT nnz,
                              const T *data, const POGS_INT *ptr,
                              const POGS_INT *ind)
    : Matrix<T>(m, n), _data(0), _ptr(0), _ind(0), _nnz(nnz) {
  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  // It should work up to 2^31 == 2B, but let's be sure.
  DEBUG_EXPECT(nnz < static_cast<POGS_INT>(1 << 29));

  // Set GPU specific data.
  GpuData<T> *info = new GpuData<T>(data, ptr, ind);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixSparse<T>::MatrixSparse(const MatrixSparse<T>& A)
    : Matrix<T>(A._m, A._n), _data(0), _ptr(0), _ind(0), _nnz(A._nnz), 
      _ord(A._ord) {

  GpuData<T> *info_A = reinterpret_cast<GpuData<T>*>(A._info);
  GpuData<T> *info = new GpuData<T>(info_A->orig_data, info_A->orig_ptr,
      info_A->orig_ind);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixSparse<T>::~MatrixSparse() {
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;

  if (this->_done_init) {
    if (_data) {
      cudaFree(_data);
      _data = 0;
      DEBUG_CUDA_CHECK_ERR();
    }

    if (_ptr) {
      cudaFree(_ptr);
      _ptr = 0;
      DEBUG_CUDA_CHECK_ERR();
    }

    if (_ind) {
      cudaFree(_ind);
      _ind = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
  }
}

template <typename T>
int MatrixSparse<T>::Init() {
  DEBUG_ASSERT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  const T *orig_data = info->orig_data;
  const POGS_INT *orig_ptr = info->orig_ptr;
  const POGS_INT *orig_ind = info->orig_ind;

  // Allocate sparse matrix on gpu.
  cudaMalloc(&_data, static_cast<size_t>(2) * _nnz * sizeof(T));
  cudaMalloc(&_ind, static_cast<size_t>(2) * _nnz * sizeof(POGS_INT));
  cudaMalloc(&_ptr, (this->_m + this->_n + 2) * sizeof(POGS_INT));
  DEBUG_CUDA_CHECK_ERR();

  if (_ord == ROW) {
    cml::spmat<T, POGS_INT, CblasRowMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    cml::spmat_memcpy(info->s_hdl, &A, orig_data, orig_ind, orig_ptr);
  } else {
    cml::spmat<T, POGS_INT, CblasColMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    cml::spmat_memcpy(info->s_hdl, &A, orig_data, orig_ind, orig_ptr);
  }
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixSparse<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  cml::vector<T> x_vec, y_vec;
  if (trans == 'n' || trans == 'N') {
    x_vec = cml::vector_view_array<T>(x, this->_n);
    y_vec = cml::vector_view_array<T>(y, this->_m);
  } else {
    x_vec = cml::vector_view_array<T>(x, this->_m);
    y_vec = cml::vector_view_array<T>(y, this->_n);
  }

  if (_ord == ROW) {
    cml::spmat<T, POGS_INT, CblasRowMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    cml::spblas_gemv(info->s_hdl, OpToCusparseOp(trans), info->descr, alpha,
        &A, &x_vec, beta, &y_vec);
  } else {
    cml::spmat<T, POGS_INT, CblasColMajor> A(_data, _ind, _ptr, this->_m,
        this->_n, _nnz);
    cml::spblas_gemv(info->s_hdl, OpToCusparseOp(trans), info->descr, alpha,
        &A, &x_vec, beta, &y_vec);
  }
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixSparse<T>::Equil(T *d, T *e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  // Extract cublas handle from _info.
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->d_hdl;

  // Number of elements in matrix.
  size_t num_el = static_cast<size_t>(2) * _nnz;

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign;
  size_t num_sign_bytes = (num_el + 7) / 8;
  cudaMalloc(&sign, num_sign_bytes);
  CUDA_CHECK_ERR();

  // Fill sign bits, assigning each thread a multiple of 8 elements.
  size_t num_chars = num_el / 8;
  size_t grid_size = cml::calc_grid_dim(num_chars, cml::kBlockSize);
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    __SetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        SquareF<T>());
  } else {
    __SetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        AbsF<T>());
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // If numel(A) is not a multiple of 8, then we need to set the last couple
  // of sign bits too.
  if (num_el > num_chars * 8) {
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      __SetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, SquareF<T>());
    } else {
      __SetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, AbsF<T>());
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Perform Sinkhorn-Knopp equilibration.
  SinkhornKnopp(this, d, e);
  cudaDeviceSynchronize();

  // Transform A = sign(A) .* sqrt(A) if 2-norm equilibration was performed,
  // or A = sign(A) .* A if the 1-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    __UnSetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        SqrtF<T>());
  } else {
    __UnSetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        IdentityF<T>());
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // Deal with last few entries if num_el is not a multiple of 8.
  if (num_el > num_chars * 8) {
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      __UnSetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, SqrtF<T>());
    } else {
      __UnSetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
          num_el - num_chars * 8, IdentityF<T>());
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Compute D := sqrt(D), E := sqrt(E), if 2-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    thrust::transform(thrust::device_pointer_cast(d),
        thrust::device_pointer_cast(d + this->_m),
        thrust::device_pointer_cast(d), SqrtF<T>());
    thrust::transform(thrust::device_pointer_cast(e),
        thrust::device_pointer_cast(e + this->_n),
        thrust::device_pointer_cast(e), SqrtF<T>());
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Compute A := D * A * E.
  MultDiag(d, e, this->_m, this->_n, _nnz, _ord, _data, _ind, _ptr);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // Scale A to have norm of 1 (in the kNormNormalize norm).
  T normA = NormEst(hdl, kNormNormalize, *this);
  CUDA_CHECK_ERR();
  cudaDeviceSynchronize();
  cml::vector<T> a_vec = cml::vector_view_array(_data, num_el);
  cml::vector_scale(&a_vec, 1 / normA);
  cudaDeviceSynchronize();

  // Scale d and e to account for normalization of A.
  cml::vector<T> d_vec = cml::vector_view_array<T>(d, this->_m);
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, this->_n);
  T normd = cml::blas_nrm2(hdl, &d_vec);
  T norme = cml::blas_nrm2(hdl, &e_vec);
//  T scale = sqrt(normd * sqrt(this->_n) / (norme * sqrt(this->_m)));
  T scale = static_cast<T>(1.);
  cml::vector_scale(&d_vec, 1 / (scale * sqrt(normA)));
  cml::vector_scale(&e_vec, scale / sqrt(normA));
  cudaDeviceSynchronize();

  cudaFree(sign);
  CUDA_CHECK_ERR();

  DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA, normd, norme);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, const MatrixSparse<T>& A) {
  switch (norm_type) {
    case kNorm2: {
      return Norm2Est(hdl, &A);
    }
    case kNormFro: {
      const cml::vector<T> a = cml::vector_view_array(A.Data(), A.Nnz());
      return cml::blas_nrm2(hdl, &a) / std::sqrt(std::min(A.Rows(), A.Cols()));
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
void __global__ __MultRow(const T *d, const T *e, T *data,
                          const POGS_INT *row_ptr, const POGS_INT *col_ind,
                          POGS_INT size) {
  POGS_INT tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (POGS_INT t = tid; t < size; t += gridDim.x * blockDim.x)
    for (POGS_INT i = row_ptr[t]; i < row_ptr[t + 1]; ++i)
      data[i] *= d[t] * e[col_ind[i]];
}

// Performs D * A * E for A in col major
template <typename T>
void __global__ __MultCol(const T *d, const T *e, T *data,
                          const POGS_INT *col_ptr, const POGS_INT *row_ind,
                          POGS_INT size) {
  POGS_INT tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (POGS_INT t = tid; t < size; t += gridDim.x * blockDim.x)
    for (POGS_INT i = col_ptr[t]; i < col_ptr[t + 1]; ++i)
      data[i] *= d[row_ind[i]] * e[t];
}

template <typename T>
void MultDiag(const T *d, const T *e, POGS_INT m, POGS_INT n, POGS_INT nnz,
              typename MatrixSparse<T>::Ord ord, T *data, const POGS_INT *ind,
              const POGS_INT *ptr) {
  if (ord == MatrixSparse<T>::ROW) {
    size_t grid_dim_row = cml::calc_grid_dim(m, cml::kBlockSize);
    __MultRow<<<grid_dim_row, cml::kBlockSize>>>(d, e, data, ptr, ind, m);
    size_t grid_dim_col = cml::calc_grid_dim(n, cml::kBlockSize);
    __MultCol<<<grid_dim_col, cml::kBlockSize>>>(d, e, data + nnz, ptr + m + 1,
        ind + nnz, n);
  } else {
    size_t grid_dim_col = cml::calc_grid_dim(n, cml::kBlockSize);
    __MultCol<<<grid_dim_col, cml::kBlockSize>>>(d, e, data, ptr, ind, n);
    size_t grid_dim_row = cml::calc_grid_dim(m, cml::kBlockSize);
    __MultRow<<<grid_dim_row, cml::kBlockSize>>>(d, e, data + nnz, ptr + n + 1,
        ind + nnz, m);
  }
}

}  // namespace

#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class MatrixSparse<double>;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class MatrixSparse<float>;
#endif

}  // namespace pogs

