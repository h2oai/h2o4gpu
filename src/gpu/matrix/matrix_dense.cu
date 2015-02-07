#include <assert.h>
#include <cublas_v2.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_rand.cuh"
#include "equil_helper.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "util.cuh"

// File scoped constants.
enum NormTypes { kNorm1, kNorm2, kNormFro };
const NormTypes kNormEquilibrate   = kNorm2; 
const NormTypes kNormNormalize     = kNorm2; 
const unsigned int kEquilIter      = 10u; 
const unsigned int kNormEstMaxIter = 50u;
const double kNormEstTol           = 1e-2;

namespace pogs {
////////////////////////////////////////////////////////////////////////////////
/////////////////////// Hidden Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

template<typename T>
struct GpuData {
  const T *orig_data;
  cublasHandle_t handle;
  GpuData(const T *orig_data) : orig_data(orig_data) {
    cublasCreate(&handle);
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(handle);
    DEBUG_CUDA_CHECK_ERR();
  }
};

cublasOperation_t OpToCublasOp(char trans) {
  assert(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
}

template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, size_t m, size_t n,
          typename MatrixDense<T>::Ord ord, const T *data);

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
  assert(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  // Set GPU specific _info.
  GpuData<T> *info = new GpuData<T>(data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDense<T>::MatrixDense(const MatrixDense<T>& A)
    : Matrix<T>(A._m, A._n), _data(0), _ord(A._ord) {

  GpuData<T> *info_A = reinterpret_cast<GpuData<T>*>(A._info);
  GpuData<T> *info = new GpuData<T>(info_A->orig_data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDense<T>::~MatrixDense() {
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;

  if (this->_done_init && _data) {
    cudaFree(_data);
    this->_data = 0;
    DEBUG_CUDA_CHECK_ERR();
  }
}
      
template <typename T>
int MatrixDense<T>::Init() {
  DEBUG_EXPECT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  // Copy Matrix to GPU.
  cudaMalloc(&_data, this->_m * this->_n * sizeof(T));
  cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),
      cudaMemcpyHostToDevice);
  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixDense<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init)
    return 1;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, this->_n);
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, this->_m);

  if (_ord == ROW) {
    cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>(_data, this->_m, this->_n);
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta,
        &y_vec);
  } else {
    cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>(_data, this->_m, this->_n);
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }
  CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixDense<T>::Equil(T *d, T *e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  // Wrap raw pointers in cml::vectors and initialize to 1.
  cml::vector<T> d_vec = cml::vector_view_array<T>(d, this->_m);
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, this->_n);
  cml::vector_set_all(&d_vec, static_cast<T>(1));
  cml::vector_set_all(&e_vec, static_cast<T>(1));

  size_t num_el = this->_m * this->_n;

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign;
  size_t num_sign_bytes = (num_el + 7) / 8;
  cudaMalloc(&sign, num_sign_bytes);
  CUDA_CHECK_ERR();

  // Fill sign bits, assigning each thread a multiple of 8 elements.
  int num_chars = num_el / 8;
  int grid_size = cml::calc_grid_dim(num_chars, cml::kBlockSize);
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
  // of sign bits to 
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
  for (unsigned int k = 0; k < kEquilIter; ++k) {
    Mul('t', static_cast<T>(1.), d_vec.data, static_cast<T>(0.), e_vec.data);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    // TODO: Figure out a better value for this constant
    cml::vector_add_constant(&e_vec, static_cast<T>(1e-4));
    cudaDeviceSynchronize();
    thrust::transform(thrust::device_pointer_cast(e_vec.data),
        thrust::device_pointer_cast(e_vec.data + e_vec.size),
        thrust::device_pointer_cast(e_vec.data), ReciprF<T>(this->_m));
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    Mul('n', static_cast<T>(1.), e_vec.data, static_cast<T>(0.), d_vec.data);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    cml::vector_add_constant(&d_vec, static_cast<T>(1e-4));
    cudaDeviceSynchronize();
    thrust::transform(thrust::device_pointer_cast(d_vec.data),
        thrust::device_pointer_cast(d_vec.data + d_vec.size),
        thrust::device_pointer_cast(d_vec.data), ReciprF<T>(this->_n));
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

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
  
  // Compute D * A * E
  MultDiag(d_vec.data, e_vec.data, this->_m, this->_n, _ord, _data);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // Scale A to have norm of 1 (in the kNormNormalize norm).
  T normA = NormEst(hdl, kNormNormalize, this->_m, this->_n, _ord, _data);
  CUDA_CHECK_ERR();
  cudaDeviceSynchronize();
  cml::vector<T> a_vec = cml::vector_view_array(_data, num_el);
  cml::vector_scale(&a_vec, 1 / normA);
  cudaDeviceSynchronize();

  // Scale d and e to account for normalization of A.
  T normd = cml::blas_nrm2(hdl, &d_vec);
  T norme = cml::blas_nrm2(hdl, &e_vec);
  T scale = sqrt(normd * sqrt(e_vec.size) / (norme * sqrt(d_vec.size)));
  cml::vector_scale(&d_vec, 1 / (scale * sqrt(normA)));
  cudaDeviceSynchronize();
  cml::vector_scale(&e_vec, scale / sqrt(normA));
  cudaDeviceSynchronize();

  cudaFree(sign);
  CUDA_CHECK_ERR();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

template <typename T, CBLAS_ORDER O>
T Norm2Est(cublasHandle_t hdl, const cml::matrix<T, O> *A) {
  // Same as MATLAB's method for norm estimation.

  T kTol = static_cast<T>(kNormEstTol);

  T norm_est = 0, norm_est_last;
  cml::vector<T> x = cml::vector_alloc<T>(A->size2);
  cml::vector<T> Sx = cml::vector_alloc<T>(A->size1);
  cml::rand(x.data, x.size);
  cudaDeviceSynchronize();

  unsigned int i = 0;
  for (i = 0; i < kNormEstMaxIter; ++i) {
    norm_est_last = norm_est;
    cml::blas_gemv(hdl, CUBLAS_OP_N, static_cast<T>(1.), A, &x,
        static_cast<T>(0.), &Sx);
    cudaDeviceSynchronize();
    cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), A, &Sx,
        static_cast<T>(0.), &x);
    cudaDeviceSynchronize();
    T normx = cml::blas_nrm2(hdl, &x);
    T normSx = cml::blas_nrm2(hdl, &Sx);
    cml::vector_scale(&x, 1 / normx);
    norm_est = normx / normSx;
    if (abs(norm_est_last - norm_est) < kTol * norm_est)
      break;
  }
  DEBUG_EXPECT_LT(i, kNormEstMaxIter);

  cml::vector_free(&x);
  cml::vector_free(&Sx);
  return norm_est;
}

// RMS-Frobenius norm = \sqrt(\sum_i \sigma_i^2 / min(m, n))
template <typename T, CBLAS_ORDER O>
T NormFroEst(cublasHandle_t hdl, const cml::matrix<T, O> *A) {
  const cml::vector<T> a = cml::vector_view_array(A->data, A->size1 * A->size2);
  T norm_est = cml::blas_nrm2(hdl, &a) /
      std::sqrt(std::min(A->size1, A->size2));
  return norm_est;
}

// Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, size_t m, size_t n,
          typename MatrixDense<T>::Ord ord, const T *data) {
  DEBUG_EXPECT_NEQ(norm_type, kNorm1);
  T norm = static_cast<T>(0.);
  switch (norm_type) {
   case kNorm1:
     // Normalize by the 2-norm. 1-norm normalization doens't make
     // make sense since it treats rows and columns differently.
   case kNorm2: {
     if (ord == MatrixDense<T>::ROW) {
       cml::matrix<T, CblasRowMajor> A =
           cml::matrix_view_array<T, CblasRowMajor>(data, m, n);
       norm = Norm2Est(hdl, &A);
     } else {
       cml::matrix<T, CblasColMajor> A =
           cml::matrix_view_array<T, CblasColMajor>(data, m, n);
       norm = Norm2Est(hdl, &A);
     }
     break;
   }
   case kNormFro: {
     if (ord == MatrixDense<T>::ROW) {
       cml::matrix<T, CblasRowMajor> A =
           cml::matrix_view_array<T, CblasRowMajor>(data, m, n);
       norm = NormFroEst(hdl, &A);
     } else {
       cml::matrix<T, CblasColMajor> A =
           cml::matrix_view_array<T, CblasColMajor>(data, m, n);
       norm = NormFroEst(hdl, &A);
     }
     break;
   }
  }
  return norm;
}

// Performs D * A * E for A in row major
template <typename T>
void __global__ __MultRow(size_t m, size_t n, const T *d, const T *e, T *data) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < m * n; t += gridDim.x * blockDim.x)
    data[t] *= d[t / n] * e[t % n];
}

// Performs D * A * E for A in col major
template <typename T>
void __global__ __MultCol(size_t m, size_t n, const T *d, const T *e, T *data) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < m * n; t += gridDim.x * blockDim.x)
    data[t] *= d[t % m] * e[t / m];
}

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDense<T>::Ord ord, T *data) {
  if (ord == MatrixDense<T>::ROW) {
    size_t grid_dim_row = cml::calc_grid_dim(m * n, cml::kBlockSize);
    __MultRow<<<grid_dim_row, cml::kBlockSize>>>(m, n, d, e, data);
  } else {
    size_t grid_dim_row = cml::calc_grid_dim(m * n, cml::kBlockSize);
    __MultCol<<<grid_dim_row, cml::kBlockSize>>>(m, n, d, e, data);
  }
}

}  // namespace

// Explicit template instantiation.
template class MatrixDense<double>;
template class MatrixDense<float>;

}  // _namespace pogs

