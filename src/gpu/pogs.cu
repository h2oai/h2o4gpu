#include "pogs.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>

#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "interface_defs.h"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector.h"
#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "util.cuh"

namespace pogs {

namespace {

template <typename T, typename Op>
struct ApplyOp: thrust::binary_function<FunctionObj<T>, FunctionObj<T>, T> {   
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  __host__ __device__ FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x);
    h.d = binary_op(h.d, x);
    h.e = binary_op(binary_op(h.e, x), x);
    return h;
  }
};

}  // namespace

template <typename T, typename M, typename P>
Pogs<T, M, P>::Pogs(const M &A)
    : _A(A), _P(_A),
      _done_init(false),
      _x(0), _y(0), _mu(0), _lambda(0), _optval(static_cast<T>(0.)),
      _de(0), _z(0), _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _gap_stop(kGapStop) {
  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
}

template <typename T, typename M, typename P>
int Pogs<T, M, P>::_Init() {
  DEBUG_ASSERT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;

  size_t m = _A.Rows();
  size_t n = _A.Cols();
  size_t min_dim = std::min(m, n);

  cudaMalloc(&_de, (m + n) * sizeof(T));
  cudaMalloc(&_z, (m + n) * sizeof(T));
  cudaMalloc(&_zt, (m + n) * sizeof(T));
  cudaMemset(_de, 0, (m + n) * sizeof(T));
  cudaMemset(_z, 0, (m + n) * sizeof(T));
  cudaMemset(_zt, 0, (m + n) * sizeof(T));
  XDEBUG_CUDA_CHECK_ERR();

  _A.Init();
  _A.Equil(_de, _de + m);
  _P.Init();
  XDEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T, typename M, typename P>
int Pogs<T, M, P>::Solve(const std::vector<FunctionObj<T> > &f,
                         const std::vector<FunctionObj<T> > &g) {
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kGamma    = static_cast<T>(1.01);
  const T kTau      = static_cast<T>(0.5);
  const T kAlpha    = static_cast<T>(1.7);
  const T kRhoMin   = static_cast<T>(1e-3);
  const T kRhoMax   = static_cast<T>(1e3);
  const T kKappa    = static_cast<T>(0.9);
  const T kOne      = static_cast<T>(1.0);
  const T kZero     = static_cast<T>(0.0);

  // Initialize Projector P and Matrix A.
  if (!_done_init)
    _Init();

  // Extract values from pogs_data
  size_t m = _A.Rows();
  size_t n = _A.Cols();
  size_t min_dim = std::min(m, n);
  thrust::device_vector<FunctionObj<T> > f_gpu = f;
  thrust::device_vector<FunctionObj<T> > g_gpu = g;

  // Create cuBLAS handle.
  cublasHandle_t hdl;
  cublasCreate(&hdl);
  XDEBUG_CUDA_CHECK_ERR();

  // Allocate data for ADMM variables.
  cml::vector<T> de = cml::vector_view_array(_de, m + n);
  cml::vector<T> z = cml::vector_view_array(_z, m + n);
  cml::vector<T> zt = cml::vector_view_array(_zt, m + n);
  cml::vector<T> zprev = cml::vector_calloc<T>(m + n);
  cml::vector<T> ztemp = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
  XDEBUG_CUDA_CHECK_ERR();

  // Create views for x and y components.
  cml::vector<T> d = cml::vector_subvector(&de, 0, m);
  cml::vector<T> e = cml::vector_subvector(&de, m, n);
  cml::vector<T> x = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y = cml::vector_subvector(&z, n, m);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n, m);
  cml::vector<T> xprev = cml::vector_subvector(&zprev, 0, n);
  cml::vector<T> yprev = cml::vector_subvector(&zprev, n, m);
  cml::vector<T> xtemp = cml::vector_subvector(&ztemp, 0, n);
  cml::vector<T> ytemp = cml::vector_subvector(&ztemp, n, m);
  XDEBUG_CUDA_CHECK_ERR();

  // TODO: Use some form of init y maybe?
  // Initialize x and y from x0 or y0
  cml::vector_memcpy(&yprev, _y);
  cml::vector_mul(&yprev, &d);
  cml::vector_memcpy(&xprev, _x);
  cml::vector_div(&xprev, &e);
  cml::vector_set_all(&x, kZero);
  XDEBUG_CUDA_CHECK_ERR();
  _P.Project(yprev.data, xprev.data, kOne, x.data, y.data);
  cudaDeviceSynchronize();
  XDEBUG_CUDA_CHECK_ERR();

  // TODO: initialize from _mu and _lambda and check if ||A^Tmu+lambda|| < tol.
//  cml::vector_memcpy(&xt, _mu);
//  cml::vector_memcpy(&yt, _lambda);
//  cml::vector_scal(&zt, -kOne / _rho);
//  cml::blas_axpy(hdl, kOne, &z, &zt);
//  ProxEval(g_gpu, _rho, x.data, x12.data);
//  ProxEval(f_gpu, _rho, y.data, y12.data);
//  cml::blas_axpy(hdl, -kOne, &z12, &zt);

  // Scale f and g to account for diagonal scaling e and d.
  thrust::transform(f_gpu.begin(), f_gpu.end(),
      thrust::device_pointer_cast(d.data), f_gpu.begin(),
      ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
  thrust::transform(g_gpu.begin(), g_gpu.end(),
      thrust::device_pointer_cast(e.data), g_gpu.begin(),
      ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));
  XDEBUG_CUDA_CHECK_ERR();

  // Signal start of execution.
  if (_verbose > 0) {
    Printf("   #      res_pri    eps_pri   res_dual   eps_dual"
        "        gap    eps_gap  objective\n");
  }

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * _abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * _abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * _abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0u, ku = 0u;
  bool converged = false;

  for (unsigned int k = 0;; ++k) {
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    cml::blas_axpy(hdl, -kOne, &zt, &z);
    ProxEval(g_gpu, _rho, x.data, x12.data);
    ProxEval(f_gpu, _rho, y.data, y12.data);
    XDEBUG_CUDA_CHECK_ERR();

    // Compute dual variable.
    T gap;
    cml::blas_axpy(hdl, -kOne, &z12, &z);
    cml::blas_dot(hdl, &z, &z12, &gap);
    gap = std::abs(gap);
    _optval = FuncEval(f_gpu, y12.data) + FuncEval(g_gpu, x12.data);
    T eps_gap = sqrtmn_atol + _rel_tol * cml::blas_nrm2(hdl, &z) *
        cml::blas_nrm2(hdl, &z12);
    T eps_pri = sqrtm_atol + _rel_tol * cml::blas_nrm2(hdl, &z12);
    T eps_dua = sqrtn_atol + _rel_tol * _rho * cml::blas_nrm2(hdl, &z);
    XDEBUG_CUDA_CHECK_ERR();

    if (converged || k == _max_iter)
      break;

    // Project onto y = Ax.
    _P.Project(x12.data, y12.data, kOne, x.data, y.data);
    cudaDeviceSynchronize();
    XDEBUG_CUDA_CHECK_ERR();

    // Apply over relaxation.
    cml::blas_scal(hdl, kAlpha, &z);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &z);
    XDEBUG_CUDA_CHECK_ERR();

    // Calculate residuals.
    cml::vector_memcpy(&ztemp, &zprev);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    T nrm_s = _rho * cml::blas_nrm2(hdl, &ztemp);

    cml::vector_memcpy(&ztemp, &z12);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    T nrm_r = cml::blas_nrm2(hdl, &ztemp);

    // Calculate exact residuals only if necessary.
    bool exact = false;
    if (nrm_r < eps_pri && nrm_s < eps_dua || true) {
      cml::vector_memcpy(&ztemp, &z12);
      _A.Mul('n', kOne, x12.data, -kOne, ytemp.data);
      cudaDeviceSynchronize();
      nrm_r = cml::blas_nrm2(hdl, &ytemp);
      if (nrm_r < eps_pri || true) {
        cml::vector_memcpy(&ztemp, &z12);
        cml::blas_axpy(hdl, kOne, &zt, &ztemp);
        cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
        _A.Mul('t', kOne, ytemp.data, kOne, xtemp.data);
        cudaDeviceSynchronize();
        nrm_s = _rho * cml::blas_nrm2(hdl, &xtemp);
        exact = true;
      }
    }
    XDEBUG_CUDA_CHECK_ERR();

    // Update dual variable.
    cml::blas_axpy(hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(hdl, -kOne, &z, &zt);
    XDEBUG_CUDA_CHECK_ERR();

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!_gap_stop || gap < eps_gap);
    if (_verbose > 0 && (k % 10 == 0 || converged)) {
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, _optval);
    }

    // Rescale rho.
    if (_adaptive_rho && !converged) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (_rho < kRhoMax) {
          _rho *= delta;
          cml::blas_scal(hdl, 1 / delta, &zt);
          delta = kGamma * delta;
          ku = k;
          if (_verbose > 1)
            Printf("+ rho %e\n", _rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (_rho > kRhoMin) {
          _rho /= delta;
          cml::blas_scal(hdl, delta, &zt);
          delta = kGamma * delta;
          kd = k;
          if (_verbose > 1)
            Printf("- rho %e\n", _rho);
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = kDeltaMin;
      }
      XDEBUG_CUDA_CHECK_ERR();
    }
  }

  // Scale x, y and l for output.
  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);
  cml::vector_mul(&y, &d);
  cml::blas_scal(hdl, _rho, &z);

  // Copy results to output.
  cml::vector_memcpy(_x, &x12);
  cml::vector_memcpy(_y, &y12);
  cml::vector_memcpy(_mu, &x);
  cml::vector_memcpy(_lambda, &y);

  // Store rho and free memory.
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  cml::vector_free(&ztemp);
  cublasDestroy(hdl);
  XDEBUG_CUDA_CHECK_ERR();
  DEBUG_PRINT("Finished Execution");

  return 0;
}

template <typename T, typename M, typename P>
Pogs<T, M, P>::~Pogs() {
  cudaFree(_de);
  cudaFree(_z);
  cudaFree(_zt);
  _de = _z = _zt = 0;
  XDEBUG_CUDA_CHECK_ERR();

  _A.Free();
  XDEBUG_CUDA_CHECK_ERR();
  _P.Free();
  XDEBUG_CUDA_CHECK_ERR();

  delete [] _x;
  delete [] _y;
  delete [] _mu;
  delete [] _lambda;
  _x = _y = _mu = _lambda = 0;
}


// Explicit template instantiation.
// Dense direct.
template class Pogs<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class Pogs<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;

// TODO: add cgls projector
// // Dense indirect.
// template class Pogs<double, MatrixDense<double>,
//     ProjectorCgls<double, MatrixDense<double> > >;
// template class Pogs<float, MatrixDense<float>,
//     ProjectorCgls<float, MatrixDense<float> > >;
// 
// // Sparse indirect.
// template class Pogs<double, MatrixSparse<double>,
//     ProjectorCgls<double, MatrixSparse<double> > >;
// template class Pogs<float, MatrixSparse<float>,
//     ProjectorCgls<float, MatrixSparse<float> > >;

}  // namespace pogs

