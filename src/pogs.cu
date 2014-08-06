#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <vector>

#include "_interface_defs.h"
#include "cml/cml_blas.cuh"
#include "cml/cml_linalg.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_vector.cuh"
#include "matrix_util.h"
#include "pogs.h"
#include "sinkhorn_knopp.cuh"

// Apply operator to h.a and h.d.
template <typename T, typename Op>
struct ApplyOp: thrust::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  __device__ FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x); h.d = binary_op(h.d, x);
    return h;
  }
};

// Proximal Operator Graph Solver.
template <typename T, typename M>
int Pogs(PogsData<T, M> *pogs_data) {
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kDeltaMax = static_cast<T>(2);
  const T kGamma = static_cast<T>(1.01);
  const T kTau = static_cast<T>(0.8);
  const T kAlpha = static_cast<T>(1.7);
  const T kKappa = static_cast<T>(0.9);

  int err = 0;

  // Extract values from pogs_data
  size_t m = pogs_data->m, n = pogs_data->n, min_dim = std::min(m, n);
  T rho = pogs_data->rho;
  const T kOne = static_cast<T>(1), kZero = static_cast<T>(0);
  thrust::device_vector<FunctionObj<T> > f = pogs_data->f;
  thrust::device_vector<FunctionObj<T> > g = pogs_data->g;

  // Create cuBLAS hdl.
  cublasHandle_t hdl;
  cublasCreate(&hdl);

  // Allocate data for ADMM variables.
  bool compute_factors = true;
  cml::vector<T> de, z, zt;
  cml::vector<T> zprev = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
  cml::matrix<T, M::Ord> A, L;
  if (pogs_data->factors.val != 0) {
    cudaMemcpy(&rho, pogs_data->factors.val, sizeof(T), cudaMemcpyDeviceToHost);
    compute_factors = rho == 0;
    if (compute_factors)
      rho = pogs_data->rho;
    de = cml::vector_view_array(pogs_data->factors.val + 1, m + n);
    z = cml::vector_view_array(pogs_data->factors.val + 1 + m + n, m + n);
    zt = cml::vector_view_array(pogs_data->factors.val + 1 + 2 * (m + n),
        m + n);
    L = cml::matrix_view_array<T, M::Ord>(
        pogs_data->factors.val + 1 + 3 * (m + n), min_dim, min_dim);
    A = cml::matrix_view_array<T, M::Ord>(
        pogs_data->factors.val + 1 + 3 * (m + n) + min_dim * min_dim, m, n);
  } else {
    de = cml::vector_calloc<T>(m + n);
    z = cml::vector_calloc<T>(m + n);
    zt = cml::vector_calloc<T>(m + n);
    L = cml::matrix_alloc<T, M::Ord>(min_dim, min_dim);
    A = cml::matrix_alloc<T, M::Ord>(m, n);
  }

  if (de.data == 0 || z.data == 0 || zt.data == 0 || zprev.data == 0 ||
      z12.data == 0 || A.data == 0 || L.data == 0)
    err = 1;

  // Create views for x and y components.
  cml::vector<T> d = cml::vector_subvector(&de, 0, m);
  cml::vector<T> e = cml::vector_subvector(&de, m, n);
  cml::vector<T> x = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y = cml::vector_subvector(&z, n, m);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n, m);

  if (compute_factors && !err) {
    // Copy A to device (assume input row-major).
    cml::matrix_memcpy(&A, pogs_data->A.val);
    err = Equilibrate(hdl, &A, &d, &e);

    if (!err) {
      // Compuate A^TA or AA^T.
      cublasOperation_t op_type = m >= n ? CUBLAS_OP_T : CUBLAS_OP_N;
      cml::blas_syrk(hdl, CUBLAS_FILL_MODE_LOWER, op_type, kOne, &A, kZero, &L);

      // Scale A.
      cml::vector<T> diag_L = cml::matrix_diagonal(&L);
      T mean_diag = cml::blas_asum(hdl, &diag_L) / static_cast<T>(min_dim);
      T sqrt_mean_diag = sqrt(mean_diag);
      cml::matrix_scale(&L, kOne / mean_diag);
      cml::matrix_scale(&A, kOne / sqrt_mean_diag);
      T factor = sqrt(cml::blas_nrm2(hdl, &d) * sqrt(static_cast<T>(n)) /
                     (cml::blas_nrm2(hdl, &e) * sqrt(static_cast<T>(m))));
      cml::blas_scal(hdl, kOne / (factor * sqrt(sqrt_mean_diag)), &d);
      cml::blas_scal(hdl, factor / sqrt(sqrt_mean_diag), &e);

      // Compute cholesky decomposition of (I + A^TA) or (I + AA^T)
      cml::vector_add_constant(&diag_L, kOne);
      cml::linalg_cholesky_decomp(hdl, &L);
    }
  }

  // Scale f and g to account for diagonal scaling e and d.
  if (!err) {
    thrust::transform(f.begin(), f.end(), thrust::device_pointer_cast(d.data),
        f.begin(), ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
    thrust::transform(g.begin(), g.end(), thrust::device_pointer_cast(e.data),
        g.begin(), ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));
  }

  // Signal start of execution.
  if (!pogs_data->quiet)
    Printf("   #      res_pri    eps_pri   res_dual   eps_dua"
           "        gap    eps_gap  objective\n");

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * pogs_data->abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * pogs_data->abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * pogs_data->abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0, ku = 0;
  bool converged;

  for (unsigned int k = 0; k < pogs_data->max_iter && !err; ++k) {
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    cml::blas_axpy(hdl, -kOne, &zt, &z);
    ProxEval(g, rho, x.data, x.stride, x12.data, x12.stride);
    ProxEval(f, rho, y.data, y.stride, y12.data, y12.stride);

    // Compute Gap.
    T gap, nrm_r = 0, nrm_s = 0;
    cml::blas_axpy(hdl, -kOne, &z12, &z);
    cml::blas_dot(hdl, &z, &z12, &gap);
    gap = rho * fabs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * cml::blas_nrm2(hdl, &z12);
    T eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * cml::blas_nrm2(hdl, &z);
    T eps_gap = sqrtmn_atol + pogs_data->rel_tol * fabs(pogs_data->optval);

    if (converged)
      break;

    // Project and Update Dual Variables
    if (m >= n) {
      cml::blas_gemv(hdl, CUBLAS_OP_T, -kOne, &A, &y, -kOne, &x);
      nrm_s = rho * cml::blas_nrm2(hdl, &x);
      cml::linalg_cholesky_svx(hdl, &L, &x);
      cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &x, kZero, &y);
      cml::blas_axpy(hdl, kOne, &zprev, &z);
    } else {
      cml::vector_memcpy(&z, &z12);
      cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &x, -kOne, &y);
      nrm_r = cml::blas_nrm2(hdl, &y);
      cml::linalg_cholesky_svx(hdl, &L, &y);
      cml::blas_gemv(hdl, CUBLAS_OP_T, -kOne, &A, &y, kZero, &x);
      cml::blas_axpy(hdl, kOne, &z12, &z);
    }

    // Apply over relaxation.
    cml::blas_scal(hdl, kAlpha, &z);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &z);

    // Update dual variable.
    cml::blas_axpy(hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(hdl, -kOne, &z, &zt);

    if (nrm_s < eps_dua && m >= n) {
      cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &x12, -kOne, &y12);
      nrm_r = cml::blas_nrm2(hdl, &y12);
      printf("a");
    } else if (nrm_r < eps_pri && m < n) {
      cml::blas_axpy(hdl, -kOne, &zprev, &z12);
      cml::blas_gemv(hdl, CUBLAS_OP_T, kOne, &A, &y12, kOne, &x12);
      nrm_s = rho * cml::blas_nrm2(hdl, &x12);
      printf("b\n");
    } else {
      // Compute upper bounds on residuals
      cml::blas_axpy(hdl, -kOne, &z, &z12);
      cml::blas_axpy(hdl, -kOne, &z, &zprev);
      nrm_r = cml::blas_nrm2(hdl, &z12);
      nrm_s = rho * cml::blas_nrm2(hdl, &zprev);
    }

    // Evaluate stopping criteria.
    converged = nrm_r < eps_pri && nrm_s < eps_dua && gap < eps_gap;
    if (!pogs_data->quiet && (k % 10 == 0 || converged))
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, pogs_data->optval);

    // Rescale rho.
    if (pogs_data->adaptive_rho && !converged) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        rho *= delta;
        cml::blas_scal(hdl, 1 / delta, &zt);
        delta = std::min(kGamma * delta, kDeltaMax);
        ku = k;
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        rho /= delta;
        cml::blas_scal(hdl, delta, &zt);
        delta = std::min(kGamma * delta, kDeltaMax);
        kd = k;
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = std::max(delta / kGamma, kDeltaMin);
      }
    }
  }
  // Scale x, y and l for output.
  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);
  cml::vector_mul(&y, &d);
  cml::blas_scal(hdl, rho, &y);

  // Copy results to output.
  if (pogs_data->y != 0 && !err)
    cml::vector_memcpy(pogs_data->y, &y12);
  if (pogs_data->x != 0 && !err)
    cml::vector_memcpy(pogs_data->x, &x12);
  if (pogs_data->l != 0 && !err)
    cml::vector_memcpy(pogs_data->l, &y);

  // Store rho and free memory.
  if (pogs_data->factors.val != 0 && !err) {
    cudaMemcpy(pogs_data->factors.val, &rho, sizeof(T), cudaMemcpyHostToDevice);
  } else {
    cml::vector_free(&de);
    cml::vector_free(&z);
    cml::vector_free(&zt);
    cml::matrix_free(&L);
    cml::matrix_free(&A);
  }
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  return err;
}

template <typename T, CBLAS_ORDER O>
int AllocDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n;
  size_t flen = 1 + 3 * (m + n) + std::min(m, n) * std::min(m, n) + m * n;
  cudaError_t err = cudaMalloc(&pogs_data->factors.val, flen * sizeof(T));
  if (err == cudaSuccess)
    err = cudaMemset(pogs_data->factors.val, 0, flen * sizeof(T));
  if (err == cudaSuccess)
    return 0;
  else
    return 1;
}

template <typename T, CBLAS_ORDER O>
void FreeDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  cudaFree(pogs_data->factors.val);
}

// Declarations.
template int Pogs<double, Dense<double, CblasRowMajor> >
    (PogsData<double, Dense<double, CblasRowMajor> > *);
template int Pogs<double, Dense<double, CblasColMajor> >
    (PogsData<double, Dense<double, CblasColMajor> > *);
template int Pogs<float, Dense<float, CblasRowMajor> >
    (PogsData<float, Dense<float, CblasRowMajor> > *);
template int Pogs<float, Dense<float, CblasColMajor> >
    (PogsData<float, Dense<float, CblasColMajor> > *);

template int AllocDenseFactors<double, CblasRowMajor>
    (PogsData<double, Dense<double, CblasRowMajor> > *);
template int AllocDenseFactors<double, CblasColMajor>
    (PogsData<double, Dense<double, CblasColMajor> > *);
template int AllocDenseFactors<float, CblasRowMajor>
    (PogsData<float, Dense<float, CblasRowMajor> > *);
template int AllocDenseFactors<float, CblasColMajor>
    (PogsData<float, Dense<float, CblasColMajor> > *);

template void FreeDenseFactors<double, CblasRowMajor>
    (PogsData<double, Dense<double, CblasRowMajor> > *);
template void FreeDenseFactors<double, CblasColMajor>
    (PogsData<double, Dense<double, CblasColMajor> > *);
template void FreeDenseFactors<float, CblasRowMajor>
    (PogsData<float, Dense<float, CblasRowMajor> > *);
template void FreeDenseFactors<float, CblasColMajor>
    (PogsData<float, Dense<float, CblasColMajor> > *);

