#include <cusparse.h>
#include <cublas_v2.h>

#include "sinkhorn_knopp.cuh"

#include <cmath>
#include <algorithm>
#include <vector>

#include "_interface_defs.h"
#include "cml/cml_linalg.cuh"
#include "cml/cml_spblas.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_vector.cuh"
#include "pogs.h"

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
template<typename T, typename M>
int Pogs(PogsData<T, M> *pogs_data) {
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kGamma = static_cast<T>(1.01);
  const T kTau = static_cast<T>(0.8);
  const T kAlpha = static_cast<T>(1.7);
  const T kKappa = static_cast<T>(0.9);
  const T kOne = static_cast<T>(1);
  const T kZero = static_cast<T>(0);
  const T kTol = static_cast<T>(1e-8);
  const CBLAS_ORDER kOrd = M::Ord == ROW ? CblasRowMajor : CblasColMajor;

  int err = 0;

  // Extract values from pogs_data
  int m = pogs_data->m, n = pogs_data->n, min_dim = std::min(m, n);
  int nnz = pogs_data->A.nnz;
  T rho = pogs_data->rho;
  thrust::device_vector<FunctionObj<T> > f = pogs_data->f;
  thrust::device_vector<FunctionObj<T> > g = pogs_data->g;
 
  // Create cuBLAS hdl.
  cublasHandle_t b_hdl;
  cublasCreate(&b_hdl);
  cusparseHandle_t s_hdl;
  cusparseCreate(&s_hdl);
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  // Allocate data for ADMM variables.
  bool pre_process = true;
  cml::vector<T> de, z, zt;
  cml::vector<T> zprev = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
  cml::spmat<T, typename M::I_t, kOrd> A;
  if (pogs_data->factors.val != 0) {
    cudaMemcpy(&rho, pogs_data->factors.val, sizeof(T), cudaMemcpyDeviceToHost);
    pre_process = (rho == 0);
    if (pre_process)
      rho = pogs_data->rho;
    de = cml::vector_view_array(pogs_data->factors.val + 1, m + n);
    z = cml::vector_view_array(pogs_data->factors.val + 1 + m + n, m + n);
    zt = cml::vector_view_array(pogs_data->factors.val + 1 + 2 * (m + n),
        m + n);
    A = cml::spmat<T, typename M::I_t, kOrd>(
        pogs_data->factors.val + 1 + 3 * (m + n),
        pogs_data->factors.ind, pogs_data->factors.ptr, m, n,
        pogs_data->factors.nnz);
  } else {
    de = cml::vector_calloc<T>(m + n);
    z = cml::vector_calloc<T>(m + n);
    zt = cml::vector_calloc<T>(m + n);
    A = cml::spmat_alloc<T, typename M::I_t, kOrd>(m, n, nnz);
  }

  if (de.data == 0 || z.data == 0 || zt.data == 0 || zprev.data == 0 ||
      z12.data == 0 || A.val == 0 || A.ind == 0 || A.ptr == 0) {
    err = 1;
  }

  // Create views for x and y components.
  cml::vector<T> d = cml::vector_subvector(&de, 0, m);
  cml::vector<T> e = cml::vector_subvector(&de, m, n);
  cml::vector<T> x = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y = cml::vector_subvector(&z, n, m);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n, m);

  if (pre_process && !err) {
    // Copy A to device (assume input row-major).
    cml::spmat_memcpy(s_hdl, &A, pogs_data->A.val, pogs_data->A.ind,
        pogs_data->A.ptr);
    cml::vector_set_all(&de, kOne);
    err = sinkhorn_knopp::Equilibrate(s_hdl, b_hdl, descr, &A, &d, &e);
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
    Printf("   #      res_pri    eps_pri   res_dual   eps_dual"
           "        gap    eps_gap  objective\n");

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * pogs_data->abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * pogs_data->abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0, ku = 0;
  bool converged = false;

  Printf("abstol = %e, reltol = %e\n", pogs_data->abs_tol, pogs_data->rel_tol);

  for (unsigned int k = 0; k < pogs_data->max_iter && !err; ++k) {
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    cml::blas_axpy(b_hdl, -kOne, &zt, &z);
    ProxEval(g, rho, x.data, x.stride, x12.data, x12.stride);
    ProxEval(f, rho, y.data, y.stride, y12.data, y12.stride);

    // Compute dual variable.
    T nrm_r = 0, nrm_s = 0, gap;
    cml::blas_axpy(b_hdl, -kOne, &z12, &z);
    cml::blas_dot(b_hdl, &z, &z12, &gap);
    gap = std::abs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    T eps_gap = pogs_data->abs_tol + pogs_data->rel_tol *
        cml::blas_nrm2(b_hdl, &z) * cml::blas_nrm2(b_hdl, &z12);
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * cml::blas_nrm2(b_hdl, &z12);
    T eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * 
        cml::blas_nrm2(b_hdl, &z);

    if (converged || k == pogs_data->max_iter - 1)
      break;

    // Project and Update Dual Variables
    cml::vector_memcpy(&y, &y12);
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, -kOne, &A,
        &x12, kOne, &y);
    nrm_r = cml::blas_nrm2(b_hdl, &y);
    cml::vector_set_all(&x, kZero);
    cml::spblas_solve(s_hdl, b_hdl, descr, &A, kOne, &y, &x, kTol, 100, true);
    cml::blas_axpy(b_hdl, kOne, &x12, &x);
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, kOne, &A,
        &x, kZero, &y);

    // Apply over relaxation.
    cml::blas_scal(b_hdl, kAlpha, &z);
    cml::blas_axpy(b_hdl, kOne - kAlpha, &zprev, &z);

    // Update dual variable.
    cml::blas_axpy(b_hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(b_hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(b_hdl, -kOne, &z, &zt);

    bool exact = false;
    cml::blas_axpy(b_hdl, -kOne, &zprev, &z12);
    cml::blas_axpy(b_hdl, -kOne, &z, &zprev);
    nrm_s = rho * cml::blas_nrm2(b_hdl, &zprev);
    if (nrm_r < eps_pri && nrm_s < eps_dua) {
      cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_TRANSPOSE, descr, kOne, &A,
          &y12, kOne, &x12);
      nrm_s = rho * cml::blas_nrm2(b_hdl, &x12);
      exact = true;
    }

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua && gap < eps_gap;
    if (!pogs_data->quiet && (k % 10 == 0 || converged))
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, pogs_data->optval);

    // Rescale rho.
    if (pogs_data->adaptive_rho && !converged) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        rho *= delta;
        cml::blas_scal(b_hdl, 1 / delta, &zt);
        delta = kGamma * delta;
        ku = k;
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        rho /= delta;
        cml::blas_scal(b_hdl, delta, &zt);
        delta = kGamma * delta;
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
  cml::blas_scal(b_hdl, rho, &y);

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
    cml::vector_memcpy(&z, &zprev);
  } else {
    cml::vector_free(&de);
    cml::vector_free(&z);
    cml::vector_free(&zt);
    cml::spmat_free(&A);
  }
  cml::vector_free(&z12);
  cml::vector_free(&zprev);

  return err;
}

// Declarations.
template int Pogs<double, Sparse<double, int, COL> >
    (PogsData<double, Sparse<double, int, COL> > *);
template int Pogs<double, Sparse<double, int, ROW> >
    (PogsData<double, Sparse<double, int, ROW> > *);
template int Pogs<float, Sparse<float, int, COL> >
    (PogsData<float, Sparse<float, int, COL> > *);
template int Pogs<float, Sparse<float, int, ROW> >
    (PogsData<float, Sparse<float, int, ROW> > *);

