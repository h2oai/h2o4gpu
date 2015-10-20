// TODO: move include once nvidia engineers get around to fixing their bugs
#include "sinkhorn_knopp.cuh"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>

#include "_interface_defs.h"
#include "cml/cml_blas.cuh"
#include "cml/cml_linalg.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_vector.cuh"
#include "matrix_util.h"
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
template <typename T, typename M>
int Pogs(PogsData<T, M> *pogs_data) {
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kDeltaMax = static_cast<T>(2);
  const T kGamma = static_cast<T>(1.01);
  const T kTau = static_cast<T>(0.8);
  const T kAlpha = static_cast<T>(1.7);
  const T kKappa = static_cast<T>(0.9);
  const CBLAS_ORDER kOrd = M::Ord == ROW ? CblasRowMajor : CblasColMajor;

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
  // cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
  // cml::vector<T> zt12 = cml::vector_calloc<T>(m+n);
  cml::vector<T> z12, zt12;
  cml::vector<T> ztemp = cml::vector_calloc<T>(m+n);
  cml::vector<T> w_ = cml::vector_calloc<T>(m+n);
  T *w = (T *) calloc(m, sizeof(T));
  cml::matrix<T, kOrd> A, L;
  bool obj_scal = false;
  if (pogs_data->factors.val != 0) {
    cudaMemcpy(&rho, pogs_data->factors.val, sizeof(T), cudaMemcpyDeviceToHost);
    compute_factors = rho == 0;
    if (compute_factors)
      rho = pogs_data->rho;
    de = cml::vector_view_array(pogs_data->factors.val + 1, m + n);
    z = cml::vector_view_array(pogs_data->factors.val + 1 + m + n, m + n);
    zt = cml::vector_view_array(pogs_data->factors.val + 1 + 2 * (m + n),
        m + n);
    z12 = cml::vector_view_array(pogs_data->factors.val + 1 + 3 * (m + n), m + n);
    zt12 = cml::vector_view_array(pogs_data->factors.val + 1 + 4 * (m + n), m + n);
    L = cml::matrix_view_array<T, kOrd>(
        pogs_data->factors.val + 1 + 5 * (m + n), min_dim, min_dim);
    A = cml::matrix_view_array<T, kOrd>(
        pogs_data->factors.val + 1 + 5 * (m + n) + min_dim * min_dim, m, n);
  } else {
    de = cml::vector_calloc<T>(m + n);
    z = cml::vector_calloc<T>(m + n);
    zt = cml::vector_calloc<T>(m + n);
    z12 = cml::vector_calloc<T>(m + n);
    zt12 = cml::vector_calloc<T>(m + n);
    L = cml::matrix_alloc<T, kOrd>(min_dim, min_dim);
    A = cml::matrix_alloc<T, kOrd>(m, n);
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
  cml::vector<T> xprev = cml::vector_subvector(&zprev, 0, n);
  cml::vector<T> yprev = cml::vector_subvector(&zprev, n, m);
  cml::vector<T> xtemp = cml::vector_subvector(&ztemp, 0, n);
  cml::vector<T> ytemp = cml::vector_subvector(&ztemp, n, m);


  if (compute_factors && !err) {
    // Copy A to device (assume input row-major).
    cml::matrix_memcpy(&A, pogs_data->A.val);


    obj_scal = false;

    // SCALE A for values of f.c
    for (unsigned int i = 0; i < m; ++i)
      obj_scal &= static_cast<bool>(pogs_data->f[i].c > 0);

    err = sinkhorn_knopp::Equilibrate(&A, &d, &e);

    if (obj_scal){
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for (unsigned int i = 0; i < m; ++i)
        w[i] = static_cast<T>(1/pogs_data->f[i].c);

      cml::vector_memcpy(&w_, w);
      sinkhorn_knopp::DiagOp<T, kOrd, CblasLeft>(&A, &w_);
    }

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


  // Initialize (x, lambda) from (x0, lambda0).
  
  // Check that both x and lambda provided
  if (pogs_data-> warm_start && !(pogs_data->x && pogs_data->nu)) {
    Printf("\nERROR: Must provide x0 and lambda0 for warm start\n");
    err=1;
  }
  if (!err && pogs_data->warm_start) {
    // x:= x0, y:= A * x
    cml::vector_memcpy(&xprev, pogs_data->x);
    cml::vector_div(&xprev, &e);
    cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &xprev, kZero, &yprev);
    cml::vector_memcpy(&z, &zprev);
    // CUDA_CHECK_ERR();

    // lambda:= lambda0, mu:= -A^T * lambda
    cml::vector_memcpy(&yprev, pogs_data->nu);
    cml::vector_div(&yprev, &d);
    cml::blas_gemv(hdl, CUBLAS_OP_T, -kOne, &A, &yprev, kZero, &xprev);
    cml::blas_scal(hdl, -kOne / rho, &zprev);
    cml::vector_memcpy(&zt, &zprev);
    // CUDA_CHECK_ERR();
  }

  pogs_data->warm_start = false;


  // Signal start of execution.
  if (!err && !pogs_data->quiet)
    Printf("   #      res_pri    eps_pri   res_dual   eps_dual"
           "        gap    eps_gap  objective\n");

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * pogs_data->abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * pogs_data->abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * pogs_data->abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0, ku = 0;
  bool converged = false;

  for (unsigned int k = 0; !err; ++k) {
    cml::vector_memcpy(&zprev, &z);

    // Evaluate proximal operators.
    // ----------------------------

    //  ("x","y"):= (x-\tilde x, y-\tilde y)
    cml::blas_axpy(hdl, -kOne, &zt, &z);

    //  (x^{1/2},y^{1/2}) = (prox_{g,rho}(x-\tilde x), prox_{f,rho}(y-\tilde y))    
    ProxEval(g, rho, x.data, x.stride, x12.data, x12.stride);
    ProxEval(f, rho, y.data, y.stride, y12.data, y12.stride);

    // Compute gap, objective and tolerances.
    // --------------------------------------
    T gap, nrm_r = 0, nrm_s = 0;
    cml::blas_axpy(hdl, -kOne, &z12, &z);
    cml::blas_dot(hdl, &z, &z12, &gap);
    gap = rho * fabs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * cml::blas_nrm2(hdl, &z12);
    T eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * cml::blas_nrm2(hdl, &z);
    T eps_gap = sqrtmn_atol + pogs_data->rel_tol * fabs(pogs_data->optval);

    // Projection & primal update.
    // ---------------------------
    // The projection step is defined as:
    // (x^{1},y^{1}) = PROJ_{y=Ax} (x^{1/2}+\tilde x, y^{1/2} + \tilde y)
    if (m >= n) {

#ifdef CORRECTPROJECTION
      cml::vector_memcpy(&z, &zprev);
      cml::blas_axpy(hdl, kAlpha, &z12, &z);
      cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &z);
      cml::blas_gemv(hdl, CUBLAS_OP_T, kOne, &A, &y, kOne, &x);
#else
      //  for  m>=n, projection  becomes the reduced updates:
      //
      //  x^{1} := (I+AᵀA)⁻¹(x^{1/2}+Aᵀy^{1/2})
      //  y^{1} := Ax^{1}
      cml::blas_gemv(hdl, CUBLAS_OP_T, -kOne, &A, &y, -kOne, &x);
#endif
      nrm_s = rho * cml::blas_nrm2(hdl, &x);
      cml::linalg_cholesky_svx(hdl, &L, &x);
      cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &x, kZero, &y);
      cml::blas_axpy(hdl, kOne, &zprev, &z);
    } else {
      //  for  m<n, projection  becomes the reduced updates:
      //
      //  y^{1} := y^{1/2}-\tilde y+(I+AAᵀ)⁻¹(Ax^{1/2}+A\tilde x-y^{1/2}-\tilde y)
      //  x^{1} := x^{1/2}+\tilde x-Aᵀ(y^{1/2}+\tilde y) + Aᵀy^{1}
      cml::vector_memcpy(&z, &z12);
      cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &x, -kOne, &y);
      nrm_r = cml::blas_nrm2(hdl, &y);
      cml::linalg_cholesky_svx(hdl, &L, &y);
      cml::blas_gemv(hdl, CUBLAS_OP_T, -kOne, &A, &y, kZero, &x);
      cml::blas_axpy(hdl, kOne, &z12, &z);
    }

#ifndef CORRECTPROJECTION
    // Apply over-relaxation.
    // ----------------------
    //  "x" := \alpha x^{1} + (1-alpha)x
    //  "y" := \alpha y^{1} + (1-alpha)y    
    cml::blas_scal(hdl, kAlpha, &z);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &z);
#endif

    // \tilde x^{1/2} := x^{1/2} - x +\tilde x
    // \tilde y^{1/2} := y^{1/2} - y +\tilde y
    cml::vector_memcpy(&zt12, &z12);
    cml::blas_axpy(hdl, -kOne, &zprev, &zt12);
    cml::blas_axpy(hdl, kOne, &zt, &zt12);


    // Update dual variable.
    // ---------------------
    // \tilde x := \tilde x + \alpha (x^{1/2} - x^{1})
    // \tilde y := \tilde y + \alpha (y^{1/2} - y^{1})
    cml::blas_axpy(hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(hdl, -kOne, &z, &zt);

    bool exact = false;
    if (m >= n) {
      cml::vector_memcpy(&zprev, &z12);
      cml::blas_axpy(hdl, -kOne, &z, &zprev);
      nrm_r = cml::blas_nrm2(hdl, &zprev);
      if (nrm_s < eps_dua && nrm_r < eps_pri) {
        cml::blas_gemv(hdl, CUBLAS_OP_N, kOne, &A, &x12, -kOne, &y12);
        nrm_r = cml::blas_nrm2(hdl, &y12);
        exact = true;
      }
    } else {
      cml::blas_axpy(hdl, -kOne, &zprev, &z12);
      cml::blas_axpy(hdl, -kOne, &z, &zprev);
      nrm_s = rho * cml::blas_nrm2(hdl, &zprev);
      if (nrm_r < eps_pri && nrm_s < eps_dua) {
        cml::blas_gemv(hdl, CUBLAS_OP_T, kOne, &A, &y12, kOne, &x12);
        nrm_s = rho * cml::blas_nrm2(hdl, &x12);
        exact = true;
      }
    }

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!pogs_data->gap_stop || gap < eps_gap);
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

    if (converged || k == pogs_data->max_iter){
      if (!converged)
        Printf("Reached max iter=%i\n",pogs_data->max_iter);      
      break;
    }

  }

  // Scale final iterates, copy to output.
  cml::vector_memcpy(&ztemp, &z);
  cml::vector_div(&ytemp, &d);
  if (obj_scal)
    cml::vector_div(&ytemp, &w_);
  cml::vector_mul(&xtemp, &e);
  if (pogs_data->y != 0 && !err)
    cml::vector_memcpy(pogs_data->y, &ytemp);
  if (pogs_data->x != 0 && !err)
    cml::vector_memcpy(pogs_data->x, &xtemp);

  cml::vector_memcpy(&ztemp, &zt);
  cml::vector_mul(&ytemp, &d);
  if (obj_scal)
    cml::vector_mul(&ytemp, &w_);
  cml::blas_scal(hdl, rho, &ytemp);
  cml::vector_div(&xtemp, &e);
  cml::blas_scal(hdl, rho, &xtemp);
  if (pogs_data->nu != 0 && !err)
    cml::vector_memcpy(pogs_data->nu, &ytemp);
  if (pogs_data->mu != 0 && !err)
    cml::vector_memcpy(pogs_data->mu, &xtemp);

  cml::vector_memcpy(&ztemp, &z12);
  cml::vector_div(&ytemp, &d);
  if (obj_scal)
    cml::vector_div(&ytemp, &w_);
  cml::vector_mul(&xtemp, &e);
  if (pogs_data->y12 != 0 && !err)
    cml::vector_memcpy(pogs_data->y12, &ytemp);
  if (pogs_data->x12 != 0 && !err)
    cml::vector_memcpy(pogs_data->x12, &xtemp);

  cml::vector_memcpy(&ztemp, &zt12);
  cml::vector_mul(&ytemp, &d);
  if (obj_scal)
    cml::vector_mul(&ytemp, &w_);
  cml::blas_scal(hdl, rho, &ytemp);
  cml::vector_div(&xtemp, &e);
  cml::blas_scal(hdl, rho, &xtemp);
  if (pogs_data->nu12 != 0 && !err)
    cml::vector_memcpy(pogs_data->nu12, &ytemp);
  if (pogs_data->mu12 != 0 && !err)
    cml::vector_memcpy(pogs_data->mu12, &xtemp);


  // Store rho and free memory.
  if (pogs_data->factors.val != 0 && !err) {
    cudaMemcpy(pogs_data->factors.val, &rho, sizeof(T), cudaMemcpyHostToDevice);
    pogs_data->rho=rho;
    cml::vector_memcpy(&z, &zprev);
  } else {
    cml::vector_free(&de);
    cml::vector_free(&z);
    cml::vector_free(&zt);
    cml::vector_free(&z12);
    cml::vector_free(&zt12);
    cml::matrix_free(&L);
    cml::matrix_free(&A);
  }
  cml::vector_free(&zprev);
  cml::vector_free(&ztemp);
  cml::vector_free(&w_);

  return err;
}

template <typename T, POGS_ORD O>
int AllocDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n;
  size_t flen = 1 + 5 * (m + n) + std::min(m, n) * std::min(m, n) + m * n;
  cudaError_t err = cudaMalloc(&pogs_data->factors.val, flen * sizeof(T));
  if (err == cudaSuccess)
    err = cudaMemset(pogs_data->factors.val, 0, flen * sizeof(T));
  if (err == cudaSuccess)
    return 0;
  else
    return 1;
}

template <typename T, POGS_ORD O>
void FreeDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  cudaFree(pogs_data->factors.val);
}

// Declarations.
template int Pogs<double, Dense<double, ROW> >
    (PogsData<double, Dense<double, ROW> > *);
template int Pogs<double, Dense<double, COL> >
    (PogsData<double, Dense<double, COL> > *);
template int Pogs<float, Dense<float, ROW> >
    (PogsData<float, Dense<float, ROW> > *);
template int Pogs<float, Dense<float, COL> >
    (PogsData<float, Dense<float, COL> > *);

template int AllocDenseFactors<double, ROW>
    (PogsData<double, Dense<double, ROW> > *);
template int AllocDenseFactors<double, COL>
    (PogsData<double, Dense<double, COL> > *);
template int AllocDenseFactors<float, ROW>
    (PogsData<float, Dense<float, ROW> > *);
template int AllocDenseFactors<float, COL>
    (PogsData<float, Dense<float, COL> > *);

template void FreeDenseFactors<double, ROW>
    (PogsData<double, Dense<double, ROW> > *);
template void FreeDenseFactors<double, COL>
    (PogsData<double, Dense<double, COL> > *);
template void FreeDenseFactors<float, ROW>
    (PogsData<float, Dense<float, ROW> > *);
template void FreeDenseFactors<float, COL>
    (PogsData<float, Dense<float, COL> > *);

