#include <cmath>
#include <algorithm>
#include <vector>

#include "_interface_defs.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "pogs.h"
#include "sinkhorn_knopp.h"

// Proximal Operator Graph Solver.
template<typename T, typename M>
int Pogs(PogsData<T, M> *pogs_data) {
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kDeltaMax = static_cast<T>(2);
  const T kGamma = static_cast<T>(1.01);
  const T kTau = static_cast<T>(0.8);
  const T kAlpha = static_cast<T>(1.7);
  const T kKappa = static_cast<T>(0.9);

  int err = 0;

  // Extract values from pogs_data.
  size_t m = pogs_data->m, n = pogs_data->n, min_dim = std::min(m, n);
  T rho = pogs_data->rho;
  const T kOne = static_cast<T>(1), kZero = static_cast<T>(0);
  std::vector<FunctionObj<T> > f = pogs_data->f;
  std::vector<FunctionObj<T> > g = pogs_data->g;

  // Allocate data for ADMM variables.
  bool compute_factors = true;
  gsl::vector<T> de, z, zt;
  gsl::vector<T> zprev = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> z12 = gsl::vector_calloc<T>(m + n);
  gsl::matrix<T> A, L;
  gsl::matrix<T> C = gsl::matrix_calloc<T>(2, m + n);
  if (pogs_data->factors != 0) {
    if (pogs_data->factors[0] > 0) {
      rho = pogs_data->factors[0];
      compute_factors = false;
    }
    de = gsl::vector_view_array(pogs_data->factors + 1, m + n);
    z = gsl::vector_view_array(pogs_data->factors + 1 + m + n, m + n);
    zt = gsl::vector_view_array(pogs_data->factors + 1 + 2 * (m + n), m + n);
    L = gsl::matrix_view_array(pogs_data->factors + 1 + 3 * (m + n), min_dim,
                               min_dim);
    A = gsl::matrix_view_array(pogs_data->factors + 1 + 3 * (m + n) +
                               min_dim * min_dim, m, n);
  } else {
    de = gsl::vector_calloc<T>(m + n);
    z = gsl::vector_calloc<T>(m + n);
    zt = gsl::vector_calloc<T>(m + n);
    L = gsl::matrix_calloc<T>(min_dim, min_dim);
    A = gsl::matrix_calloc<T>(m, n);
  }
  if (de.data == 0 || z.data == 0 || zt.data == 0 || zprev.data == 0 ||
      z12.data == 0 || A.data == 0 || L.data == 0 || C.data == 0)
    err = 1;

  // Create views for x and y components.
  gsl::matrix<T> Cx = gsl::matrix_submatrix(&C, 0, 0, 2, n);
  gsl::matrix<T> Cy = gsl::matrix_submatrix(&C, 0, n, 2, m);
  gsl::vector<T> d = gsl::vector_subvector(&de, 0, m);
  gsl::vector<T> e = gsl::vector_subvector(&de, m, n);
  gsl::vector<T> x = gsl::vector_subvector(&z, 0, n);
  gsl::vector<T> y = gsl::vector_subvector(&z, n, m);
  gsl::vector<T> x12 = gsl::vector_subvector(&z12, 0, n);
  gsl::vector<T> y12 = gsl::vector_subvector(&z12, n, m);
  gsl::vector<T> cz0 = gsl::matrix_row(&C, 0);
  gsl::vector<T> cx0 = gsl::vector_subvector(&cz0, 0, n);
  gsl::vector<T> cy0 = gsl::vector_subvector(&cz0, n, m);
  gsl::vector<T> cz1 = gsl::matrix_row(&C, 1);
  gsl::vector<T> cx1 = gsl::vector_subvector(&cz1, 0, n);
  gsl::vector<T> cy1 = gsl::vector_subvector(&cz1, n, m);
  
  if (compute_factors && !err) {
    // Equilibrate A.
    gsl::matrix<T> Ain = gsl::matrix_const_view_array(pogs_data->A, m, n);
    gsl::matrix_memcpy(&A, &Ain);
    err = Equilibrate(&A, &d, &e, true);
    if (!err) {
      // Compute A^TA or AA^T.
      CBLAS_TRANSPOSE_t mult_type = m >= n ? CblasTrans : CblasNoTrans;
      gsl::blas_syrk(CblasLower, mult_type, kOne, &A, kZero, &L);

      // Scale A.
      gsl::vector<T> diag_L = gsl::matrix_diagonal(&L);
      T mean_diag = gsl::blas_asum(&diag_L) / static_cast<T>(min_dim);
      T sqrt_mean_diag = std::sqrt(mean_diag);
      gsl::matrix_scale(&L, kOne / mean_diag);
      gsl::matrix_scale(&A, kOne / sqrt_mean_diag);
      T factor = std::sqrt(gsl::blas_nrm2(&d) * std::sqrt(static_cast<T>(n)) /
                          (gsl::blas_nrm2(&e) * std::sqrt(static_cast<T>(m))));
      gsl::blas_scal(kOne / (factor * std::sqrt(sqrt_mean_diag)), &d);
      gsl::blas_scal(factor / (std::sqrt(sqrt_mean_diag)), &e);

      // Compute cholesky decomposition of (I + A^TA) or (I + AA^T).
      gsl::vector_add_constant(&diag_L, kOne);
      gsl::linalg_cholesky_decomp(&L);
    }
  }

  // Scale f and g to account++ for diagonal scaling e and d.
  for (unsigned int i = 0; i < m && !err; ++i) {
    f[i].a /= gsl::vector_get(&d, i);
    f[i].d /= gsl::vector_get(&d, i);
  }
  for (unsigned int j = 0; j < n && !err; ++j) {
    g[j].a *= gsl::vector_get(&e, j);
    g[j].d *= gsl::vector_get(&e, j);
  }

  // Signal start of execution.
  if (!pogs_data->quiet)
    Printf("   #      res_pri    eps_pri   res_dual   eps_dual"
           "        gap    eps_gap  objective\n");

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * pogs_data->abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * pogs_data->abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * pogs_data->abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0, ku = 0;

  for (unsigned int k = 0; k < pogs_data->max_iter && !err; ++k) {
    gsl::vector_memcpy(&zprev, &z);

    // Evaluate proximal operators.
    gsl::vector_memcpy(&cz0, &z);
    gsl::blas_axpy(-kOne, &zt, &cz0);
    ProxEval(g, rho, cx0.data, x12.data);
    ProxEval(f, rho, cy0.data, y12.data);

    // Compute gap, objective and tolerances.
    T gap, nrm_r, nrm_s;
    gsl::blas_axpy(-kOne, &z12, &cz0);
    gsl::blas_dot(&cz0, &z12, &gap);
    gap = std::abs(gap * rho);
    T obj = FuncEval(f, y12.data) + FuncEval(g, x12.data);
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * gsl::blas_nrm2(&z12);
    T eps_dual = sqrtn_atol + pogs_data->rel_tol * rho * gsl::blas_nrm2(&cz0);
    T eps_gap = sqrtmn_atol + pogs_data->rel_tol * std::abs(obj);

    // Store dual variable
    if (pogs_data->l != 0)
      gsl::vector_memcpy(pogs_data->l, &cy0);

    // Project and Update Dual Variables.
    if (m >= n) {
      gsl::blas_gemv(CblasTrans, kOne, &A, &cy0, kOne, &cx0);
      nrm_s = rho * gsl::blas_nrm2(&cx0);
      gsl::linalg_cholesky_svx(&L, &cx0);
      gsl::vector_memcpy(&cy0, &y);
      gsl::vector_memcpy(&cz1, &z12);
      gsl::blas_gemm(CblasNoTrans, CblasTrans, -kOne, &Cx, &A, kOne, &Cy);
      nrm_r = gsl::blas_nrm2(&cy1);
      gsl::vector_memcpy(&y, &cy0);
      gsl::blas_axpy(-kOne, &cx0, &x);
    } else {
      gsl::vector_memcpy(&z, &z12);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, -kOne, &y);
      nrm_r = gsl::blas_nrm2(&y);
      gsl::linalg_cholesky_svx(&L, &y);
      gsl::vector_memcpy(&cy1, &y);
      gsl::vector_memcpy(&cx1, &x12);
      gsl::blas_scal(-kOne, &cy0);
      gsl::blas_gemm(CblasNoTrans, CblasNoTrans, -kOne, &Cy, &A, kOne, &Cx);
      nrm_s = rho * gsl::blas_nrm2(&cx0);
      gsl::vector_memcpy(&x, &cx1);
      gsl::blas_axpy(kOne, &y12, &y);
    }

    // Apply over relaxation.
    gsl::blas_scal(kAlpha, &z);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &z);

    // Update dual variable.
    gsl::blas_axpy(kAlpha, &z12, &zt);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &zt);
    gsl::blas_axpy(-kOne, &z, &zt);

    // Evaluate stopping criteria.
    bool converged = nrm_r < eps_pri && nrm_s < eps_dual && gap < eps_gap;
    if (!pogs_data->quiet && (k % 10 == 0 || converged))
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dual, gap, eps_gap, obj);
    if (converged)
      break;

    // Rescale rho.
    if (pogs_data->adaptive_rho) {
      if (nrm_s < xi * eps_dual && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        rho *= delta;
        gsl::blas_scal(1 / delta, &zt);
        delta = std::min(kGamma * delta, kDeltaMax);
        ku = k;
      } else if (nrm_s > xi * eps_dual && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        rho /= delta;
        gsl::blas_scal(delta, &zt);
        delta = std::min(kGamma * delta, kDeltaMax);
        kd = k;
      } else if (nrm_s < xi * eps_dual && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = std::max(delta / kGamma, kDeltaMin);
      }
    }
  }

  pogs_data->optval = FuncEval(f, y12.data) + FuncEval(g, x12.data);

  // Copy results to output and scale.
  for (unsigned int i = 0; i < m && pogs_data->y != 0 && !err; ++i)
    pogs_data->y[i] = gsl::vector_get(&y12, i) / gsl::vector_get(&d, i);
  for (unsigned int j = 0; j < n && pogs_data->x != 0 && !err; ++j)
    pogs_data->x[j] = gsl::vector_get(&x12, j) * gsl::vector_get(&e, j);
  for (unsigned int i = 0; i < m && pogs_data->l != 0 && !err; ++i)
    pogs_data->l[i] = rho * pogs_data->l[i] * gsl::vector_get(&d, i);

  // Store rho and free memory.
  if (pogs_data->factors != 0 && !err) {
    pogs_data->factors[0] = rho;
  } else {
    gsl::vector_free(&de);
    gsl::vector_free(&z);
    gsl::vector_free(&zt);
    gsl::matrix_free(&L);
    gsl::matrix_free(&A);
  }
  gsl::matrix_free(&C);
  gsl::vector_free(&z12);
  gsl::vector_free(&zprev);
  return err;
}

template <>
int AllocFactors(PogsData<double, double*> *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n;
  size_t flen = 1 + 3 * (m + n) + std::min(m, n) * std::min(m, n) + m * n;
  pogs_data->factors = new double[flen]();
  if (pogs_data->factors != 0)
    return 0;
  else
    return 1;
}

template <>
int AllocFactors(PogsData<float, float*> *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n;
  size_t flen = 1 + 3 * (m + n) + std::min(m, n) * std::min(m, n) + m * n;
  pogs_data->factors = new float[flen]();
  if (pogs_data->factors != 0)
    return 0;
  else
    return 1;
}

template <>
void FreeFactors(PogsData<double, double*> *pogs_data) {
  delete [] pogs_data->factors;
}

template <>
void FreeFactors(PogsData<float, float*> *pogs_data) {
  delete [] pogs_data->factors;
}

template int Pogs<double>(PogsData<double, double*> *);
template int Pogs<float>(PogsData<float, float*> *);

