#include <cmath>
#include <algorithm>
#include <vector>

#include "_interface_defs.h"
#include "gsl/cblas.h"
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
  const T kRhoMax = static_cast<T>(1e4);
  const T kRhoMin = static_cast<T>(1e-4);
  const CBLAS_ORDER kOrd = M::Ord == ROW ? CblasRowMajor : CblasColMajor;

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
  gsl::matrix<T, kOrd> A, L;
  if (pogs_data->factors.val != 0) {
    compute_factors = pogs_data->factors.val[0] == 0;
    if (!compute_factors)
      rho = pogs_data->factors.val[0];
    de = gsl::vector_view_array(pogs_data->factors.val + 1, m + n);
    z = gsl::vector_view_array(pogs_data->factors.val + 1 + m + n, m + n);
    zt = gsl::vector_view_array(pogs_data->factors.val + 1 + 2 * (m + n),
        m + n);
    L = gsl::matrix_view_array<T, kOrd>(
        pogs_data->factors.val + 1 + 3 * (m + n), min_dim, min_dim);
    A = gsl::matrix_view_array<T, kOrd>(
        pogs_data->factors.val + 1 + 3 * (m + n) + min_dim * min_dim, m, n);
  } else {
    de = gsl::vector_calloc<T>(m + n);
    z = gsl::vector_calloc<T>(m + n);
    zt = gsl::vector_calloc<T>(m + n);
    L = gsl::matrix_calloc<T, kOrd>(min_dim, min_dim);
    A = gsl::matrix_calloc<T, kOrd>(m, n);
  }
  if (de.data == 0 || z.data == 0 || zt.data == 0 || zprev.data == 0 ||
      z12.data == 0 || A.data == 0 || L.data == 0)
    err = 1;

  // Create views for x and y components.
  gsl::vector<T> d = gsl::vector_subvector(&de, 0, m);
  gsl::vector<T> e = gsl::vector_subvector(&de, m, n);
  gsl::vector<T> x = gsl::vector_subvector(&z, 0, n);
  gsl::vector<T> y = gsl::vector_subvector(&z, n, m);
  gsl::vector<T> x12 = gsl::vector_subvector(&z12, 0, n);
  gsl::vector<T> y12 = gsl::vector_subvector(&z12, n, m);
  
  if (compute_factors && !err) {
    // Equilibrate A.
    gsl::matrix<T, kOrd> Ain = gsl::matrix_view_array<T, kOrd>(
        pogs_data->A.val, m, n);
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

      // Initialize x and y from x0 or y0
      if (pogs_data->init_x && !pogs_data->init_y && pogs_data->x) {
        gsl::vector_memcpy(&x, pogs_data->x);
        gsl::vector_div(&x, &e);
        gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, kZero, &y);
      } else if (pogs_data->init_y && !pogs_data->init_x && pogs_data->y) {
        gsl::vector_memcpy(&y, pogs_data->y);
        gsl::vector_mul(&y, &d);
        gsl::matrix<T, kOrd> AA = gsl::matrix_alloc<T, kOrd>(min_dim, min_dim);
        gsl::matrix_memcpy(&AA, &L);
        gsl::vector<T> diag_AA = gsl::matrix_diagonal(&AA);
        gsl::vector_add_constant(&diag_AA, static_cast<T>(1e-4));
        gsl::linalg_cholesky_decomp(&AA);
        if (m >= n) {
          gsl::blas_gemv(CblasTrans, kOne, &A, &y, kZero, &x);
          gsl::linalg_cholesky_svx(&AA, &x);
        } else {
          gsl::linalg_cholesky_svx(&AA, &y);
          gsl::blas_gemv(CblasTrans, kOne, &A, &y, kZero, &x);
        }
        gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, kZero, &y);
        gsl::matrix_free(&AA);
      }

      // Compute cholesky decomposition of (I + A^TA) or (I + AA^T).
      gsl::vector_add_constant(&diag_L, kOne);
      gsl::linalg_cholesky_decomp(&L);

      // TODO: Issue warning if x == NULL or y == NULL
      // Initialize x and y from guess x0 and y0
      if (pogs_data->init_x && pogs_data->init_y &&
          pogs_data->x && pogs_data->y) {
        gsl::vector_memcpy(&x, pogs_data->x);
        gsl::vector_memcpy(&y, pogs_data->y);
        gsl::vector_div(&x, &e);
        gsl::vector_mul(&y, &d);
        if (m >= n) {
          gsl::blas_gemv(CblasTrans, kOne, &A, &y, kOne, &x);
          gsl::linalg_cholesky_svx(&L, &x);
        } else {
          gsl::blas_gemv(CblasNoTrans, -kOne, &A, &x, kOne, &y);
          gsl::linalg_cholesky_svx(&L, &y);
          gsl::blas_gemv(CblasTrans, kOne, &A, &y, kOne, &x);
        }
        gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, kZero, &y);
      }
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
  bool converged = false;

  for (unsigned int k = 0; !err; ++k) {
    gsl::vector_memcpy(&zprev, &z);

    // Evaluate proximal operators.
    gsl::blas_axpy(-kOne, &zt, &z);
    ProxEval(g, rho, x.data, x.stride, x12.data, x12.stride);
    ProxEval(f, rho, y.data, y.stride, y12.data, y12.stride);

    // Compute gap, objective and tolerances.
    T gap, nrm_r = 0, nrm_s = 0;
    gsl::blas_axpy(-kOne, &z12, &z);
    gsl::blas_dot(&z, &z12, &gap);
    gap = rho * std::fabs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * gsl::blas_nrm2(&z12);
    T eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * gsl::blas_nrm2(&z);
    T eps_gap = sqrtmn_atol + pogs_data->rel_tol * std::fabs(pogs_data->optval);

    if (converged && k < pogs_data->max_iter)
      break;

    // Project and Update Dual Variables.
    if (m >= n) {
      gsl::blas_gemv(CblasTrans, -kOne, &A, &y, -kOne, &x);
      nrm_s = rho * gsl::blas_nrm2(&x);
      gsl::linalg_cholesky_svx(&L, &x);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, kZero, &y);
      gsl::blas_axpy(kOne, &zprev, &z);
    } else {
      gsl::vector_memcpy(&z, &z12);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, -kOne, &y);
      nrm_r = gsl::blas_nrm2(&y);
      gsl::linalg_cholesky_svx(&L, &y);
      gsl::blas_gemv(CblasTrans, -kOne, &A, &y, kZero, &x);
      gsl::blas_axpy(kOne, &z12, &z);
    }

    // Apply over relaxation.
    gsl::blas_scal(kAlpha, &z);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &z);

    // Update dual variable.
    gsl::blas_axpy(kAlpha, &z12, &zt);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &zt);
    gsl::blas_axpy(-kOne, &z, &zt);

    bool exact = false;
    if (m >= n) {
      gsl::vector_memcpy(&zprev, &z12);
      gsl::blas_axpy(-kOne, &z, &zprev);
      nrm_r = gsl::blas_nrm2(&zprev);
      if (nrm_s < eps_dua && nrm_r < eps_pri) {
        gsl::blas_gemv(CblasNoTrans, kOne, &A, &x12, -kOne, &y12);
        nrm_r = gsl::blas_nrm2(&y12);
        exact = true;
      }
    } else {
      gsl::blas_axpy(-kOne, &zprev, &z12);
      gsl::blas_axpy(-kOne, &z, &zprev);
      nrm_s = rho * gsl::blas_nrm2(&zprev);
      if (nrm_r < eps_pri && nrm_s < eps_dua) {
        gsl::blas_gemv(CblasTrans, kOne, &A, &y12, kOne, &x12);
        nrm_s = rho * gsl::blas_nrm2(&x12);
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
        if (rho < kRhoMax) {
          rho *= delta;
          gsl::blas_scal(1 / delta, &zt);
          delta = std::min(kGamma * delta, kDeltaMax);
          ku = k;
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (rho > kRhoMin) {
          rho /= delta;
          gsl::blas_scal(1 / delta, &zt);
          gsl::blas_scal(delta, &zt);
          delta = std::min(kGamma * delta, kDeltaMax);
          kd = k;
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = std::max(delta / kGamma, kDeltaMin);
      }
    }
  }
  // Scale x, y and l for output.
  gsl::vector_div(&y12, &d);
  gsl::vector_mul(&x12, &e);
  gsl::vector_mul(&y, &d);
  gsl::blas_scal(rho, &y);

  // Copy results to output.
  if (pogs_data->y != 0 && !err)
    gsl::vector_memcpy(pogs_data->y, &y12);
  if (pogs_data->x != 0 && !err)
    gsl::vector_memcpy(pogs_data->x, &x12);
  if (pogs_data->l != 0 && !err)
    gsl::vector_memcpy(pogs_data->l, &y);

  // Store rho and free memory.
  if (pogs_data->factors.val != 0 && !err) {
    pogs_data->factors.val[0] = rho;
    gsl::vector_memcpy(&z, &zprev);
  } else {
    gsl::vector_free(&de);
    gsl::vector_free(&z);
    gsl::vector_free(&zt);
    gsl::matrix_free(&L);
    gsl::matrix_free(&A);
  }
  gsl::vector_free(&z12);
  gsl::vector_free(&zprev);
  return err;
}

template <typename T, POGS_ORD O>
int AllocDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n;
  size_t flen = 1 + 3 * (m + n) + std::min(m, n) * std::min(m, n) + m * n;
  pogs_data->factors.val = new T[flen]();
  if (pogs_data->factors.val != 0)
    return 0;
  else
    return 1;
}

template <typename T, POGS_ORD O>
void FreeDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  delete [] pogs_data->factors.val;
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

