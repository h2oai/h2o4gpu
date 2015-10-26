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
  gsl::vector<T> z12, zt12;  
  gsl::vector<T> ztemp = gsl::vector_calloc<T>(m + n);
  gsl::matrix<T, kOrd> A, L;
  if (pogs_data->factors.val != 0) {
    compute_factors = pogs_data->factors.val[0] == 0;
    if (!compute_factors)
      rho = pogs_data->factors.val[0];
    de = gsl::vector_view_array(pogs_data->factors.val + 1, m + n);
    z = gsl::vector_view_array(pogs_data->factors.val + 1 + m + n, m + n);
    zt = gsl::vector_view_array(pogs_data->factors.val + 1 + 2 * (m + n), m + n);
    z12 = gsl::vector_view_array(pogs_data->factors.val + 1 + 3 * (m + n), m + n);
    zt12 = gsl::vector_view_array(pogs_data->factors.val + 1 + 4 * (m + n), m + n);
    L = gsl::matrix_view_array<T, kOrd>(
        pogs_data->factors.val + 1 + 5 * (m + n), min_dim, min_dim);
    A = gsl::matrix_view_array<T, kOrd>(
        pogs_data->factors.val + 1 + 5 * (m + n) + min_dim * min_dim, m, n);
  } else {
    de = gsl::vector_calloc<T>(m + n);
    z = gsl::vector_calloc<T>(m + n);
    zt = gsl::vector_calloc<T>(m + n);
    z12 = gsl::vector_calloc<T>(m + n);
    zt12 = gsl::vector_calloc<T>(m + n);
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
  gsl::vector<T> xprev = gsl::vector_subvector(&zprev, 0, n);
  gsl::vector<T> yprev = gsl::vector_subvector(&zprev, n, m);
  gsl::vector<T> xtemp = gsl::vector_subvector(&ztemp, 0, n);
  gsl::vector<T> ytemp = gsl::vector_subvector(&ztemp, n, m);


  if (compute_factors && !err) {
    // Equilibrate A.
    gsl::matrix<T, kOrd> Ain = gsl::matrix_view_array<T, kOrd>(
        pogs_data->A.val, m, n);
    gsl::matrix_memcpy(&A, &Ain);
    err = Equilibrate(&A, &d, &e, true);


    if (!err) {
      // Compute AᵀA or AAᵀ.
      CBLAS_TRANSPOSE_t mult_type = m >= n ? CblasTrans : CblasNoTrans;
      
      // "L":= AᵀA or AAᵀ
      gsl::blas_syrk(CblasLower, mult_type, kOne, &A, kZero, &L);

      // Scale A, L.
      gsl::vector<T> diag_L = gsl::matrix_diagonal(&L);
      T mean_diag = gsl::blas_asum(&diag_L) / static_cast<T>(min_dim);
      T sqrt_mean_diag = std::sqrt(mean_diag);

      gsl::matrix_scale(&L, kOne / mean_diag);
      gsl::matrix_scale(&A, kOne / sqrt_mean_diag);
      T factor = std::sqrt(gsl::blas_nrm2(&d) * std::sqrt(static_cast<T>(n)) /
                          (gsl::blas_nrm2(&e) * std::sqrt(static_cast<T>(m))));
      gsl::blas_scal(kOne / (factor * std::sqrt(sqrt_mean_diag)), &d);
      gsl::blas_scal(factor / (std::sqrt(sqrt_mean_diag)), &e);

      // Compute cholesky decomposition of (I + AᵀA) or (I + AAᵀ).
      // L= chol(I+AᵀA) or chol(I+AAᵀ)
      gsl::vector_add_constant(&diag_L, kOne);
      gsl::linalg_cholesky_decomp(&L);
    }
  }


  // Scale f and g to account for diagonal scaling e and d.
  for (unsigned int i = 0; i < m && !err; ++i) {
    f[i].a /= gsl::vector_get(&d, i);
    f[i].d /= gsl::vector_get(&d, i);
  }
  for (unsigned int j = 0; j < n && !err; ++j) {
    g[j].a *= gsl::vector_get(&e, j);
    g[j].d *= gsl::vector_get(&e, j);
  }

  // Initialize (x,y,\tilde x, \tilde y) from (xₒ, νₒ).

  // Check that guesses for both xₒ and νₒ provided
  if (pogs_data->warm_start && !(pogs_data->x != 0 && pogs_data->nu != 0)) {
    Printf("\nERROR: Must provide x0 and nu0 for warm start\n");
    err=1;
  }
  if (!err && pogs_data->warm_start) {
    //  x:= xₒ, 
    //  y:= Axₒ
    gsl::vector_memcpy(&xprev, pogs_data->x);
    gsl::vector_div(&xprev, &e);
    gsl::blas_gemv(CblasNoTrans, kOne, &A, &xprev, kZero, &yprev);
    gsl::vector_memcpy(&z, &zprev);

    //  ν:= νₒ, 
    //  µ:= -Aᵀνₒ
    //  (\tilde x, \tilde y)= (1/ρ)*(µ,ν)
    gsl::vector_memcpy(&yprev, pogs_data->nu);
    gsl::vector_div(&yprev, &d);
    gsl::blas_gemv(CblasTrans, -kOne, &A, &yprev, kZero, &xprev);
    gsl::blas_scal(-kOne / rho, &zprev);
    gsl::vector_memcpy(&zt, &zprev);
  }
  // else, (x,y, \tilde x, \tilde y) = (0,0,0,0) 

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

  T eps_pri, eps_dua, eps_gap, gap, nrm_r=0, nrm_s=0;


  for (unsigned int k = 0; !err; ++k) {
    //  (x_prev,y_prev)=(x,y)
    gsl::vector_memcpy(&zprev, &z);

    // Evaluate proximal operators.
    // ----------------------------

    gsl::vector_memcpy(&ztemp, &z);
    gsl::blas_axpy(-kOne, &zt, &ztemp);

    //  (x^{1/2},y^{1/2}) = (prox_{g,rho}(x-\tilde x), prox_{f,rho}(y-\tilde y))
    ProxEval(g, rho, xtemp.data, xtemp.stride, x12.data, x12.stride);
    ProxEval(f, rho, ytemp.data, ytemp.stride, y12.data, y12.stride);


    // Apply over-relaxation before projection
    // ---------------------------------------
    gsl::vector_memcpy(&ztemp, &zt);
    gsl::blas_axpy(kAlpha, &z12, &ztemp);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &ztemp);


    // Projection & primal update.
    // ---------------------------
    // The projection step is defined as:
    // (x^{1},y^{1}) = PROJ_{y=Ax} (x^{1/2}+\tilde x, y^{1/2} + \tilde y)
    //               = PROJ_{y=Ax} ("c", "d")
    if (m >= n) {
      //  for  m>=n, projection  becomes the reduced updates:
      //  x^{1} := (I+AᵀA)⁻¹(c + A^Td)
      //  y^{1} := Ax^{1}
      gsl::blas_gemv(CblasTrans, kOne, &A, &ytemp, kOne, &xtemp);
      gsl::linalg_cholesky_svx(&L, &xtemp);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &xtemp, kZero, &y);
      gsl::vector_memcpy(&x, &xtemp);
    } else {
      //  for  m<n, projection  becomes the reduced updates:
      //  y^{1} := d + (I+AAᵀ)⁻¹(Ac - d)
      //  x^{1} := c - Aᵀ(I+AAᵀ)⁻¹(Ac - d)
      gsl::vector_memcpy(&z, &ztemp);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, -kOne, &y);
      gsl::linalg_cholesky_svx(&L, &y);
      gsl::blas_gemv(CblasTrans, -kOne, &A, &y, kOne, &x);
      gsl::blas_axpy(kOne, &ytemp, &y);       
    }


    // Update dual variables (with over-relxation).
    // --------------------------------------------

    // \tilde x^{1/2} := x^{1/2} - x +\tilde x
    // \tilde y^{1/2} := y^{1/2} - y +\tilde y
    gsl::vector_memcpy(&zt12, &z12);
    gsl::blas_axpy(-kOne, &zprev, &zt12);
    gsl::blas_axpy(kOne, &zt, &zt12);

    // \tilde x := \tilde x + \alpha x^{1/2} + (1-\alpha) x - x^{1}
    // \tilde y := \tilde y + \alpha y^{1/2} + (1-\alpha) y - y^{1}
    gsl::blas_axpy(kAlpha, &z12, &zt);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &zt);
    gsl::blas_axpy(-kOne, &z, &zt);



    // Compute gap, objective and tolerances.
    // --------------------------------------
    gsl::blas_dot(&zt12, &z12, &gap);
    gap = rho * std::fabs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    eps_pri = sqrtm_atol + pogs_data->rel_tol * gsl::blas_nrm2(&y12);
    eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * gsl::blas_nrm2(&x12);
    eps_gap = sqrtmn_atol + pogs_data->rel_tol * std::fabs(pogs_data->optval);


    // Calculate residuals.
    // --------------------
    gsl::vector_memcpy(&ztemp, &zprev);
    gsl::blas_axpy(-kOne, &z, &ztemp);
    nrm_s = gsl::blas_nrm2(&ztemp);

    gsl::vector_memcpy(&ztemp, &z12);
    gsl::blas_axpy(-kOne, &z, &ztemp);
    nrm_r = gsl::blas_nrm2(&ztemp);




    // Calculate exact residuals only if necessary.
    // --------------------------------------------
    bool exact = false;
    if (nrm_r < eps_pri && nrm_s < eps_dua) {
      gsl::vector_memcpy(&ztemp, &z12);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &xtemp, -kOne, &ytemp);
      nrm_r = gsl::blas_nrm2(&ytemp);
      if (nrm_r < eps_pri)  {
        gsl::vector_memcpy(&ztemp, &zt12);
        gsl::blas_gemv(CblasTrans, kOne, &A, &ytemp, kOne, &xtemp);
        nrm_s = rho * gsl::blas_nrm2(&xtemp);
        exact = true;
      }
    }

    // Evaluate stopping criteria.
    // ---------------------------
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!pogs_data->gap_stop || gap < eps_gap);
    if (!pogs_data->quiet && (k % 10 == 0 || converged))
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, pogs_data->optval);

    if (converged || k == pogs_data->max_iter){
      if (!converged)
        Printf("Reached max iter=%i\n",pogs_data->max_iter);      
      break;
    }

    // Rescale rho.
    // ------------
    if (pogs_data->adaptive_rho && !converged) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (rho < kRhoMax) {
          rho *= delta;
          // printf("+RHO: %.3e\n", rho);
          gsl::blas_scal(1 / delta, &zt);
          delta = std::min(kGamma * delta, kDeltaMax);
          ku = k;
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (rho > kRhoMin) {
          rho /= delta;
          // printf("-RHO: %.3e\n", rho);
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

  // Scale final iterates, copy to output.


  gsl::vector_memcpy(&ztemp, &z12);
  gsl::vector_div(&ytemp, &d);
  gsl::vector_mul(&xtemp, &e);
  if (pogs_data->y != 0 && !err)
    gsl::vector_memcpy(pogs_data->y, &ytemp);
  if (pogs_data->x != 0 && !err)
    gsl::vector_memcpy(pogs_data->x, &xtemp);


  gsl::vector_memcpy(&ztemp, &zt12);
  gsl::blas_scal(-rho, &ztemp);
  gsl::vector_mul(&ytemp, &d); 
  gsl::vector_div(&xtemp, &e); 
  if (pogs_data->nu != 0 && !err)
    gsl::vector_memcpy(pogs_data->nu, &ytemp);
  if (pogs_data->mu != 0 && !err)
    gsl::vector_memcpy(pogs_data->mu, &xtemp);


  // Store rho and free memory.
  if (pogs_data->factors.val != 0 && !err) {
    pogs_data->factors.val[0] = rho;
    pogs_data->rho=rho;
    gsl::vector_memcpy(&z, &z12);
  } else {
    gsl::vector_free(&de);
    gsl::vector_free(&z);
    gsl::vector_free(&zt);
    gsl::vector_free(&z12);
    gsl::vector_free(&zt12);
    gsl::matrix_free(&L);
    gsl::matrix_free(&A);
  }
  gsl::vector_free(&zprev);
  gsl::vector_free(&ztemp);

  return err;
}

template <typename T, POGS_ORD O>
int AllocDenseFactors(PogsData<T, Dense<T, O> > *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n;
  size_t flen = 1 + 5 * (m + n) + std::min(m, n) * std::min(m, n) + m * n;
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

