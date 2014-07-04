#include <cmath>
#include <algorithm>
#include <vector>

#include "gsl/gsl_blas.hpp"
#include "gsl/gsl_linalg.hpp"
#include "gsl/gsl_matrix.hpp"
#include "gsl/gsl_vector.hpp"

#include "pogs.hpp"
#include "sinkhorn_knopp.hpp"

#ifdef __MEX__
#define printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);
#endif  // __MEX__

// Proximal Operator Graph Solver.
template<typename T, typename M>
void Pogs(PogsData<T, M> *pogs_data) {
  // Extract values from pogs_data
  size_t m = pogs_data->m, n = pogs_data->n, min_dim = std::min(m, n);
  const T kOne = static_cast<T>(1), kZero = static_cast<T>(0);

  // Copy f and g to device
  std::vector<FunctionObj<T> > f = pogs_data->f;
  std::vector<FunctionObj<T> > g = pogs_data->g;

  // Allocate data for ADMM variables.
  gsl::vector<T> d = gsl::vector_calloc<T>(m);
  gsl::vector<T> e = gsl::vector_calloc<T>(n);
  gsl::vector<T> z = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> zt = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> z12 = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> z_prev = gsl::vector_calloc<T>(m + n);
  gsl::matrix<T> L = gsl::matrix_calloc<T>(min_dim, min_dim);
  gsl::matrix<T> AA = gsl::matrix_calloc<T>(min_dim, min_dim);
  gsl::matrix<T> A = gsl::matrix_calloc<T>(m, n);

  // Create views for x and y components.
  gsl::vector<T> x = gsl::vector_subvector(&z, 0, n);
  gsl::vector<T> y = gsl::vector_subvector(&z, n, m);
  gsl::vector<T> xt = gsl::vector_subvector(&zt, 0, n);
  gsl::vector<T> yt = gsl::vector_subvector(&zt, n, m);
  gsl::vector<T> x12 = gsl::vector_subvector(&z12, 0, n);
  gsl::vector<T> y12 = gsl::vector_subvector(&z12, n, m);

  // Equilibrate A.
  const gsl::matrix<T> A_in = gsl::matrix_const_view_array(pogs_data->A, m, n);
  SinkhornKnopp(&A_in, &A, &d, &e);

  // Compute A^TA or AA^T.
  CBLAS_TRANSPOSE_t mult_type = m >= n ? CblasTrans : CblasNoTrans;
  gsl::blas_syrk(CblasLower, mult_type, kOne, &A, kZero, &AA);

  // Scale A.
  gsl::vector<T> diag_AA = gsl::matrix_diagonal(&AA);
  T mean_diag = gsl::blas_asum(&diag_AA) / static_cast<T>(min_dim);
  T sqrt_mean_diag = std::sqrt(mean_diag);
  gsl::matrix_scale(&AA, kOne / mean_diag);
  gsl::matrix_scale(&A, kOne / sqrt_mean_diag);
  gsl::vector_scale(&e, kOne / std::sqrt(sqrt_mean_diag));
  gsl::vector_scale(&d, kOne / std::sqrt(sqrt_mean_diag));

  // Compute cholesky decomposition of (I + A^TA) or (I + AA^T).
  gsl::matrix_memcpy(&L, &AA);
  gsl::vector<T> diag_L = gsl::matrix_diagonal(&L);
  gsl::vector_add_constant(&diag_L, kOne);
  gsl::linalg_cholesky_decomp(&L);

  // Scale f and g to account for diagonal scaling e and d.
  for (unsigned int j = 0; j < m; ++j) {
    f[j].a /= gsl::vector_get(&d, j);
    f[j].d /= gsl::vector_get(&d, j);
  }
  for (unsigned int i = 0; i < n; ++i) {
    g[i].a *= gsl::vector_get(&e, i);
    g[i].d *= gsl::vector_get(&e, i);
  }

  // Signal start of execution.
  if (!pogs_data->quiet)
    printf("%4s %12s %10s %10s %10s %10s %9s\n",
           "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective", "gap");

  T rho = pogs_data->rho;
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * pogs_data->abs_tol;

  for (unsigned int k = 0; k < pogs_data->max_iter; ++k) {
    gsl::vector_sub(&z, &zt);
    ProxEval(g, rho, x.data, x12.data);
    ProxEval(f, rho, y.data, y12.data);

    // Compute Gap.
    gsl::vector_sub(&z, &z12);
    T gap;
    gsl::blas_dot(&z, &z12, &gap);

    // Project and Update Dual Variables.
    gsl::vector_add(&zt, &z12);
    if (m >= n) {
      gsl::vector_memcpy(&x, &xt);
      gsl::blas_gemv(CblasTrans, kOne, &A, &yt, kOne, &x);
      gsl::linalg_cholesky_svx(&L, &x);
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &x, kZero, &y);
      gsl::vector_sub(&yt, &y);
    } else {
      gsl::blas_gemv(CblasNoTrans, kOne, &A, &xt, kZero, &y);
      gsl::blas_symv(CblasLower, kOne, &AA, &yt, kOne, &y);
      gsl::linalg_cholesky_svx(&L, &y);
      gsl::vector_sub(&yt, &y);
      gsl::vector_memcpy(&x, &xt);
      gsl::blas_gemv(CblasTrans, kOne, &A, &yt, kOne, &x);
    }
    gsl::vector_sub(&xt, &x);

    // Compute primal and dual tolerances.
    T nrm_z = gsl::blas_nrm2(&z);
    T nrm_zt = gsl::blas_nrm2(&zt);
    T nrm_z12 = gsl::blas_nrm2(&z12);
    T eps_pri = sqrtn_atol + pogs_data->rel_tol * std::max(nrm_z12, nrm_z);
    T eps_dual = sqrtn_atol + pogs_data->rel_tol * rho * nrm_zt;

    // Compute ||r^k||_2 and ||s^k||_2 (use z_prev for temp storage).
    gsl::vector_sub(&z_prev, &z);
    T nrm_s = rho * gsl::blas_nrm2(&z_prev);
    gsl::vector_memcpy(&z_prev, &z12);
    gsl::vector_sub(&z_prev, &z);
    T nrm_r = gsl::blas_nrm2(&z_prev);

    // Evaluate stopping criteria.
    bool converged = nrm_r <= eps_pri && nrm_s <= eps_dual;
    if (!pogs_data->quiet && (k % 10 == 0 || converged)) {
      T obj = FuncEval(f, y12.data) + FuncEval(g, x12.data);
      printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e %.3e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dual, obj, std::abs(gap));
    }

    if (converged)
      break;

    // Rescale rho.
    // if (nrm_r > 10.0 * nrm_s / rho) {
    //   rho *= 5.0;
    //   gsl::vector_scale(&zt, 0.2);
    // } else if (nrm_r < 0.1 * nrm_s / rho) {
    //   rho *= 0.2;
    //   gsl::vector_scale(&zt, 5.0);
    // }

    // Make copy of z.
    gsl::vector_memcpy(&z_prev, &z);
  }

  // Copy results to output and scale.
  for (unsigned int i = 0; i < m && pogs_data->y != 0; ++i)
    pogs_data->y[i] = gsl::vector_get(&y12, i) / gsl::vector_get(&d, i);
  for (unsigned int j = 0; j < n && pogs_data->x != 0; ++j)
    pogs_data->x[j] = gsl::vector_get(&x12, j) * gsl::vector_get(&e, j);
  pogs_data->optval = FuncEval(f, y12.data) + FuncEval(g, x12.data);

  // Free up memory.
  gsl::matrix_free(&A);
  gsl::matrix_free(&L);
  gsl::matrix_free(&AA);
  gsl::vector_free(&d);
  gsl::vector_free(&e);
  gsl::vector_free(&z);
  gsl::vector_free(&zt);
  gsl::vector_free(&z12);
  gsl::vector_free(&z_prev);
}

template void Pogs<double>(PogsData<double, double*> *);
template void Pogs<float>(PogsData<float, float*> *);

