#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <algorithm>
#include <vector>

#include "pogs.hpp"
#include "sinkhorn_knopp.hpp"

#ifdef __MEX__
#define printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);
#endif  // __MEX__

// Proximal Operator Graph Solver.
template<>
void Pogs(PogsData<double, double*> *pogs_data) {
  // Extract values from pogs_data
  size_t n = pogs_data->n;
  size_t m = pogs_data->m;
  size_t min_dim = std::min(m, n);
  std::vector<FunctionObj<double>> f = pogs_data->f;
  std::vector<FunctionObj<double>> g = pogs_data->g;

  // Allocate data for ADMM variables.
  gsl_vector *d = gsl_vector_calloc(m);
  gsl_vector *e = gsl_vector_calloc(n);
  gsl_vector *z = gsl_vector_calloc(m + n);
  gsl_vector *zt = gsl_vector_calloc(m + n);
  gsl_vector *z12 = gsl_vector_calloc(m + n);
  gsl_vector *z_prev = gsl_vector_calloc(m + n);
  gsl_matrix *A = gsl_matrix_calloc(m, n);
  gsl_matrix *AA = gsl_matrix_calloc(min_dim, min_dim);
  gsl_matrix *L = gsl_matrix_calloc(min_dim, min_dim);

  // Create views for x and y components.
  gsl_vector_view x = gsl_vector_subvector(z, 0, n);
  gsl_vector_view y = gsl_vector_subvector(z, n, m);
  gsl_vector_view xt = gsl_vector_subvector(zt, 0, n);
  gsl_vector_view yt = gsl_vector_subvector(zt, n, m);
  gsl_vector_view x12 = gsl_vector_subvector(z12, 0, n);
  gsl_vector_view y12 = gsl_vector_subvector(z12, n, m);

  // Equilibrate A.
  gsl_matrix_const_view A_in = gsl_matrix_const_view_array(pogs_data->A, m, n);
  SinkhornKnopp(&A_in.matrix, A, d, e);

  // Compute A^TA or AA^T.
  CBLAS_TRANSPOSE_t mult_type = m >= n ? CblasTrans : CblasNoTrans;
  gsl_blas_dsyrk(CblasLower, mult_type, 1.0, A, 0.0, AA);

  // Scale A.
  gsl_vector_view diag_AA = gsl_matrix_diagonal(AA);
  double mean_diag = gsl_blas_dasum(&diag_AA.vector) /
      static_cast<double>(min_dim);
  double sqrt_mean_diag = sqrt(mean_diag);
  gsl_matrix_scale(AA, 1.0 / mean_diag);
  gsl_matrix_scale(A, 1.0 / sqrt_mean_diag);
  gsl_vector_scale(e, 1.0 / sqrt(sqrt_mean_diag));
  gsl_vector_scale(d, 1.0 / sqrt(sqrt_mean_diag));

  // Compute cholesky decomposition of (I + A^TA) or (I + AA^T).
  gsl_matrix_memcpy(L, AA);
  gsl_vector_view diag_L = gsl_matrix_diagonal(L);
  gsl_vector_add_constant(&diag_L.vector, 1.0);
  gsl_linalg_cholesky_decomp(L);

  // Scale f and g to account for diagonal scaling e and d.
  for (unsigned int j = 0; j < m; ++j) {
    f[j].a /= gsl_vector_get(d, j);
    f[j].d /= gsl_vector_get(d, j);
  }
  for (unsigned int i = 0; i < n; ++i) {
    g[i].a *= gsl_vector_get(e, i);
    g[i].d *= gsl_vector_get(e, i);
  }

  // Signal start of execution.
  if (!pogs_data->quiet)
    printf("%4s %12s %10s %10s %10s %10s\n",
           "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

  double rho = pogs_data->rho;
  double sqrtn_atol = sqrt(static_cast<double>(n)) * pogs_data->abs_tol;

  for (unsigned int k = 0; k < pogs_data->max_iter; ++k) {
    gsl_vector_sub(z, zt);
    ProxEval(g, rho, x.vector.data, x12.vector.data);
    ProxEval(f, rho, y.vector.data, y12.vector.data);

    // Project and Update Dual Variables.
    gsl_vector_add(zt, z12);
    if (m >= n) {
      gsl_vector_memcpy(&x.vector, &xt.vector);
      gsl_blas_dgemv(CblasTrans, 1.0, A, &yt.vector, 1.0, &x.vector);
      gsl_linalg_cholesky_svx(L, &x.vector);
      gsl_blas_dgemv(CblasNoTrans, 1.0, A, &x.vector, 0.0, &y.vector);
      gsl_vector_sub(&yt.vector, &y.vector);
    } else {
      gsl_blas_dgemv(CblasNoTrans, 1.0, A, &xt.vector, 0.0, &y.vector);
      gsl_blas_dsymv(CblasLower, 1.0, AA, &yt.vector, 1.0, &y.vector);
      gsl_linalg_cholesky_svx(L, &y.vector);
      gsl_vector_sub(&yt.vector, &y.vector);
      gsl_vector_memcpy(&x.vector, &xt.vector);
      gsl_blas_dgemv(CblasTrans, 1.0, A, &yt.vector, 1.0, &x.vector);
    }
    gsl_vector_sub(&xt.vector, &x.vector);

    // Compute primal and dual tolerances.
    double nrm_z = gsl_blas_dnrm2(z);
    double nrm_zt = gsl_blas_dnrm2(zt);
    double nrm_z12 = gsl_blas_dnrm2(z12);
    double eps_pri = sqrtn_atol + pogs_data->rel_tol * std::max(nrm_z12, nrm_z);
    double eps_dual = sqrtn_atol + pogs_data->rel_tol * rho * nrm_zt;

    // Compute ||r^k||_2 and ||s^k||_2.
    gsl_vector_sub(z12, z);
    gsl_vector_sub(z_prev, z);
    double nrm_r = gsl_blas_dnrm2(z12);
    double nrm_s = rho * gsl_blas_dnrm2(z_prev);

    // Evaluate stopping criteria.
    bool converged = nrm_r <= eps_pri && nrm_s <= eps_dual;
    if (!pogs_data->quiet && (k % 10 == 0 || converged)) {
      double obj = FuncEval(f, y.vector.data) + FuncEval(g, x.vector.data);
      printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dual, obj);
    }

   if (converged)
     break;

    // Rescale rho.
    if (nrm_r > 10.0 * nrm_s / rho) {
      rho *= 5.0;
      gsl_vector_scale(zt, 0.2);
    } else if (nrm_r < 0.1 * nrm_s / rho) {
      rho *= 0.2;
      gsl_vector_scale(zt, 5.0);
    }

    // Make copy of z.
    gsl_vector_memcpy(z_prev, z);
  }

  // Copy results to output and scale.
  for (unsigned int i = 0; i < m && pogs_data->y != 0; ++i)
    pogs_data->y[i] = gsl_vector_get(&y.vector, i) / gsl_vector_get(d, i);
  for (unsigned int j = 0; j < n && pogs_data->x != 0; ++j)
    pogs_data->x[j] = gsl_vector_get(&x.vector, j) * gsl_vector_get(e, j);
  pogs_data->optval = FuncEval(f, y.vector.data) + FuncEval(g, x.vector.data);

  // Free up memory.
  gsl_matrix_free(A);
  gsl_matrix_free(L);
  gsl_matrix_free(AA);
  gsl_vector_free(d);
  gsl_vector_free(e);
  gsl_vector_free(z);
  gsl_vector_free(zt);
  gsl_vector_free(z12);
  gsl_vector_free(z_prev);
}

