#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../src/interface_c/pogs_c_api.h"

// Change these two definitions to switch between float and double.
#define POGS PogsD
typedef double real_t;

// Uniform random value in [a, b)
inline real_t runif(real_t a, real_t b) {
  return (b - a) * rand() / RAND_MAX + a;
}

// Max function.
inline real_t max(real_t a, real_t b) {
  return a < b ? b : a;
}

// Lasso
//   minimize (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
int main() {
  // Define input variables.
  size_t m = 100;
  size_t n = 1000;
  real_t *A = (real_t *) malloc(m * n * sizeof(real_t));
  real_t *b = (real_t *) malloc(m * sizeof(real_t));
  real_t *x_true = (real_t *) malloc(n * sizeof(real_t));

  // Define output variables.
  real_t *x = (real_t *) malloc(n * sizeof(real_t));
  real_t *y = (real_t *) malloc(m * sizeof(real_t));
  real_t *mu = (real_t *) malloc(n * sizeof(real_t));
  real_t *nu = (real_t *) malloc(m * sizeof(real_t));
  real_t optval;
  unsigned int final_iter = 0;
  int status;

  // Generate random A.
  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = runif((real_t) -1, (real_t) 1);

  // Generate true x.
  for (unsigned int i = 0; i < n; ++i)
    x_true[i] = rand() < 0.8 * RAND_MAX ? 0 : runif(1.0 / n, 1.0 / n);

  // Compute b = A*x_true + noise.
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * x_true[j];

  for (unsigned int i = 0; i < m; ++i)
    b[i] += runif(0.5, 0.5);

  // Compute lambda_max = ||A'*b||_inf.
  real_t lambda_max = (real_t) 0;
  for (unsigned int j = 0; j < n; ++j) {
    real_t u = (real_t)0;
    for (unsigned int i = 0; i < m; ++i)
      u += A[i + j * m] * b[i];
    lambda_max = max(lambda_max, fabs(u));
  }

  // Define f(y) = (1/2)||y - b||_2^2.
  real_t *f_a = (real_t *) malloc(m * sizeof(real_t));
  real_t *f_b = (real_t *) malloc(m * sizeof(real_t));
  real_t *f_c = (real_t *) malloc(m * sizeof(real_t));
  real_t *f_d = (real_t *) malloc(m * sizeof(real_t));
  real_t *f_e = (real_t *) malloc(m * sizeof(real_t));
  enum FUNCTION *f_h = (enum FUNCTION *) malloc(m * sizeof(enum FUNCTION));
  for (unsigned int i = 0; i < m; ++i) {
    f_a[i] = f_c[i] = (real_t) 1;
    f_b[i] = b[i];
    f_d[i] = f_e[i] = (real_t) 0;
    f_h[i] = SQUARE;
  }

  // Define g(x) = (lambda_max / 2) * ||x||_1.
  real_t *g_a = (real_t *) malloc(n * sizeof(real_t));
  real_t *g_b = (real_t *) malloc(n * sizeof(real_t));
  real_t *g_c = (real_t *) malloc(n * sizeof(real_t));
  real_t *g_d = (real_t *) malloc(n * sizeof(real_t));
  real_t *g_e = (real_t *) malloc(n * sizeof(real_t));
  enum FUNCTION *g_h = (enum FUNCTION *) malloc(n * sizeof(enum FUNCTION));
  for (unsigned int i = 0; i < n; ++i) {
    g_a[i] = (real_t) 0.5 * lambda_max;
    g_b[i] = g_d[i] = g_e[i] = 0;
    g_c[i] = (real_t) 1;
    g_h[i] = ABS;
  }

  // Set up parameters.
  enum ORD ord = ROW_MAJ;
  real_t rho = (real_t) 1;
  real_t abs_tol = (real_t) 1e-4;
  real_t rel_tol = (real_t) 1e-3;
  unsigned int max_iter = 2000u;
  int verbose = 2;
  int adaptive_rho = 1;
  int gap_stop = 0;
  int warm_start = 0;

  SettingsD * settings = calloc(1,sizeof(SettingsD));
  settings->rho=rho;
  settings->abs_tol=abs_tol;
  settings->rel_tol=rel_tol;
  settings->max_iters=max_iter;
  settings->verbose=verbose;
  settings->adaptive_rho=adaptive_rho;
  settings->gap_stop=gap_stop;
  settings->warm_start=warm_start;

  InfoD * info = calloc(1,sizeof(InfoD));
  info->iter=&final_iter;
  info->status=&status;
  info->obj=&optval;
  info->rho=&rho;

  SolutionD * sol = calloc(1,sizeof(SolutionD));
  sol->x=x;
  sol->y=y;
  sol->mu=mu;
  sol->nu=nu;


  // Solve
  void * work = pogs_init_dense_double(ord, m, n, A);
  int err = pogs_solve_double(work, settings, sol, info, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h);
  pogs_finish_double(work);


  printf("Lasso optval = %e, final iter = %u\n", optval, final_iter);

  // Clean up.
  free(A);
  free(b);
  free(x);
  free(y);
  free(mu);
  free(nu);
  free(x_true);

  free(f_a);
  free(f_b);
  free(f_c);
  free(f_d);
  free(f_e);
  free(f_h);

  free(g_a);
  free(g_b);
  free(g_c);
  free(g_d);
  free(g_e);
  free(g_h);

  free(settings);
  free(info);
  free(sol);

  return 0;
}

