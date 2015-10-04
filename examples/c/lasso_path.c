#include <math.h>
#include <stdio.h>
#include <stdlib.h> 

#include "pogs_c.h"

// change these definitions to switch between single- and double-precision
typedef double real_t;        
#define POGS_SETTINGS PogsSettingsD       // PogsSettingsD or PogsSettingsS
#define POGS_SOLUTION PogsSolutionD       // PogsSolutionD or PogsSolutionS
#define POGS_INFO PogsInfoD               // PogsInfoD or PogsInfoS
#define POGS_INIT pogs_init_dense_double  // pogs_init_dense_double or pogs_init_dense_single
#define POGS_SOLVE pogs_solve_double      // pogs_solve_double or pogs_solve_single
#define POGS_FINISH pogs_finish_double    // pogs_finish_double or pogs_finish_single


// Uniform random value in [a, b)
inline real_t runif(real_t a, real_t b) {
  return (b - a) * rand() / RAND_MAX + a;
}

// Max function.
inline real_t max(real_t a, real_t b) {
  return a < b ? b : a;
}

real_t maxdiff(real_t *a, real_t *b, unsigned int n){
  real_t diff = (real_t) 0;
  for (unsigned int i = 0; i < n; ++i)
    diff = max(diff, fabs(a[i]-b[i]) );
  return diff;
}

real_t asum(real_t *a, unsigned int n){
  real_t sum= (real_t) 0;
  for (unsigned int i = 0; i < n; ++i)
    sum += fabs(a[i]);
  return sum;
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
  real_t *x_last = (real_t *) malloc(n * sizeof(real_t));

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

  // Setup parameters.
  enum ORD ord = ROW_MAJ;
  POGS_SETTINGS *settings = &(POGS_SETTINGS){
    .rho = (real_t) 1,
    .abs_tol = (real_t) 1e-4,
    .rel_tol = (real_t) 1e-3,
    .max_iters = 2000u,
    .verbose = 1u,
    .adaptive_rho = 1,
    .gap_stop = 0,
    .warm_start = 0
  };

  // Output variables.
  POGS_SOLUTION *solution = &(POGS_SOLUTION){
    .x = (real_t *) malloc(n * sizeof(real_t)),
    .y = (real_t *) malloc(m * sizeof(real_t)),
    .nu = (real_t *) malloc(m * sizeof(real_t)),
    .mu = (real_t *) malloc(n * sizeof(real_t)),
    .x12 = (real_t *) malloc(n * sizeof(real_t)),
    .y12 = (real_t *) malloc(m * sizeof(real_t)),
    .nu12 = (real_t *) malloc(m * sizeof(real_t)),
    .mu12 = (real_t *) malloc(n * sizeof(real_t))
  };
  
  POGS_INFO *info = &(POGS_INFO){
    .iter=0,
    .status=0,
    .obj=(real_t) 0,
    .rho=(real_t) 0,
    .solvetime=(real_t) 0
  };


  unsigned int nlambda = 100;


  // Solve
  void * p_work = POGS_INIT(ord, m, n, A);
  

  for (unsigned int i = 0; i < nlambda; ++i) {
    real_t lambda = (real_t) exp((log(lambda_max) * (nlambda - 1 - i) +
        (real_t) 1e-2 * log(lambda_max) * i) / (nlambda - 1));

    for (unsigned int i = 0; i < n; ++i)
      g_c[i] = lambda;

    info->status = POGS_SOLVE(p_work, settings, solution, info, \
      f_a, f_b, f_c, f_d, f_e, f_h, \
      g_a, g_b, g_c, g_d, g_e, g_h);


    if ( maxdiff(solution->x, x_last, n) < ((real_t) 1e-3) * asum(solution->x, n))
      break;

    for (int i = 0; i < n; ++i)
      x_last[i] = solution->x[i];
  }


  POGS_FINISH(p_work);

  
  printf("Lasso path optval = %e\n", info->obj);

  // Clean up.
  free(A);
  free(b);
  free(solution->x);
  free(solution->y);
  free(solution->nu);
  free(solution->mu);  
  free(solution->x12);
  free(solution->y12);
  free(solution->nu12);
  free(solution->mu12);  
  free(x_true);
  free(x_last);

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

  return 0;
}

