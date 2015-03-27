
#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "pogs_c.h"

template <typename T, ORD O>
int Pogs(size_t m, size_t n, T *A,
         T *f_a, T *f_b, T *f_c, T *f_d, T *f_e, FUNCTION *f_h,
         T *g_a, T *g_b, T *g_c, T *g_d, T *g_e, FUNCTION *g_h,
         T rho, T abs_tol, T rel_tol, unsigned int max_iter, int verbose,
         bool adaptive_rho, bool gap_stop, T *x, T *y, T *l, T *optval) {
  // Create pogs struct.
  char ord = O == ROW_MAJ ? 'r' : 'c';
  pogs::MatrixDense<T> A_(ord, m, n, A);
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);

  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;
  
  // Set f and g.
  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(static_cast<Function>(f_h[i]), f_a[i], f_b[i], f_c[i],
        f_d[i], f_e[i]);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(static_cast<Function>(g_h[i]), g_a[i], g_b[i], g_c[i],
         g_d[i], g_e[i]);

  // Set parameters.
  pogs_data.SetRho(rho);
  pogs_data.SetAbsTol(abs_tol);
  pogs_data.SetRelTol(rel_tol);
  pogs_data.SetMaxIter(max_iter);
  pogs_data.SetVerbose(verbose);
  pogs_data.SetAdaptiveRho(adaptive_rho);
  pogs_data.SetGapStop(gap_stop);

  // Solve.
  int err = pogs_data.Solve(f, g);
  *optval = pogs_data.GetOptval();

  memcpy(x, pogs_data.GetX(), n);
  memcpy(y, pogs_data.GetY(), m);
  memcpy(l, pogs_data.GetLambda(), m);

  return err;
}

extern "C" {

int PogsD(enum ORD ord, size_t m, size_t n, double *A,
          double *f_a, double *f_b, double *f_c, double *f_d, double *f_e,
          enum FUNCTION *f_h,
          double *g_a, double *g_b, double *g_c, double *g_d, double *g_e,
          enum FUNCTION *g_h,
          double rho, double abs_tol, double rel_tol, unsigned int max_iter,
          int verbose, int adaptive_rho, int gap_stop,
          double *x, double *y, double *l, double *optval) {
  if (ord == COL_MAJ) {
    return Pogs<double, COL_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho), 
        static_cast<bool>(gap_stop), x, y, l, optval);
  } else {
    return Pogs<double, ROW_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho), 
        static_cast<bool>(gap_stop), x, y, l, optval);
  }
}

int PogsS(enum ORD ord, size_t m, size_t n, float *A,
          float *f_a, float *f_b, float *f_c, float *f_d, float *f_e,
          enum FUNCTION *f_h,
          float *g_a, float *g_b, float *g_c, float *g_d, float *g_e,
          enum FUNCTION *g_h,
          float rho, float abs_tol, float rel_tol, unsigned int max_iter,
          int verbose, int adaptive_rho, int gap_stop,
          float *x, float *y, float *l, float *optval) {
  if (ord == COL_MAJ) {
    return Pogs<float, COL_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho), 
        static_cast<bool>(gap_stop), x, y, l, optval);
  } else {
    return Pogs<float, ROW_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho),
        static_cast<bool>(gap_stop), x, y, l, optval);
  }
}

}

