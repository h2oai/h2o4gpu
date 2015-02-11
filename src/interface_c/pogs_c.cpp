#include "pogs.h"
#include "pogs_c.h"

template <typename T, ORD O>
int Pogs(size_t m, size_t n, const T *A,
         const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e,
         const FUNCTION *f_h,
         const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e,
         const FUNCTION *g_h,
         T rho, T abs_tol, T rel_tol, unsigned int max_iter, bool quiet,
         bool adaptive_rho, bool gap_stop, T *x, T *y, T *l, T *optval) {
  // Create pogs struct.
  // TODO: Get rid of const cast with new interface.
  const Dense<T, static_cast<POGS_ORD>(O)> A_(const_cast<T*>(A));
  PogsData<T, Dense<T, static_cast<POGS_ORD>(O)> > pogs_data(A_, m, n);
  pogs_data.x = x;
  pogs_data.y = y;
  
  // Set f and g.
  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(static_cast<Function>(f_h[i]),
        f_a[i], f_b[i], f_c[i], f_d[i], f_e[i]);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(static_cast<Function>(g_h[i]),
        g_a[i], g_b[i], g_c[i], g_d[i], g_e[i]);

  // Set parameters.
  pogs_data.rho = rho;
  pogs_data.abs_tol = abs_tol;
  pogs_data.rel_tol = rel_tol;
  pogs_data.max_iter = max_iter;
  pogs_data.quiet = quiet;
  pogs_data.adaptive_rho = adaptive_rho;
  pogs_data.gap_stop = gap_stop;

  // Solve.
  int err = Pogs(&pogs_data);
  *optval = pogs_data.optval;

  return err;
}

extern "C" {
int PogsD(enum ORD ord, size_t m, size_t n, const double *A,
          const double *f_a, const double *f_b, const double *f_c,
          const double *f_d, const double *f_e, const enum FUNCTION *f_h,
          const double *g_a, const double *g_b, const double *g_c,
          const double *g_d, const double *g_e, const enum FUNCTION *g_h,
          double rho, double abs_tol, double rel_tol, unsigned int max_iter,
          int quiet, int adaptive_rho, int gap_stop,
          double *x, double *y, double *l, double *optval) {
  if (ord == COL_MAJ) {
    return Pogs<double, COL_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        static_cast<bool>(quiet), static_cast<bool>(adaptive_rho), 
        static_cast<bool>(gap_stop), x, y, l, optval);
  } else {
    return Pogs<double, ROW_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        static_cast<bool>(quiet), static_cast<bool>(adaptive_rho), 
        static_cast<bool>(gap_stop), x, y, l, optval);
  }
}

int PogsS(enum ORD ord, size_t m, size_t n, const float *A,
          const float *f_a, const float *f_b, const float *f_c,
          const float *f_d, const float *f_e, const enum FUNCTION *f_h,
          const float *g_a, const float *g_b, const float *g_c,
          const float *g_d, const float *g_e, const enum FUNCTION *g_h,
          float rho, float abs_tol, float rel_tol, unsigned int max_iter,
          int quiet, int adaptive_rho, int gap_stop,
          float *x, float *y, float *l, float *optval) {
  if (ord == COL_MAJ) {
    return Pogs<float, COL_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        static_cast<bool>(quiet), static_cast<bool>(adaptive_rho), 
        static_cast<bool>(gap_stop), x, y, l, optval);
  } else {
    return Pogs<float, ROW_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        static_cast<bool>(quiet), static_cast<bool>(adaptive_rho),
        static_cast<bool>(gap_stop), x, y, l, optval);
  }
}

}

