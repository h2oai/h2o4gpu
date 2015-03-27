#ifndef POGS_C_H
#define POGS_C_H

#include <stddef.h>

// Wrapper for POGS, a solver for convex problems in the form
//   min. \sum_i f(y_i) + g(x_i)
//   s.t.  y = Ax,
//  where 
//   f_i(y_i) = c_i * h_i(a_i * y_i - b_i) + d_i * y_i + e_i * x_i^2,
//   g_i(x_i) = c_i * h_i(a_i * x_i - b_i) + d_i * x_i + e_i * x_i^2.
//
// Input arguments (real_t is either double or float)
// - ORD ord           : Specifies row/colum major ordering of matrix A.
// - size_t m          : First dimensions of matrix A.
// - size_t n          : Second dimensions of matrix A.
// - real_t *A         : Pointer to matrix A.
// - real_t *f_a-f_e   : Pointer to array of a_i-e_i's in function f_i(y_i).
// - FUNCTION *f_h     : Pointer to array of h_i's in function f_i(y_i).
// - real_t *g_a-g_e   : Pointer to array of a_i-e_i's in function g_i(x_i).
// - FUNCTION *g_h     : Pointer to array of h_i's in function g_i(x_i).
// - real_t rho        : Initial value for rho parameter.
// - real_t abs_tol    : Absolute tolerance (recommended 1e-4).
// - real_t rel_tol    : Relative tolerance (recommended 1e-3).
// - uint max_iter     : Maximum number of iterations (recommended 1e3-2e3).
// - int quiet         : Output to screen if quiet = 0.
// - int adaptive_rho  : No adaptive rho update if adaptive_rho = 0.
// - int gap_stop      : Additionally use the gap as a stopping criteria.
//
// Output arguments (real_t is either double or float)
// - real_t *x         : Array for solution vector x.
// - real_t *y         : Array for solution vector y.
// - real_t *l         : Array for dual vector lambda.
// - real_t *optval    : Pointer to single real for f(y^*) + g(x^*).
//
// Author: Chris Fougner (fougner@stanford.edu)
//

// Possible column and row ordering.
enum ORD { COL_MAJ, ROW_MAJ };

// Possible objective values.
enum FUNCTION { ABS,       // f(x) = |x|
                EXP,       // f(x) = e^x
                HUBER,     // f(x) = huber(x)
                IDENTITY,  // f(x) = x
                INDBOX01,  // f(x) = I(0 <= x <= 1)
                INDEQ0,    // f(x) = I(x = 0)
                INDGE0,    // f(x) = I(x >= 0)
                INDLE0,    // f(x) = I(x <= 0)
                LOGISTIC,  // f(x) = log(1 + e^x)
                MAXNEG0,   // f(x) = max(0, -x)
                MAXPOS0,   // f(x) = max(0, x)
                NEGENTR,   // f(x) = x log(x)
                NEGLOG,    // f(x) = -log(x)
                RECIPR,    // f(x) = 1/x
                SQUARE,    // f(x) = (1/2) x^2
                ZERO };    // f(x) = 0

#ifdef __cplusplus
extern "C" {
#endif

int PogsD(enum ORD ord, size_t m, size_t n, double *A,
          double *f_a, double *f_b, double *f_c, double *f_d, double *f_e,
          enum FUNCTION *f_h,
          double *g_a, double *g_b, double *g_c, double *g_d, double *g_e,
          enum FUNCTION *g_h,
          double rho, double abs_tol, double rel_tol, unsigned int max_iter,
          int verbose, int adaptive_rho, int gap_stop,
          double *x, double *y, double *l, double *optval);

int PogsS(enum ORD ord, size_t m, size_t n, float *A,
          float *f_a, float *f_b, float *f_c, float *f_d, float *f_e,
          enum FUNCTION *f_h,
          float *g_a, float *g_b, float *g_c, float *g_d, float *g_e,
          enum FUNCTION *g_h,
          float rho, float abs_tol, float rel_tol, unsigned int max_iter,
          int verbose, int adaptive_rho, int gap_stop,
          float *x, float *y, float *l, float *optval);

#ifdef __cplusplus
}
#endif

#endif  // POGS_C_H

