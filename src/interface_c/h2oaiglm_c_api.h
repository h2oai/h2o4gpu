#ifndef H2OAIGLM_C_H
#define H2OAIGLM_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
  
// Possible column and row ordering.
enum ORD { COL_MAJ, ROW_MAJ };

// Possible projector implementations
enum PROJECTOR { DIRECT, INDIRECT };

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

// Possible status values.
enum STATUS { H2OAIGLM_SUCCESS,    // Converged succesfully.
      H2OAIGLM_INFEASIBLE, // Problem likely infeasible.
      H2OAIGLM_UNBOUNDED,  // Problem likely unbounded
      H2OAIGLM_MAX_ITER,   // Reached max iter.
      H2OAIGLM_NAN_FOUND,  // Encountered nan.
      H2OAIGLM_ERROR };    // Generic error, check logs.

struct PogsSettingsS{
  float rho, abs_tol, rel_tol;
  unsigned int max_iters, verbose;
  int adaptive_rho, equil, gap_stop, warm_start;
  int nDev,wDev;
};

struct PogsSettingsD{
  double rho, abs_tol, rel_tol;
  unsigned int max_iters, verbose;
  int adaptive_rho, equil, gap_stop, warm_start;
  int nDev,wDev;
};

struct PogsInfoS{
    unsigned int iter;
    int status;
    float obj, rho, solvetime;
};

struct PogsInfoD{
    unsigned int iter;
    int status;
    double obj, rho, solvetime;
};

struct PogsSolutionS{
    float *x, *y, *mu, *nu; 
};

struct PogsSolutionD{
    double *x, *y, *mu, *nu; 
};




void * h2oaiglm_init_dense_single(int wDev, enum ORD ord, size_t m, size_t n, const float *A);
void * h2oaiglm_init_dense_double(int wDev, enum ORD ord, size_t m, size_t n, const double *A);
void * h2oaiglm_init_sparse_single(int wDev, enum ORD ord, size_t m, size_t n, size_t nnz, const float *nzvals, const int *indices, const int *pointers);
void * h2oaiglm_init_sparse_double(int wDev, enum ORD ord, size_t m, size_t n, size_t nnz, const double *nzvals, const int *indices, const int *pointers);
int h2oaiglm_solve_single(void *work, struct PogsSettingsS *settings, struct PogsSolutionS *solution, struct PogsInfoS *info,
                      const float *f_a, const float *f_b, const float *f_c,const float *f_d, const float *f_e, const enum FUNCTION *f_h,
                      const float *g_a, const float *g_b, const float *g_c,const float *g_d, const float *g_e, const enum FUNCTION *g_h);
int h2oaiglm_solve_double(void *work, struct PogsSettingsD *settings, struct PogsSolutionD *solution, struct PogsInfoD *info,
                      const double *f_a, const double *f_b, const double *f_c,const double *f_d, const double *f_e, const enum FUNCTION *f_h,
                      const double *g_a, const double *g_b, const double *g_c,const double *g_d, const double *g_e, const enum FUNCTION *g_h);
void h2oaiglm_finish_single(void * work);
void h2oaiglm_finish_double(void * work);

#ifdef __cplusplus
}
#endif

#endif  // H2OAIGLM_C_H











