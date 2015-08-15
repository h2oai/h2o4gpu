#ifndef POGS_C_H
#define POGS_C_H

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
enum STATUS { POGS_SUCCESS,    // Converged succesfully.
      POGS_INFEASIBLE, // Problem likely infeasible.
      POGS_UNBOUNDED,  // Problem likely unbounded
      POGS_MAX_ITER,   // Reached max iter.
      POGS_NAN_FOUND,  // Encountered nan.
      POGS_ERROR };    // Generic error, check logs.

struct PogsSettingsS{
  float rho, abs_tol, rel_tol;
  unsigned int max_iters, verbose;
  int adaptive_rho, gap_stop, warm_start;
};

struct PogsSettingsD{
  double rho, abs_tol, rel_tol;
  unsigned int max_iters, verbose;
  int adaptive_rho, gap_stop, warm_start;
};

struct PogsInfoS{
    unsigned int iter;
    int status;
    float obj, rho;
};

struct PogsInfoD{
    unsigned int iter;
    int status;
    double obj, rho;
};

struct PogsSolutionS{
    float *x, *y, *mu, *nu; 
};

struct PogsSolutionD{
    double *x, *y, *mu, *nu; 
};




void * pogs_init_dense_single(enum ORD ord, size_t m, size_t n, const float *A);
void * pogs_init_dense_double(enum ORD ord, size_t m, size_t n, const double *A);
void * pogs_init_sparse_single(enum ORD ord, size_t m, size_t n, size_t nnz, const float *nzvals, const int *indices, const int *pointers);
void * pogs_init_sparse_double(enum ORD ord, size_t m, size_t n, size_t nnz, const double *nzvals, const int *indices, const int *pointers);
int pogs_solve_single(void *work, struct PogsSettingsS *settings, struct PogsSolutionS *solution, struct PogsInfoS *info,
                      const float *f_a, const float *f_b, const float *f_c,const float *f_d, const float *f_e, const enum FUNCTION *f_h,
                      const float *g_a, const float *g_b, const float *g_c,const float *g_d, const float *g_e, const enum FUNCTION *g_h);
int pogs_solve_double(void *work, struct PogsSettingsD *settings, struct PogsSolutionD *solution, struct PogsInfoD *info,
                      const double *f_a, const double *f_b, const double *f_c,const double *f_d, const double *f_e, const enum FUNCTION *f_h,
                      const double *g_a, const double *g_b, const double *g_c,const double *g_d, const double *g_e, const enum FUNCTION *g_h);
void pogs_finish_single(void * work);
void pogs_finish_double(void * work);


#ifdef __cplusplus
}
#endif

#endif  // POGS_C_H











