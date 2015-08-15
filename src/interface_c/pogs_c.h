#ifndef POGS_C_H
#define POGS_C_H

#include <stddef.h>


// TODO: arg for projector? or do we leave dense-direct and sparse-indirect pairings fixed?
// TODO: get primal and dual variables and rho
// TODO: set primal and dual variables and rho
// TODO: methods for warmstart with x,nu,rho,options
// TODO: pass in void *
// TODO: pogs shutdown
// TODO: julia finalizer

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
// - real_t *nu        : Array for dual vector nu.
// - real_t *optval    : Pointer to single real for f(y^*) + g(x^*).
// - uint final_iter   : # of iterations at termination
//
// Author: Chris Fougner (fougner@stanford.edu)
//

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

// created and managed by caller
template <typename T>
struct PogsSettings{
  T rho, abs_tol, rel_tol;
  unsigned int max_iters, verbose;
  int adaptive_rho, gap_stop, warm_start;
};

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


// created and managed by caller
template <typename T>
struct PogsInfo{
    unsigned int iter;
    int status;
    T obj, rho;
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


// created and managed by caller
template <typename T>
struct PogsSolution{
    T *x, *y, *mu, *nu; 
};

struct PogsSolutionS{
    float *x, *y, *mu, *nu; 
};

struct PogsSolutionD{
    double *x, *y, *mu, *nu; 
};

// created and managed locally
struct PogsWork{
    size_t m,n;
    bool directbit, densebit, rowmajorbit;
    void *pogs_data, *f, *g;

    PogsWork(size_t m_, size_t n_, bool direct_, bool dense_, bool rowmajor_, void *pogs_data_, void *f_, void *g_){
      m=m_;n=n_;
      directbit=direct_; densebit=dense_; rowmajorbit=rowmajor_;
      pogs_data= pogs_data_; f=f_; g=g_;
    }
};


bool VerifyPogsWork(void * work);

// Dense
template <typename T>
void * PogsInit(size_t m, size_t n, const T *A, const char ord);

// Sparse 
template <typename T>
void * PogsInit(size_t m, size_t n, size_t nnz, const T *nzvals, const int *nzindices, const int *pointers, const char ord);

// TODO: check implementation efficiency (currently suspect)
template <typename T>
void PogsFunctionUpdate(size_t m, std::vector<FunctionObj<T> > *f, const T *f_a, const T *f_b, const T *f_c, 
                          const T *f_d, const T *f_e, const FUNCTION *f_h);

template <typename T>
void PogsRun(pogs::PogsDirect<T, pogs::MatrixDense<T> > &pogs_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
  const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution);


template <typename T>
void PogsRun(pogs::PogsDirect<T, pogs::MatrixSparse<T> > &pogs_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution);

template<typename T>
void PogsRun(pogs::PogsIndirect<T, pogs::MatrixDense<T> > &pogs_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution);

template<typename T>
void PogsRun(pogs::PogsIndirect<T, pogs::MatrixSparse<T> > &pogs_data, const std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution);

template<typename T>
int PogsRun(void *work, const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e, const FUNCTION *f_h,
            const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e, const FUNCTION *g_h,
            void *settings, void *info, void *solution);

template<typename T>
void PogsShutdown(void * work);

                

#ifdef __cplusplus
extern "C" {
#endif


void * pogs_init_dense_single(enum ORD ord, size_t m, size_t n, const float *A);
void * pogs_init_dense_double(enum ORD ord, size_t m, size_t n, const double *A);
void * pogs_init_sparse_single(enum ORD ord, size_t m, size_t n, size_t nnz, const float *nzvals, const int *indices, const int *pointers);
void * pogs_init_sparse_double(enum ORD ord, size_t m, size_t n, size_t nnz, const double *nzvals, const int *indices, const int *pointers);
int pogs_solve_single(void *work, PogsSettingsS *settings, PogsSolutionS *solution, PogsInfoS *info,
                      const float *f_a, const float *f_b, const float *f_c,const float *f_d, const float *f_e, const enum FUNCTION *f_h,
                      const float *g_a, const float *g_b, const float *g_c,const float *g_d, const float *g_e, const enum FUNCTION *g_h);
int pogs_solve_double(void *work, PogsSettingsD *settings, PogsSolutionD *solution, PogsInfoD *info,
                      const double *f_a, const double *f_b, const double *f_c,const double *f_d, const double *f_e, const enum FUNCTION *f_h,
                      const double *g_a, const double *g_b, const double *g_c,const double *g_d, const double *g_e, const enum FUNCTION *g_h);
void pogs_finish_single(void * work);
void pogs_finish_double(void * work);


#ifdef __cplusplus
}
#endif

#endif  // POGS_C_H











