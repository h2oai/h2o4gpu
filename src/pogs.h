#ifndef POGS_H_
#define POGS_H_

#include <vector>

#ifdef __CUDACC__
#include "cml/cblas.h"
#else
#include "gsl/cblas.h"
#endif
#include "prox_lib.h"

// Data structure for input to Pogs().
template <typename T, typename M>
struct PogsData {
  // Input.
  std::vector<FunctionObj<T> > f, g;
  const M A;
  size_t m, n;

  // Output.
  T *x, *y, *l, optval;

  // Parameters.
  T rho, abs_tol, rel_tol;
  unsigned int max_iter;
  bool quiet, adaptive_rho;

  // Factors
  M factors;

  // Constructor.
  PogsData(const M &A, size_t m, size_t n)
      : A(A), m(m), n(n), x(0), y(0), l(0), rho(1),
        abs_tol(static_cast<T>(1e-4)), rel_tol(static_cast<T>(1e-3)),
        max_iter(2000), quiet(false), adaptive_rho(true), factors(0) { }
};

// Pogs solver.
template <typename T, typename M>
int Pogs(PogsData<T, M> *pogs_data);

// Dense matrix type.
template <typename T, CBLAS_ORDER O>
struct Dense {
  static const CBLAS_ORDER Ord = O;
  T *val;
  Dense(T *val) : val(val) { }; 
};

// Sparse matrix type.
template <typename T>
struct CSC {
  T *val;
  size_t *col_ptr;
  size_t *row_ind;
};

// Factor allocation and freeing.
template <typename T, CBLAS_ORDER O>
int AllocDenseFactors(PogsData<T, Dense<T, O> > *pogs_data);

template <typename T, CBLAS_ORDER O>
void FreeDenseFactors(PogsData<T, Dense<T, O> > *pogs_data);

template <typename T>
int AllocSparseFactors(PogsData<T, CSC<T> > *pogs_data);

template <typename T>
void FreeSparseFactors(PogsData<T, CSC<T> > *pogs_data);

#endif  // POGS_H_

