#ifndef POGS_H_
#define POGS_H_

#include <vector>

#if defined(__CUDACC__) || defined(__CUDA)
#include "cml/cblas.h"
#else
#include "gsl/cblas.h"
#endif
#include "prox_lib.h"

enum SP_FORMAT { CSC, CSR };
typedef int INT;

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
template <typename T, typename I, SP_FORMAT F>
struct Sparse {
  static const SP_FORMAT Fmt = F;
  T *val;
  I *ptr;
  I *ind;
  I nnz;
  Sparse(T *val, I *ptr, I *ind, I nnz)
      : val(val), ptr(ptr), ind(ind), nnz(nnz) { };
};

// Factor allocation and freeing.
template <typename T, CBLAS_ORDER O>
int AllocDenseFactors(PogsData<T, Dense<T, O> > *pogs_data);

template <typename T, CBLAS_ORDER O>
void FreeDenseFactors(PogsData<T, Dense<T, O> > *pogs_data);

template <typename T, SP_FORMAT F>
int AllocSparseFactors(PogsData<T, Sparse<T, F> > *pogs_data);

template <typename T, SP_FORMAT F>
void FreeSparseFactors(PogsData<T, Sparse<T, F> > *pogs_data);

#endif  // POGS_H_

