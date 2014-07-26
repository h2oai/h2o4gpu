#ifndef POGS_H_
#define POGS_H_

#include <vector>

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

  // Factors (dim = 1 + 2(m+n) + min(m,n)^2).
  T *factors;

  // Constructor.
  PogsData(const M &A, size_t m, size_t n)
      : A(A), m(m), n(n), x(0), y(0), l(0), rho(1),
        abs_tol(static_cast<T>(1e-4)), rel_tol(static_cast<T>(1e-3)),
        max_iter(1000), quiet(false), adaptive_rho(true), factors(0) { }
};

template <typename T, typename M>
void Pogs(PogsData<T, M> *pogs_data);

#endif  // POGS_H_

