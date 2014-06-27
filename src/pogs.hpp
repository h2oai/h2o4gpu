#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <vector>

#include "prox_lib.hpp"

// Data structure for input to Solver().
template <typename T, typename M>
struct PogsData {
  // Input.
  std::vector<FunctionObj<T> > f, g;
  const M A;
  size_t m, n;

  // Output.
  T *x, *y, *x_dual, *y_dual, optval;

  // Parameters.
  T rho;
  unsigned int max_iter;
  T rel_tol, abs_tol;
  bool quiet;

  // Constructor.
  PogsData(const M &A, size_t m, size_t n)
      : A(A), m(m), n(n), x(0), y(0), rho(static_cast<T>(1)), max_iter(1000),
        rel_tol(static_cast<T>(1e-3)), abs_tol(static_cast<T>(1e-4)),
        quiet(false) { }
};

template <typename T, typename M>
void Pogs(PogsData<T, M> *pogs_data);

#endif /* SOLVER_HPP_ */

