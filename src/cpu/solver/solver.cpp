#ifndef SOLVER_SOLVER_H_
#define SOLVER_SOLVER_H_
// Setup,
// Solves ||Ax - b||_2  + s ||x||^2
#include "matrix/matrix.h"

template <typename T, typename M>
class Solver {
 public:
  Solver();

  virtual ~Solver();

  virtual int Solve(const M& A, T s, T *x) = 0;
};

// TODO: How is this going to work with matrix data :O

#endif  // SOLVER_SOLVER_H_

