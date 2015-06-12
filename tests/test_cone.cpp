#include "pogs.h"

/*
 *  min. [1 0] [x_1 x_2]'
 *  s.t.
 *       [-1 -1]               [0]
 *       [ 1 -1] [x_1 x_2]' <= [0]
 *       [ 0  1]               [2]
 */
template <typename T>
void lp_cone_dense_row_direct() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., -1., 1., -1., 0., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(1000);
  pogs_data.SetVerbose(5);
  pogs_data.Solve(b, c);
}

template <typename T>
void lp_cone_dense_row_indirect() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., -1., 1., -1., 0., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(1000);
  pogs_data.SetVerbose(5);
  pogs_data.Solve(b, c);
}

template <typename T>
void lp_cone_dense_col_direct() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., 1., 0., -1., -1., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('c', m, n, A.data());
  pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(1000);
  pogs_data.SetVerbose(5);
  pogs_data.Solve(b, c);
}

template <typename T>
void lp_cone_dense_col_indirect() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., 1., 0., -1., -1., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('c', m, n, A.data());
  pogs::PogsIndirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(1000);
  pogs_data.SetVerbose(5);
  pogs_data.Solve(b, c);
}


int main() {
  lp_cone_dense_row_direct<double>();
  lp_cone_dense_row_indirect<double>();
  lp_cone_dense_col_direct<double>();
  lp_cone_dense_col_indirect<double>();
}

