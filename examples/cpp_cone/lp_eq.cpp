#include <numeric>
#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Linear program in equality form.
//   minimize    c^T * x
//   subject to  b - Ax = 0
//               x >= 0.
//
// See <pogs>/matlab/examples/lp_eq.m for detailed description.
template <typename T>
double LpEq(size_t m, size_t n) {
  std::vector<T> A(m * n), b(m, static_cast<T>(0)), c(n);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));

  // Generate A and c according to:
  //   A = 1 / n * rand(m, n)
  //   c = 1 / n * rand(n, 1)
  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = u_dist(generator) / static_cast<T>(n);
  for (unsigned int i = 0; i < n; ++i)
    c[i] = u_dist(generator) / static_cast<T>(n);

  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
  std::vector<T> v(n);
  for (unsigned int i = 0; i < n; ++i)
    v[i] = u_dist(generator);

  for (unsigned int i = 0; i < m; ++i) {
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * v[j];
  }

  std::vector<ConeConstraint> Kx, Ky;
  std::vector<CONE_IDX> idx_x(n), idx_y(m);
  std::iota(std::begin(idx_x), std::end(idx_x), 0);
  std::iota(std::begin(idx_y), std::end(idx_y), 0);
  Kx.emplace_back(kConeNonNeg, idx_x);
  Ky.emplace_back(kConeZero, idx_y);

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsIndirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  double t = timer<double>();
  pogs_data.SetVerbose(5);
//  pogs_data.SetRelTol(static_cast<T>(1e-3));
//  pogs_data.SetAbsTol(static_cast<T>(1e-4));
  pogs_data.SetRelTol(static_cast<T>(1e-6));
  pogs_data.SetAbsTol(static_cast<T>(1e-6));

  pogs_data.SetMaxIter(10000u);
  pogs_data.Solve(b, c);

  return timer<double>() - t;
}

template double LpEq<double>(size_t m, size_t n);
template double LpEq<float>(size_t m, size_t n);

