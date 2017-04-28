#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "h2oaiglm.h"
#include "timer.h"

// Linear program in equality form.
//   minimize    c^T * x
//   subject to  Ax = b
//               x >= 0.
//
// See <h2oaiglm>/matlab/examples/lp_eq.m for detailed description.
template <typename T>
double LpEq(size_t m, size_t n) {
  std::vector<T> A((m + 1) * n);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));

  // Generate A and c according to:
  //   A = 1 / n * rand(m, n)
  //   c = 1 / n * rand(n, 1)
  for (unsigned int i = 0; i < (m + 1) * n; ++i)
    A[i] = u_dist(generator) / static_cast<T>(n);

  h2oaiglm::MatrixDense<T> A_('r', m + 1, n, A.data());
  h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > h2oaiglm_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
  std::vector<T> v(n);
  for (unsigned int i = 0; i < n; ++i)
    v[i] = u_dist(generator);

  f.reserve(m + 1);
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = 0; j < n; ++j)
      b_i += A[i * n + j] * v[j];
    f.emplace_back(kIndEq0, static_cast<T>(1), b_i);
  }
  f.emplace_back(kIdentity);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kIndGe0);

  double t = timer<double>();
  h2oaiglm_data.Solve(f, g);

  return timer<double>() - t;
}

template double LpEq<double>(size_t m, size_t n);
template double LpEq<float>(size_t m, size_t n);

