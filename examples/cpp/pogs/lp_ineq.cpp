#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "h2ogpumlglm.h"
#include "timer.h"

// Linear program in inequality form.
//   minimize    c^T * x
//   subject to  Ax <= b.
//
// See <h2ogpuml>/matlab/examples/lp_ineq.m for detailed description.
template <typename T>
double LpIneq(size_t m, size_t n) {
  std::vector<T> A(m * n);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  // Generate A according to:
  //   A = [-1 / n *rand(m - n, n); -eye(n)]
  for (size_t i = 0; i < (m - n) * n; ++i)
    A[i] = -static_cast<T>(1) / static_cast<T>(n) * u_dist(generator);
  for (size_t i = (m - n) * n; i < m * n; ++i)
    A[i] = (i - (m - n) * n) % (n + 1) == 0 ? static_cast<T>(-1) : static_cast<T>(0);

  h2ogpuml::MatrixDense<T> A_('r', m, n, A.data());
  h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > h2ogpuml_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  // Generate b according to:
  //   b = A * rand(n, 1) + 0.2 * rand(m, 1)
  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = 0; j < n; ++j)
      b_i += A[i * n + j] * u_dist(generator);
    b_i += static_cast<T>(0.2) * u_dist(generator);
    f.emplace_back(kIndLe0, static_cast<T>(1), b_i);
  }

  // Generate c according to:
  //   c = rand(n, 1)
  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kIdentity, u_dist(generator) / n);
 
  double t = timer<double>();
  h2ogpuml_data.Solve(f, g);

  return timer<double>() - t;
}

template double LpIneq<double>(size_t m, size_t n);
template double LpIneq<float>(size_t m, size_t n);

