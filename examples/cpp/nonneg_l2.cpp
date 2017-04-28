#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "h2oaiglm.h"
#include "timer.h"

using namespace h2oaiglm;

// Non-Negative Least Squares.
//   minimize    (1/2) ||Ax - b||_2^2
//   subject to  x >= 0.
//
// See <h2oaiglm>/matlab/examples/nonneg_l2.m for detailed description.
template <typename T>
double NonNegL2(size_t m, size_t n) {
  std::vector<T> A(m * n);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  // Generate A according to:
  //   A = 1 / n * rand(m, n)
  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = static_cast<T>(1) / static_cast<T>(n) * u_dist(generator);

  h2oaiglm::MatrixDense<T> A_('r', m, n, A.data());
  h2oaiglm::H2OAIGLMDirect<T, h2oaiglm::MatrixDense<T> > h2oaiglm_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  // Generate b according to:
  //   n_half = floor(2 * n / 3);
  //   b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.1 * randn(m, 1)
  f.reserve(m); 
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = 0; j < n; j++)
      b_i += 3 * j < 2 * n ? A[i * n + j] : -A[i * n + j];
    b_i += static_cast<T>(0.01) * n_dist(generator);
    f.emplace_back(kSquare, static_cast<T>(1), b_i);
  }

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kIndGe0);

  double t = timer<double>();
  h2oaiglm_data.Solve(f, g);

  return timer<double>() - t;
}

template double NonNegL2<double>(size_t m, size_t n);
template double NonNegL2<float>(size_t m, size_t n);

