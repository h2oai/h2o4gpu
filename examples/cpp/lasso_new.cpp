#include <random>
#include <vector>

#include "pogs.h"
#include "matrix/matrix_dense.h"
#include "timer.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
double Lasso(size_t m, size_t n) {
  std::vector<T> A(m * n);
  std::vector<T> b(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = n_dist(generator);

  std::vector<T> x_true(n);
  for (unsigned int i = 0; i < n; ++i)
    x_true[i] = u_dist(generator) < 0.8 ? 0 : n_dist(generator) / n;

#pragma omp parallel for
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * x_true[j];
      // b[i] += A[i + j * m] * x_true[j];

  for (unsigned int i = 0; i < m; ++i)
    b[i] += static_cast<T>(0.5) * n_dist(generator);

  T lambda_max = static_cast<T>(0);
#pragma omp parallel for reduction(max : lambda_max)
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      //u += A[i * n + j] * b[i];
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  //pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);
  pogs::PogsIndirect<T, pogs::MatrixDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double Lasso<double>(size_t m, size_t n);
template double Lasso<float>(size_t m, size_t n);

