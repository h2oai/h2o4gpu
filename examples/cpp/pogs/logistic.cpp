#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "h2ogpumlglm.h"
#include "timer.h"

using namespace h2ogpuml;

// Logistic
//   minimize    \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1
//   subject to  y = Ax
//
// for 50 values of \lambda.
// See <h2ogpuml>/matlab/examples/logistic_regression.m for detailed description.
template <typename T>
double Logistic(size_t m, size_t n) {
  std::vector<T> A(m * (n + 1));
  std::vector<T> d(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  for (unsigned int i = 0; i < m; ++i) {
    for (unsigned int j = 0; j < n; ++j)
      A[i * (n + 1) + j] = n_dist(generator);
    A[i * (n + 1) + n] = 1;
  }

  std::vector<T> x_true(n + 1);
  for (unsigned int i = 0; i < n; ++i)
    x_true[i] = u_dist(generator) < 0.8 ? 0 : n_dist(generator) / n;
  x_true[n] = n_dist(generator) / n;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < m; ++i) {
    d[i] = 0;
    for (unsigned int j = 0; j < n + 1; ++j)
      // u += A[i + j * m] * x_true[j];
      d[i] += A[i * n + j] * x_true[j];
  }
  for (unsigned int i = 0; i < m; ++i)
    d[i] = 1 / (1 + std::exp(-d[i])) > u_dist(generator);

  T lambda_max = static_cast<T>(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(max : lambda_max)
#endif
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      // u += A[i * n + j] * (static_cast<T>(0.5) - d[i]);
      u += A[i + j * m] * (static_cast<T>(0.5) - d[i]);
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  h2ogpuml::MatrixDense<T> A_('r', m, n + 1, A.data());
  h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > h2ogpuml_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kLogistic, 1, 0, 1, -d[i]);

  g.reserve(n + 1);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);
  g.emplace_back(kZero);

  double t = timer<double>();
  h2ogpuml_data.Solve(f, g);

  return timer<double>() - t;
}

template double Logistic<double>(size_t m, size_t n);
template double Logistic<float>(size_t m, size_t n);

