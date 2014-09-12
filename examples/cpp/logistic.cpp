#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Logistic
//   minimize    \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1
//   subject to  y = Ax
//
// for 50 values of \lambda.
// See <pogs>/matlab/examples/logistic_regression.m for detailed description.
template <typename T>
double Logistic(size_t m, size_t n) {
  std::vector<T> A(m * (n + 1));
  std::vector<T> d(m);
  std::vector<T> x(n + 1);
  std::vector<T> y(m);

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

#pragma omp parallel for
  for (unsigned int i = 0; i < m; ++i) {
    d[i] = 0;
    for (unsigned int j = 0; j < n + 1; ++j)
      // u += A[i + j * m] * x_true[j];
      d[i] += A[i * n + j] * x_true[j];
  }
  for (unsigned int i = 0; i < m; ++i)
    d[i] = 1 / (1 + std::exp(-d[i])) > u_dist(generator);

  T lambda_max = static_cast<T>(0);
#pragma omp parallel for reduction(max : lambda_max)
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      // u += A[i * n + j] * (static_cast<T>(0.5) - d[i]);
      u += A[i + j * m] * (static_cast<T>(0.5) - d[i]);
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  Dense<T, ROW> A_(A.data());
  PogsData<T, Dense<T, ROW>> pogs_data(A_, m, n + 1);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kLogistic, 1, 0, 1, -d[i]);

  pogs_data.g.reserve(n + 1);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);
  pogs_data.g.emplace_back(kZero);

  double t = timer<double>();
  Pogs(&pogs_data);

  return timer<double>() - t;
}

template double Logistic<double>(size_t m, size_t n);
template double Logistic<float>(size_t m, size_t n);

