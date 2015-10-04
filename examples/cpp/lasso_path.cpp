#include <limits>
#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

template <typename T>
T MaxDiff(std::vector<T> *v1, std::vector<T> *v2) {
  T max_diff = 0;
#pragma omp parallel for reduction(max : max_diff)
  for (unsigned int i = 0; i < v1->size(); ++i)
    max_diff = std::max(max_diff, std::abs((*v1)[i] - (*v2)[i]));
  return max_diff;
}

template <typename T>
T Asum(std::vector<T> *v) {
  T asum = 0;
#pragma omp parallel for reduction(+ : asum)
  for (unsigned int i = 0; i < v->size(); ++i)
    asum += std::abs((*v)[i]);
  return asum;
}

// LassoPath
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// for 50 values of \lambda.
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double LassoPath(size_t m, size_t n) {
  unsigned int nlambda = 100;
  std::vector<T> A(m * n);
  std::vector<T> b(m);
  std::vector<T> x(n);
  std::vector<T> x_last(n, std::numeric_limits<T>::max());
  std::vector<T> y(m);
  std::vector<T> mu(n);
  std::vector<T> nu(m);
  std::vector<T> x12(n);
  std::vector<T> y12(m);
  std::vector<T> mu12(n);
  std::vector<T> nu12(m);

  // Generate data
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

  for (unsigned int i = 0; i < m; ++i) {
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * x_true[j];
      // b[i] += A[i + j * m] * x_true[j];
    b[i] += static_cast<T>(0.5) * n_dist(generator);
  }

  T lambda_max = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      //u += A[i * n + j] * b[i];
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  // Set up pogs datastructure.
  Dense<T, ROW> A_(A.data());
  PogsData<T, Dense<T, ROW>> pogs_data(A_, m, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();
  pogs_data.nu = nu.data();
  pogs_data.mu = mu.data();
  pogs_data.x12 = x12.data();
  pogs_data.y12 = y12.data();
  pogs_data.mu12 = mu12.data();
  pogs_data.nu12 = nu12.data();



  pogs_data.f.reserve(m);
  pogs_data.g.reserve(n);

  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kAbs);
  
  AllocDenseFactors(&pogs_data);

  double t = timer<double>();
  for (unsigned int i = 0; i < nlambda; ++i) {
    T lambda = std::exp((std::log(lambda_max) * (nlambda - 1 - i) +
        static_cast<T>(1e-2) * std::log(lambda_max) * i) / (nlambda - 1));

    for (unsigned int i = 0; i < n; ++i)
      pogs_data.g[i].c = lambda;

    Pogs(&pogs_data);

    if (MaxDiff(&x, &x_last) < 1e-3 * Asum(&x))
      break;
    x_last = x;
  }
  FreeDenseFactors(&pogs_data);

  return timer<double>() - t;
}

template double LassoPath<double>(size_t m, size_t n);
template double LassoPath<float>(size_t m, size_t n);

