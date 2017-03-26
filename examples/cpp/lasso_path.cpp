#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

using namespace pogs;

template <typename T>
T MaxDiff(std::vector<T> *v1, std::vector<T> *v2) {
  T max_diff = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max : max_diff)
#endif
  for (unsigned int i = 0; i < v1->size(); ++i)
    max_diff = std::max(max_diff, std::abs((*v1)[i] - (*v2)[i]));
  return max_diff;
}

template <typename T>
T Asum(std::vector<T> *v) {
  T asum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : asum)
#endif
  for (unsigned int i = 0; i < v->size(); ++i)
    asum += std::abs((*v)[i]);
  return asum;
}

// LassoPath
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// for 100 values of \lambda.
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double LassoPath(size_t m, size_t n) {
  unsigned int nlambda = 100;
  std::vector<T> A(m * n);
  std::vector<T> b(m);
  std::vector<T> x_last(n, std::numeric_limits<T>::max());

  // Generate data
  fprintf(stdout,"BEGIN FILL DATA\n");
  double t0 = timer<double>();
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
  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;
  f.reserve(m);
  g.reserve(n);

  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs);

  double t1 = timer<double>();
  fprintf(stdout,"END FILL DATA\n");

  fprintf(stdout,"BEGIN SOLVE\n");
  double t = timer<double>();
  for (unsigned int i = 0; i < nlambda; ++i) {
    T lambda = std::exp((std::log(lambda_max) * (nlambda - 1 - i) +
        static_cast<T>(1e-2) * std::log(lambda_max) * i) / (nlambda - 1));

    for (unsigned int i = 0; i < n; ++i)
      g[i].c = lambda;

    pogs_data.Solve(f, g);

    std::vector<T> x(n);
    for (unsigned int i = 0; i < n; ++i)
      x[i] = pogs_data.GetX()[i];

    if (MaxDiff(&x, &x_last) < 1e-3 * Asum(&x))
      break;
    x_last = x;
  }
  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 1 tfd %g ts %g\n",t1-t0,tf-t);

  return tf-t;
}

template double LassoPath<double>(size_t m, size_t n);
template double LassoPath<float>(size_t m, size_t n);

