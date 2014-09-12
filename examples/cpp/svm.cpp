#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Support Vector Machine.
//   minimize    (1/2) ||w||_2^2 + \lambda \sum (a_i^T * [w; b] + 1)_+.
//
// See <pogs>/matlab/examples/svm.m for detailed description.
template <typename T>
double Svm(size_t m, size_t n) {
  std::vector<T> A(m * (n + 1));
  std::vector<T> x(n + 1);
  std::vector<T> y(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  // Generate A according to:
  //   x = [randn(N, n) + ones(N, n); randn(N, n) - ones(N, n)]
  //   y = [ones(N, 1); -ones(N, 1)]
  //   A = [(-y * ones(1, n)) .* x, -y]
  for (unsigned int i = 0; i < m; ++i) {
    T sign_yi = i < m / 2 ? static_cast<T>(1) :
                                 static_cast<T>(-1);
    for (unsigned int j = 0; j < n; ++j) {
      A[i * (n + 1) + j] = -sign_yi * (n_dist(generator) + sign_yi);
    }
    A[i * (n + 1) + n] = -sign_yi;
  }

  Dense<T, ROW> A_(A.data());
  PogsData<T, Dense<T, ROW>> pogs_data(A_, m, n + 1);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  T lambda = static_cast<T>(1);

  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kMaxPos0, static_cast<T>(1),
                             static_cast<T>(-1), lambda);

  pogs_data.g.reserve(n + 1);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kSquare);
  pogs_data.g.emplace_back(kZero);

  double t = timer<double>();
  Pogs(&pogs_data);
  return timer<double>() - t;
}

template double Svm<double>(size_t m, size_t n);
template double Svm<float>(size_t m, size_t n);

