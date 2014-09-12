#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Non-Negative Least Squares.
//   minimize    (1/2) ||Ax - b||_2^2
//   subject to  x >= 0.
//
// See <pogs>/matlab/examples/nonneg_l2.m for detailed description.
template <typename T>
double NonNegL2(size_t m, size_t n) {
  std::vector<T> A(m * n);
  std::vector<T> x(n);
  std::vector<T> y(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  // Generate A according to:
  //   A = 1 / n * rand(m, n)
  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = static_cast<T>(1) / static_cast<T>(n) * u_dist(generator);

  Dense<T, ROW> A_(A.data());
  PogsData<T, Dense<T, ROW>> pogs_data(A_, m, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  // Generate b according to:
  //   n_half = floor(2 * n / 3);
  //   b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.1 * randn(m, 1)
  pogs_data.f.reserve(m); 
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = 0; j < n; j++)
      b_i += 3 * j < 2 * n ? A[i * n + j] : -A[i * n + j];
    b_i += static_cast<T>(0.01) * n_dist(generator);
    pogs_data.f.emplace_back(kSquare, static_cast<T>(1), b_i);
  }

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kIndGe0);

  double t = timer<double>();
  Pogs(&pogs_data);

  return timer<double>() - t;
}

template double NonNegL2<double>(size_t m, size_t n);
template double NonNegL2<float>(size_t m, size_t n);

