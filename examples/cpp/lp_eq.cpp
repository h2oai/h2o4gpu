#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Linear program in equality form.
//   minimize    c^T * x
//   subject to  Ax = b
//               x >= 0.
//
// See <pogs>/matlab/examples/lp_eq.m for detailed description.
template <typename T>
double LpEq(size_t m, size_t n) {
  std::vector<T> A((m + 1) * n);
  std::vector<T> x(n);
  std::vector<T> y(m + 1);
  std::vector<T> mu(n);
  std::vector<T> nu(m+1);
  std::vector<T> x12(n);
  std::vector<T> y12(m+1);
  std::vector<T> mu12(n);
  std::vector<T> nu12(m+1);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));

  // Generate A and c according to:
  //   A = 4 / n * rand(m, n)
  //   c = rand(n, 1)
  for (unsigned int i = 0; i < (m + 1) * n; ++i)
    A[i] = u_dist(generator) / static_cast<T>(n);

  Dense<T, ROW> A_(A.data());
  PogsData<T, Dense<T, ROW>> pogs_data(A_, m + 1, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();
  pogs_data.nu = nu.data();
  pogs_data.mu = mu.data();
  pogs_data.x12 = x12.data();
  pogs_data.y12 = y12.data();
  pogs_data.mu12 = mu12.data();
  pogs_data.nu12 = nu12.data();

  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
  std::vector<T> v(n);
  for (unsigned int i = 0; i < n; ++i)
    v[i] = u_dist(generator);

  pogs_data.f.reserve(m + 1);
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = 0; j < n; ++j)
      b_i += A[i * n + j] * v[j];
    pogs_data.f.emplace_back(kIndEq0, static_cast<T>(1), b_i);
  }
  pogs_data.f.emplace_back(kIdentity);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kIndGe0);

  double t = timer<double>();
  Pogs(&pogs_data);

  return timer<double>() - t;
}

template double LpEq<double>(size_t m, size_t n);
template double LpEq<float>(size_t m, size_t n);

