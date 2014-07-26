#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Linear program in inequality form.
//   minimize    c^T * x
//   subject to  Ax <= b.
//
// See <pogs>/matlab/examples/lp_ineq.m for detailed description.
template <typename T>
T LpIneq(size_t m, size_t n) {
  std::vector<T> A(m * n);
  std::vector<T> x(n);
  std::vector<T> y(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  // Generate A according to:
  //   A = [-1 / n *rand(m - n, n); -eye(n)]
  for (unsigned int i = 0; i < (m - n) * n; ++i)
    A[i] = -static_cast<T>(1) / static_cast<T>(n) * u_dist(generator);
  for (unsigned int i = static_cast<unsigned int>((m - n) * n); i < m * n; ++i)
    A[i] = (i - (m - n) * n) % (n + 1) == 0 ? -1 : 0;

  PogsData<T, T*> pogs_data(A.data(), m, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  // Generate b according to:
  //   b = A * rand(n, 1) + 0.2 * rand(m, 1)
  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = 0; j < n; ++j)
      b_i += A[i * n + j] * u_dist(generator);
    b_i += static_cast<T>(0.2) * u_dist(generator);
    pogs_data.f.emplace_back(kIndLe0, static_cast<T>(1), b_i);
  }

  // Generate c according to:
  //   c = rand(n, 1)
  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kIdentity, u_dist(generator));
 
  T t = timer<T>();
  Pogs(&pogs_data);

  return timer<T>() - t;
}

template double LpIneq<double>(size_t m, size_t n);
template float LpIneq<float>(size_t m, size_t n);

