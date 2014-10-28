#include <random>
#include <vector>

#include "mat_gen.h"
#include "pogs.h"
#include "timer.h"

// Linear program in equality form.
//   minimize    c^T * x
//   subject to  Ax = b
//               x >= 0.
//
// See <pogs>/matlab/examples/lp_eq.m for detailed description.
template <typename T>
double LpEq(int m, int n, int nnz) {
  std::vector<T> val(nnz);
  std::vector<int> col_ind(nnz);
  std::vector<int> row_ptr(m + 2);
  std::vector<T> x(n);
  std::vector<T> y(m + 1);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));

  // Generate A and c according to:
  //   A = 4 / n * rand(m, n)
  //   c = rand(n, 1)
  nnz = MatGenApprox(m + 1, n, nnz, val.data(), row_ptr.data(), col_ind.data(),
    static_cast<T>(0), static_cast<T>(4.0 / n));
  
  Sparse<T, int, ROW> A_(val.data(), row_ptr.data(), col_ind.data(), nnz);
  PogsData<T, Sparse<T, int, ROW>> pogs_data(A_, m + 1, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();


  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
  std::vector<T> v(n);
  for (unsigned int i = 0; i < n; ++i)
    v[i] = 1 + u_dist(generator);

  pogs_data.f.reserve(m + 1);
  for (unsigned int i = 0; i < m; ++i) {
    T b_i = static_cast<T>(0);
    for (unsigned int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
      b_i += val[j] * v[col_ind[j]];
    pogs_data.f.emplace_back(kIndEq0, static_cast<T>(1), b_i);
  }
  pogs_data.f.emplace_back(kIdentity);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kIndGe0, static_cast<T>(1), static_cast<T>(1));

  double t = timer<double>();
  Pogs(&pogs_data);

  return timer<double>() - t;
}

template double LpEq<double>(int m, int n, int nnz);
template double LpEq<float>(int m, int n, int nnz);

