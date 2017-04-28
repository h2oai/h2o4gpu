#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "h2oaiglm.h"
#include "timer.h"

using namespace h2oaiglm;

// Support Vector Machine.
//   minimize    (1/2) ||w||_2^2 + \lambda \sum (a_i^T * [w; b] + 1)_+.
//
// See <h2oaiglm>/matlab/examples/svm.m for detailed description.
template <typename T>
double Svm(size_t m, size_t n) {
  std::vector<T> A(m * (n + 1));

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

  h2oaiglm::MatrixDense<T> A_('r', m, n + 1, A.data());
  h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > h2oaiglm_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  T lambda = static_cast<T>(1);

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kMaxPos0, static_cast<T>(1),
                             static_cast<T>(-1), lambda);

  g.reserve(n + 1);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kSquare);
  g.emplace_back(kZero);

  double t = timer<double>();
  h2oaiglm_data.Solve(f, g);

  return timer<double>() - t;
}

template double Svm<double>(size_t m, size_t n);
template double Svm<float>(size_t m, size_t n);

