#include <random>
#include <vector>

#include "matrix/matrix_sparse.h"
#include "mat_gen.h"
#include "h2ogpuml.h"
#include "timer.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <h2ogpuml>/matlab/examples/lasso.m for detailed description.
template <typename T>
double Lasso(int m, int n, int nnz) {
  char ord = 'c';

  std::vector<T> A(m * n, static_cast<T>(0.));
  std::vector<T> val(nnz);
  std::vector<int> col_ind(nnz);
  std::vector<int> row_ptr;
  std::vector<T> b(m);
  std::vector<T> x(n);
  std::vector<T> y(m);

  std::default_random_engine generator;
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));
 
  std::vector<std::tuple<int, int, T>> entries;
  if (ord == 'r') {
    row_ptr.reserve(m + 1);
    nnz = MatGenApprox(m, n, nnz, val.data(), row_ptr.data(), col_ind.data(),
        static_cast<T>(-1), static_cast<T>(1), entries);
    for (int i = 0; i < m; ++i) {
      for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        A[i * n + col_ind[k]] = val[k];
    }
  } else {
    row_ptr.reserve(n + 1);
    nnz = MatGenApprox(n, m, nnz, val.data(), row_ptr.data(), col_ind.data(),
        static_cast<T>(-1), static_cast<T>(1), entries);
    for (int j = 0; j < n; ++j) {
      for (int k = row_ptr[j]; k < row_ptr[j + 1]; ++k)
        A[col_ind[k] + j * m] = val[k];
    }
  }

  for (unsigned int i = 0; i < m; ++i)
    b[i] = static_cast<T>(4) * n_dist(generator);

  T lambda_max = 1;

  h2ogpuml::MatrixSparse<T> A_(ord, m, n, nnz, val.data(), row_ptr.data(),
      col_ind.data());
  h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T>> h2ogpuml_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);

  double t = timer<double>();
  h2ogpuml_data.Solve(f, g);

  return timer<double>() - t;
}

template double Lasso<double>(int m, int n, int nnz);
template double Lasso<float>(int m, int n, int nnz);

