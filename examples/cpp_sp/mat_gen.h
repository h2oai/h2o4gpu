#ifndef MAT_GEN_H_
#define MAT_GEN_H_

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T>
inline T rand(T lb, T ub) {
  return (ub - lb) * (rand() / static_cast<T>(RAND_MAX)) + lb;
}

template <typename T>
void MatGen(int m, int n, int nnz, T *val, int *rptr, int *cind, T lb, T ub) {
  T kRandMax = static_cast<T>(RAND_MAX);
  T kM = static_cast<T>(m);
  T kN = static_cast<T>(n);

  int num = 0;
  for (int i = 0; i < m; ++i) {
    rptr[i] = num;
    for (int j = 0; j < n && num < nnz; ++j) {
      if (rand() / kRandMax * ((kM - i) * kN - j) < (nnz - num)) {
        val[num] = rand(lb, ub);
        cind[num] = j;
        num++;
      }
    }
  }
  rptr[m] = nnz;
}

template <typename T>
int MatGenApprox(int m, int n, int nnz, T *val, int *rptr, int *cind, T lb,
                 T ub) {
  std::vector<std::pair<int, int>> indices;
  std::vector<T> values;

  indices.reserve(nnz);
  values.reserve(nnz);

  for (size_t i = 0; i < nnz; ++i) {
    indices.emplace_back(std::min(m - 1, static_cast<int>(rand(0, m))),
        std::min(n - 1, static_cast<int>(rand(0, n))));
    values.push_back(rand(lb, ub));
  }

  std::sort(indices.begin(), indices.end());

  int row_ind = 1;
  int col_ind = 1;
  rptr[0] = 0; 
  cind[0] = indices[0].second;
  val[0] = values[0];
  for (size_t i = 1; i < nnz; ++i) {
    if (indices[i-1] == indices[i])
      continue;
    for (size_t j = indices[i-1].first; j < indices[i].first; ++j)
      rptr[row_ind++] = col_ind;
    cind[col_ind] = indices[i].second;
    val[col_ind++] = values[i];
  }

  for (size_t j = indices[nnz-1].first; j < m + 1; ++j)
    rptr[row_ind++] = col_ind;
  return col_ind;
}

#endif  // MAT_GEN_H_

