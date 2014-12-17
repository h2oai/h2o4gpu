#ifndef MAT_GEN_H_
#define MAT_GEN_H_

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <vector>

using std::get;

template <typename T>
inline T rand(T lb, T ub) {
  return std::min(static_cast<T>(ub - lb - 1),
      static_cast<T>((ub - lb) * (rand() / (1.0 * RAND_MAX)))) + lb;
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
                 T ub, const std::vector<std::tuple<int, int, T>> &entries) {
  std::vector<std::tuple<int, int, unsigned char, T>> values;

  values.reserve(nnz);

  for (const auto &e : entries) {
    values.push_back(std::make_tuple(get<0>(e), get<1>(e),
        static_cast<unsigned char>(0), get<2>(e)));
  }

  for (size_t i = entries.size(); i < nnz; ++i) {
    values.push_back(std::make_tuple(rand(0, m),
        rand(0, n), static_cast<unsigned char>(1), rand(lb, ub)));
  }

  std::sort(values.begin(), values.end());

  int index = 1;
  cind[0] = get<1>(values[0]);
  val[0] = get<3>(values[0]);
  rptr[0] = 0;
  for (size_t j = 1; j <= get<0>(values[0]); ++j)
    rptr[j] = 0;
  for (size_t i = 1; i < nnz; ++i) {
    if (get<0>(values[i-1]) == get<0>(values[i]) &&
        get<1>(values[i-1]) == get<1>(values[i]))
      continue;
    for (size_t j = get<0>(values[i-1]) + 1; j <= get<0>(values[i]); ++j)
      rptr[j] = index;
    cind[index] = get<1>(values[i]);
    val[index] = get<3>(values[i]);
    index++;
  }

  for (size_t j = get<0>(values[nnz-1]) + 1; j <= m; ++j)
    rptr[j] = index;
  printf("ci %d\n", index);
  return index;
}

#endif  // MAT_GEN_H_

