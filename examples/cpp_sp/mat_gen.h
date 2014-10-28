#ifndef MAT_GEN_H_
#define MAT_GEN_H_

#include <stdlib.h>
#include <cmath>

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
        val[num] = (ub - lb) * (rand() / kRandMax) + lb;
        cind[num] = j;
        num++;
      }
    }
  }
  rptr[m] = nnz;
}

#endif  // MAT_GEN_H_

