#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <cstdlib>

template <typename T>
double Lasso(int m, int n, int nnz);

template <typename T>
double LassoPath(int m, int n, int nnz);

// template <typename T>
// double Logistic(int m, int n, int nnz);

template <typename T>
double LpEq(int m, int n, int nnz);

// template <typename T>
// double LpIneq(int m, int n, int nnz);
// 
// template <typename T>
// double NonNegL2(int m, int n, int nnz);
// 
// template <typename T>
// double Svm(int m, int n, int nnz);

#endif  // EXAMPLES_H_

