#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <cstdlib>

template <typename T>
double Lasso(size_t m, size_t n);

template <typename T>
double LpEq(size_t m, size_t n);

template <typename T>
double LpIneq(size_t m, size_t n);

template <typename T>
double NonNegL2(size_t m, size_t n);

template <typename T>
double Svm(size_t m, size_t n);

#endif  // EXAMPLES_H_

