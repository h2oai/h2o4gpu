#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <cstdlib>

template <typename T>
T Lasso(size_t m, size_t n);

template <typename T>
T LpEq(size_t m, size_t n);

template <typename T>
T LpIneq(size_t m, size_t n);

template <typename T>
T NonNegL2(size_t m, size_t n);

template <typename T>
T Svm(size_t m, size_t n);

#endif  // EXAMPLES_H_

