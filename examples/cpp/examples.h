#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <cstdlib>

template <typename T>
double Lasso(const std::vector<T>&A, const std::vector<T>&b);

template <typename T>
double LassoPath(const std::vector<T>&A, const std::vector<T>&b);

template <typename T>
double ElasticNet(const std::vector<T>&A, const std::vector<T>&b, const std::vector<T>&w, int, int, int, int, int, int, int, int, double);

template <typename T>
double Logistic(size_t m, size_t n);

template <typename T>
double LpEq(size_t m, size_t n);

template <typename T>
double LpIneq(size_t m, size_t n);

template <typename T>
double NonNegL2(size_t m, size_t n);

template <typename T>
double Svm(size_t m, size_t n);

#endif  // EXAMPLES_H_

