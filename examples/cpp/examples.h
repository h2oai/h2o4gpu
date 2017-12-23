#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <cstdlib>
#include <vector>

template <typename T>
double ElasticNet(const std::vector<T>&A, const std::vector<T>&b, const std::vector<T>&w, int, int, int, int, int, int, int, int, double);

#endif  // EXAMPLES_H_

