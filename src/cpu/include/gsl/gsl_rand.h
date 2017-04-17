#ifndef GSL_RAND_CUH_
#define GSL_RAND_CUH_

#include <random>

namespace gsl {

template <typename T>
void rand(T *x, size_t size) {
  std::default_random_engine generator;
  std::uniform_real_distribution<T> dist(static_cast<T>(0), static_cast<T>(1));
  
  for (size_t i = 0; i < size; ++i){
    x[i] = dist(generator);
  }
}

}  // namespace gsl

#endif  // GSL_RAND_CUH_

