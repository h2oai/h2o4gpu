/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef GENERATOR_HPP_
#define GENERATOR_HPP_

#include "KmMatrix.hpp"

namespace h2o4gpu {
namespace Matrix {

template <typename T>
class GeneratorBase {
 public:
  virtual KmMatrix<T> generate() = 0;
  virtual KmMatrix<T> generate(size_t _size) = 0;
};

}  // namespace Matrix
}  // namespace h2o4gpu


#endif  // GENERATOR_HPP_
