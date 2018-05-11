/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../../include/solver/ffm_api.h"
#include <vector>

namespace ffm {

template<typename T>
class Model {
 public:
  Model() {}

  Model(Params const &params);

  Model(Params const &params, T *weights);

  void copyTo(T *dstWeights);

  std::vector<T> weights;
  std::vector<T> gradients;

 private:

  int numFeatures;
  int numFields;
  int k;

  bool normalize;
};

}
