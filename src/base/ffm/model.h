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
  Model(Params &params);

  Model(Params &params, const T *weights);

  virtual void copyTo(T *dstWeights);

  std::vector<T> weights;

  virtual T* weightsRawPtr(int i);

 private:

  int numFeatures;
  int numFields;
  int k;

  bool normalize;
};

}
