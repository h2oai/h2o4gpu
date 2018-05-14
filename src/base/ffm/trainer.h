/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "model.h"
#include "batching.h"
#include "../../include/data/ffm/data.h"
#include <vector>

namespace ffm {

template<typename T>
class Trainer {
 public:
  Trainer(const Dataset<T> &dataset, Model<T> &model, Params &params);

  T oneEpoch(bool update);

  void predict(T *predictions);

  bool earlyStop();

  // Global model for this machine
  Model<T> &model;

 private:
  Params &params;

  // Vector of datasets split for threads/GPUs
  std::vector<DatasetBatcher<T>*> trainDataBatcher;

};

}