/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "model.h"
#include "batching.h"
#include "../../include/data/ffm/data.h"
#include <vector>
#include <algorithm>

namespace ffm {

template<typename T>
class Trainer {
 public:
  Trainer(Params &params);
  Trainer(const T *weights, Params &params);
  ~Trainer();

  void setTrainingDataset(Dataset<T> &dataset);
  void setValidationDataset(Dataset<T> &dataset);

  double validationLoss();

  double trainOneEpoch();

  double oneEpoch(std::vector<DatasetBatcher<T> *> dataBatcher, bool update);

  void predict(T *predictions);

  bool hasValidationData() {
    return std::any_of(this->validationDataBatcher.cbegin(), this->validationDataBatcher.cend(),
                       [](DatasetBatcher<T> *batcher){ return !batcher->empty(); });
  }

  // Global model for this machine
  Model<T> *model;

 private:
  Params &params;

  // Vector of train datasets splits for threads/GPUs
  std::vector<DatasetBatcher<T> *> trainDataBatcher;

  // Vector of validation datasets split for threads/GPUs
  std::vector<DatasetBatcher<T> *> validationDataBatcher;

};

}