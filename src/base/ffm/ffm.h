/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../../include/solver/ffm_api.h"
#include "trainer.h"
#include "model.h"
#include "../../include/data/ffm/data.h"
#include "../../common/logger.h"

namespace ffm {

template <typename T>
class FFM  {
 public:
  FFM(Params & params);
  FFM(Params & params, T *weights);

  int fit(Dataset<T> &dataset, Dataset<T> &valid_dataset);

  void predict(Dataset<T> &dataset, T *predictions);

  bool validate();

  Trainer<T> trainer;

 private:
  Params params;
};

}