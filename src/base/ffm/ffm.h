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
  FFM(Params const & params);
  FFM(Params const & params, T *weights);

  Model<T> model;

  void fit(const Dataset<T> &dataset);

  void predict(const Dataset<T> &dataset, T *predictions);

 private:
  const Params params;
};

}