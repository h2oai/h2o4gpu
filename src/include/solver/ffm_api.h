/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../data/ffm/data.h"
#include "../../common/logger.h"

namespace ffm {

typedef struct Params {
  int verbose = 0;

  float learningRate = 0.2;
  float regLambda = 0.00002;

  int nIter = 10;
  int batchSize = 1000;

  size_t numRows = 0;
  size_t numFeatures = 0;
  size_t numFields = 0;
  int k = 4;

  bool normalize = true;
  bool autoStop = false;

  // For CPU number of threads to be used
  // For GPU number of GPUs to be used
  int nGpus = 1;

  void printParams() {
    log_verbose(verbose, "learningRate = %f \n regLambda = %f \n nIter = %d \n batchSize = %d \n numRows = %d \n numFeatures = %d \n numFields = %d \n k = %d", learningRate, regLambda, nIter, batchSize, numRows, numFeatures, numFields, k);
  }

} Params;

void ffm_fit_float(Row<float> *rows, float *w, Params _param);
void ffm_fit_double(Row<double> *rows, double *w, Params _param);

}
