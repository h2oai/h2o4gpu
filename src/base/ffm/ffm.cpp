/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "ffm.h"
#include <numeric>
#include <algorithm>

namespace ffm {

template<typename T>
FFM<T>::FFM(Params const &params) : params(params), model(params) {}

template<typename T>
void FFM<T>::fit(const Dataset<T> &dataset) {
  Trainer<T> trainer(dataset, this->model, this->params);

  for (int epoch = 1; epoch <= this->params.nIter; epoch++) {
    trainer.oneEpoch(true);
    if (trainer.earlyStop()) {
      break;
    }
  }
}

template<typename T>
Dataset<T> &rowsToDataset(Row<T> *rows, Params &params) {
  size_t numRows = params.numRows;
  int *labels = new int[numRows];
  T *scales = new T[numRows];

  size_t totalNumNodes = 0;
  for (size_t i = 0; i < numRows; i++) {
    totalNumNodes += rows[i].size;
  }

  size_t numFields = 0;
  size_t numFeatures = 0;

  size_t position = 0;
  for (int i = 0; i < numRows; i++) {
    Row<T> row = rows[i];

    T scale = 0.0;

    for (int i = 0; i < row.size; i++) {
      numFeatures = std::max(numFeatures, row.data[i]->featureIdx + 1);
      numFields = std::max(numFields, row.data[i]->fieldIdx + 1);

      scale += row.data[i]->value * row.data[i]->value;
      position++;
    }

    row.label = row.label > 0 ? 1 : -1;
    row.scale = 1.0 / scale;
  }

  std::vector<Row<T>*> rowVec(numRows);
  for(int r = 0; r < numRows; r++) {
    rowVec[r] = &rows[r];
  }

  params.numFeatures = numFeatures;
  params.numFields = numFields;

  // TODO needs to be deleted!
  Dataset<T> *dataset = new Dataset<T>(numFields, numFeatures, numRows, totalNumNodes, rowVec);
  return *dataset;
}

/**
 * C API method
 */
void ffm_float(Row<float> *rows, float *w, Params _param) {
  log_debug(_param.verbose, "Converting %d float rows into a dataset.", _param.numRows);
  Dataset<float> dataset = rowsToDataset(rows, _param);
  FFM<float> ffm(_param);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM fit for float.");
  ffm.fit(dataset);
  ffm.model.copyTo(w);
}

void ffm_double(Row<double> *rows, double *w, Params _param) {
  log_debug(_param.verbose, "Converting %d double rows into a dataset.", _param.numRows);
  Dataset<double> dataset = rowsToDataset(rows, _param);
  FFM<double> ffm(_param);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM fit for double.");
  ffm.fit(dataset);
  ffm.model.copyTo(w);
}

} // namespace ffm