/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "ffm.h"
#include "../../common/timer.h"
#include <numeric>
#include <algorithm>

namespace ffm {

template<typename T>
FFM<T>::FFM(Params &params) : params(params), model(params) {}

template<typename T>
FFM<T>::FFM(Params & params, T *weights) : params(params), model(params, weights) {}

template<typename T>
void FFM<T>::fit(const Dataset<T> &dataset) {
  Trainer<T> trainer(dataset, this->model, this->params);

  Timer timer;

  for (int epoch = 1; epoch <= this->params.nIter; epoch++) {
    timer.tic();
    trainer.oneEpoch(true);
    timer.toc();
    log_debug(params.verbose, "Epoch took %f.", timer.pop());
    if (trainer.earlyStop()) {
      break;
    }
  }
}

template<typename T>
void FFM<T>::predict(const Dataset<T> &dataset, T *predictions) {
  Trainer<T> trainer(dataset, this->model, this->params);
  trainer.predict(predictions);
}

/**
 * C API method
 */
void ffm_fit_float(size_t* features, size_t* fields, float* values, int *labels, float *scales, size_t *rowPositions, float *w, Params &_param) {
  log_debug(_param.verbose, "Converting %d float rows into a dataset.", _param.numRows);
  Dataset<float> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, labels, scales, rowPositions);
  FFM<float> ffm(_param);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM fit for float.");
  Timer timer;
  timer.tic();
  ffm.fit(dataset);
  timer.toc();
  log_debug(_param.verbose, "Float fit took %f.", timer.pop());
  ffm.model.copyTo(w);
}

void ffm_fit_double(size_t* features, size_t* fields, double* values, int *labels, double *scales, size_t *rowPositions, double *w, Params &_param) {
  log_debug(_param.verbose, "Converting %d double rows into a dataset.", _param.numRows);
  Dataset<double> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, labels, scales, rowPositions);
  FFM<double> ffm(_param);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM fit for double.");
  ffm.fit(dataset);
  ffm.model.copyTo(w);
}

void ffm_predict_float(size_t *features, size_t* fields, float* values, float *scales, size_t* rowPositions, float *predictions, float *w, Params &_param) {
  log_debug(_param.verbose, "Converting %d float rows into a dataset for predictions.", _param.numRows);
  Dataset<float> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, nullptr, scales, rowPositions);
  FFM<float> ffm(_param, w);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM predict for float.");
  ffm.predict(dataset, predictions);
}

void ffm_predict_double(size_t *features, size_t* fields, double* values, double *scales, size_t* rowPositions, double *predictions, double *w, Params &_param){
  log_debug(_param.verbose, "Converting %d double rows into a dataset for predictions.", _param.numRows);
  Dataset<double> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, nullptr, scales, rowPositions);
  FFM<double> ffm(_param, w);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM predict for double.");
  ffm.predict(dataset, predictions);
}

} // namespace ffm