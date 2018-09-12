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
FFM<T>::FFM(Params &params) : params(params), trainer(params) {}

template<typename T>
FFM<T>::FFM(Params & params, T *weights) : params(params), trainer(weights, params) {}

template<typename T>
int FFM<T>::fit(Dataset<T> &dataset, Dataset<T> &valid_dataset) {
  this->trainer.setTrainingDataset(dataset);
  this->trainer.setValidationDataset(valid_dataset);

  Timer timer;

  int epoch = 1;
  for (; epoch <= this->params.nIter; epoch++) {
    log_info(this->params.verbose, "Iteration %d/%d.", epoch, this->params.nIter);
    timer.tic();
    double trainingLoss = trainer.trainOneEpoch();

    log_info(this->params.verbose, "Train log loss = %f", trainingLoss);

    if(this->validate()) {
      double validationLoss = trainer.validationLoss();

      log_info(this->params.verbose, "Validation log loss = %f", validationLoss);

      if(this->params.autoStop && validationLoss > this->trainer.model->bestValidationLoss) {
        log_debug(params.verbose, "Validation loss increased, stopping early.");
        break;
      }
      this->trainer.model->bestValidationLoss = validationLoss;
    }

    timer.toc();
    log_debug(params.verbose, "Epoch took %f.", timer.pop());
  }

  return epoch;
}

template<typename T>
void FFM<T>::predict(Dataset<T> &dataset, T *predictions) {
  this->trainer.setTrainingDataset(dataset);
  trainer.predict(predictions);
}

template<typename T>
bool FFM<T>::validate() {
  return this->trainer.hasValidationData();
}

template <typename T>
void computeScales(T *scales, const T* values, const int *rowPositions, Params &_param) {
#pragma omp parallel for
  for(int i = 0; i < _param.numRows; i++) {
    if(_param.normalize) {
      T scale = 0.0;
      for (int e = rowPositions[i]; e < rowPositions[i + 1]; e++) {
        scale += values[e] * values[e];
      }
      scales[i] = 1.0 / scale;
    } else {
      scales[i] = 1.0;
    }
  }
}

template <typename T>
T maxElement(const T *data, int size) {
  T maxVal = 0.0;

#pragma omp parallel for reduction(max : maxVal)
  for (int i = 0; i < size; i++) {
    if (data[i] > maxVal) {
      maxVal = data[i];
    }
  }

  return maxVal;
}

/**
 * C API method
 */

int ffm_fit_float(int *features, int* fields, float* values, int *labels, int *rowPositions,
                   int *features_v, int* fields_v, float* values_v, int *labels_v, int *rowPositions_v,
                   float *w, Params &_param) {
  log_debug(_param.verbose, "Converting %d float rows into a training dataset.", _param.numRows);
  float *scales = (float*) malloc(sizeof(float) * _param.numRows);
  computeScales(scales, values, rowPositions, _param);

  _param.numFields = maxElement(fields, _param.numNodes) + 1;
  _param.numFeatures = maxElement(features, _param.numNodes) + 1;

  Dataset<float> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, labels, scales, rowPositions);

  Dataset<float> *validationDataset = new Dataset<float>();
  if(features_v) {
    log_debug(_param.verbose, "Converting %d float rows into a validation dataset.", _param.numRowsVal);

    float *scales_v = (float*) malloc(sizeof(float) * _param.numRowsVal);
    computeScales(scales_v, values_v, rowPositions_v, _param);

    validationDataset = new Dataset<float>(_param.numFields, _param.numFeatures, _param.numRowsVal, _param.numNodesVal, features_v, fields_v, values_v, labels_v, scales_v, rowPositions_v);
  }

  FFM<float> ffm(_param);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM fit for float.");
  Timer timer;
  timer.tic();
  int iterations = ffm.fit(dataset, *validationDataset);
  ffm.trainer.model->copyTo(w);
  timer.toc();
  log_debug(_param.verbose, "Float fit took %f.", timer.pop());

  delete validationDataset;

  return iterations;
}

int ffm_fit_double(int *features, int* fields, double* values, int *labels, int *rowPositions,
                    int *features_v, int* fields_v, double* values_v, int *labels_v, int *rowPositions_v,
                    double *w, Params &_param) {
  log_debug(_param.verbose, "Converting %d double rows into a dataset.", _param.numRows);
  double *scales = (double*) malloc(sizeof(double) * _param.numRows);
  computeScales(scales, values, rowPositions, _param);

  _param.numFields = maxElement(fields, _param.numNodes) + 1;
  _param.numFeatures = maxElement(features, _param.numNodes) + 1;

  Dataset<double> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, labels, scales, rowPositions);

  Dataset<double> *validationDataset = new Dataset<double>();
  if(features_v) {
    log_debug(_param.verbose, "Converting %d double rows into a validation dataset.", _param.numRowsVal);

    double *scales_v = (double*) malloc(sizeof(double) * _param.numRowsVal);
    computeScales(scales_v, values_v, rowPositions_v, _param);

    validationDataset = new Dataset<double>(_param.numFields, _param.numFeatures, _param.numRowsVal, _param.numNodesVal, features_v, fields_v, values_v, labels_v, scales_v, rowPositions_v);
  }

  FFM<double> ffm(_param);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM fit for double.");
  Timer timer;
  timer.tic();
  int iterations = ffm.fit(dataset, *validationDataset);
  ffm.trainer.model->copyTo(w);
  timer.toc();
  log_debug(_param.verbose, "Double fit took %f.", timer.pop());

  delete validationDataset;

  return iterations;
}

void ffm_predict_float(int *features, int* fields, float* values, int* rowPositions, float *predictions, float *w, Params &_param) {
  log_debug(_param.verbose, "Converting %d float rows into a dataset for predictions.", _param.numRows);
  float *scales = (float*) malloc(sizeof(float) * _param.numRows);
  computeScales(scales, values, rowPositions, _param);

  Dataset<float> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, nullptr, scales, rowPositions);
  FFM<float> ffm(_param, w);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM predict for float.");
  ffm.predict(dataset, predictions);
}

void ffm_predict_double(int *features, int* fields, double* values, int* rowPositions, double *predictions, double *w, Params &_param){
  log_debug(_param.verbose, "Converting %d double rows into a dataset for predictions.", _param.numRows);
  double *scales = (double*) malloc(sizeof(double) * _param.numRows);
  computeScales(scales, values, rowPositions, _param);

  Dataset<double> dataset(_param.numFields, _param.numFeatures, _param.numRows, _param.numNodes, features, fields, values, nullptr, scales, rowPositions);
  FFM<double> ffm(_param, w);
  _param.printParams();
  log_debug(_param.verbose, "Running FFM predict for double.");
  ffm.predict(dataset, predictions);
}

} // namespace ffm