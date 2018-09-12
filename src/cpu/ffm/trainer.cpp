/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "../../base/ffm/trainer.h"

namespace ffm {

template<typename T>
Trainer<T>::Trainer(Params &params) : params(params), trainDataBatcher(1), validationDataBatcher(1) {
  this->model = new Model<T>(params);
}

template<typename T>
Trainer<T>::Trainer(const T* weights, Params &params) : params(params),
                                                        trainDataBatcher(1),
                                                        validationDataBatcher(1) {
  this->model = new Model<T>(params, weights);
}

template<typename T>
void Trainer<T>::setTrainingDataset(Dataset<T> &dataset) {
  DatasetBatcher<T> *batcher = new DatasetBatcher<T>(dataset, params, params.numRows);
  trainDataBatcher[0] = batcher;
}

template<typename T>
void Trainer<T>::setValidationDataset(Dataset<T> &dataset) {
  DatasetBatcher<T> *batcher = new DatasetBatcher<T>(dataset, params, params.numRowsVal);
  validationDataBatcher[0] = batcher;
}

template<typename T>
Trainer<T>::~Trainer() {
  delete trainDataBatcher[0];
  delete validationDataBatcher[0];
  delete model;
}

// Original code at https://github.com/guestwalk/libffm
template<typename T>
inline double wTx(
    int start, int end,
    const int *__restrict__ features, const int *__restrict__ fields, const T *__restrict__ values, const T r,
    T *__restrict__ weights,
    int kALIGN, int k, int numFields, float lambda, float eta, T kappa, const bool update) {
  int align0 = kALIGN * k;
  int align1 = numFields * align0;

  double t = 0;
  for (int n1 = start; n1 < end; n1++) {
    int j1 = features[n1];
    int f1 = fields[n1];
    T v1 = values[n1];

    for (int n2 = n1 + 1; n2 < end; n2++) {
      int j2 = features[n2];
      int f2 = fields[n2];
      T v2 = values[n2];

      int idx1 = (int) j1 * align1 + f2 * align0;
      int idx2 = (int) j2 * align1 + f1 * align0;
      T *w1 = weights + idx1;
      T *w2 = weights + idx2;

      double v = v1 * v2 * r;

      if (update) {
        T *wg1 = w1 + 1;
        T *wg2 = w2 + 1;

        for (int d = 0; d < align0; d += kALIGN) {
          T g1 = lambda * w1[d] + kappa * w2[d] * v;
          T g2 = lambda * w2[d] + kappa * w1[d] * v;

          wg1[d] += g1 * g1;
          wg2[d] += g2 * g2;

          w1[d] -= eta / std::sqrt(wg1[d]) * g1;
          w2[d] -= eta / std::sqrt(wg2[d]) * g2;

        }
      } else {
        for (int d = 0; d < align0; d += kALIGN) {
          t += w1[d] * w2[d] * v;
        }
      }
    }
  }

  return t;
}

template<typename T>
void Trainer<T>::predict(T *predictions) {
  double loss = 0;

  while (this->trainDataBatcher[0]->hasNext()) {
    DatasetBatch<T> *batch = this->trainDataBatcher[0]->nextBatch(this->params.batchSize);

    T *weightsPtr = this->model->weights.data();

    // Used to cast from global indexing to batch local indexing of row positions
    int offset = batch->rowPositions[0];
#pragma omp parallel for
    for(int row = 0; row < batch->numRows; row++) {
      T t = wTx(
          batch->rowPositions[row] - offset, batch->rowPositions[row + 1] - offset,
          batch->features, batch->fields, batch->values, batch->scales[row],
          weightsPtr, 1, params.k, params.numFields, params.regLambda, params.learningRate, (T)0, false);

      predictions[row] = 1.0 / (1.0 + std::exp(-t));

    }
    delete batch;
  }

  this->trainDataBatcher[0]->reset();
}

template<typename T>
double Trainer<T>::validationLoss() {
  return this->oneEpoch(this->validationDataBatcher, false);
}

template<typename T>
double Trainer<T>::trainOneEpoch() {
  return this->oneEpoch(this->trainDataBatcher, true);
}

template<typename T>
double Trainer<T>::oneEpoch(std::vector<DatasetBatcher<T>*> dataBatcher, bool update) {
  log_debug(this->params.verbose, "Computing an FFM epoch (update = %s)", update ? "true" : "false");

  double loss = 0;

  while (dataBatcher[0]->hasNext()) {
    DatasetBatch<T> *batch = dataBatcher[0]->nextBatch(this->params.batchSize);

    T *weightsPtr = this->model->weights.data();

    // Used to cast from global indexing to batch local indexing of row positions
    int offset = batch->rowPositions[0];
#pragma omp parallel for schedule(static) reduction(+: loss)
    for(int row = 0; row < batch->numRows; row++) {
      double t = wTx(
          batch->rowPositions[row] - offset, batch->rowPositions[row + 1] - offset,
          batch->features, batch->fields, batch->values, batch->scales[row],
          weightsPtr, 2, params.k, params.numFields, params.regLambda, params.learningRate, (T)0, false);

      int y = batch->labels[row];

      T expnyt = std::exp(-y*t);

      loss += std::log(1+expnyt);

      if (update) {
        T kappa = -y * expnyt/(1+expnyt);

        wTx(batch->rowPositions[row] - offset, batch->rowPositions[row + 1] - offset,
                  batch->features, batch->fields, batch->values, batch->scales[row],
                  weightsPtr, 2, params.k, params.numFields, params.regLambda, params.learningRate, kappa, true);
      }
    }
    delete batch;
  }

  dataBatcher[0]->reset();

  return loss / params.numRows;
}

template
class Trainer<float>;
template
class Trainer<double>;

}