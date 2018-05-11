/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "../../base/ffm/trainer.h"
#include "batching_gpu.cuh"
#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

namespace ffm {

template<typename T>
Trainer<T>::Trainer(const Dataset<T> &dataset, Model<T> &model, Params const &params)
    : trainDataBatcher(1), model(model), params(params) {
  // TODO delete in destructor
  DatasetBatcherGPU<T> *batcher = new DatasetBatcherGPU<T>(dataset, params);
  trainDataBatcher[0] = batcher;
}

/**
 * Original ffm gradient/weight update method from https://github.com/guestwalk/libffm with slight adjustments
 */
template<typename T>
T wTx(Row<T> *row,
      thrust::device_vector<T> &weights,
      thrust::device_vector<T> &weightsGradient,
      Params params,
      T kappa = 0,
      bool update = false,
      int verbose = 0) {
  log_verbose(params.verbose, "KAPPA = %f", kappa);
  thrust::device_vector < Node<T> * > nodes(row->data);

  size_t alignFeat1 = params.numFields * params.k;
  size_t alignFeat2 = params.k;

  T loss = 0.0;

  T* weightsPtr = thrust::raw_pointer_cast(weights.data());
  T* weightsGradientPtr = thrust::raw_pointer_cast(weightsGradient.data());

  T r = params.normalize ? row->scale : 1.0;

//#pragma omp parallel for schedule(static) reduction(+: loss)
  for (size_t n1 = 0; n1 < row->size; n1++) {
    Node<T> *node1 = nodes[n1];

    loss += thrust::transform_reduce(nodes.begin() + n1 + 1, nodes.end(), [=]__device__(Node<T> * node2) {
      size_t feature1 = node1->featureIdx;
      size_t field1 = node1->fieldIdx;
      T value1 = node1->value;

      size_t feature2 = node2->featureIdx;
      size_t field2 = node2->fieldIdx;
      T value2 = node2->value;
      T localt = 0;

      if (feature1 >= params.numFeatures || field1 >= params.numFields ||
          feature2 >= params.numFeatures || field2 >= params.numFields) {
        return localt;
      }

      size_t idx1 = feature1 * alignFeat1 + field2 * alignFeat2;
      size_t idx2 = feature2 * alignFeat1 + field1 * alignFeat2;
      T *w1 = weightsPtr + idx1;
      T *w2 = weightsPtr + idx2;

      T v = value1 * value2 * r;

      if (update) {
        T *wg1 = weightsGradientPtr + idx1;
        T *wg2 = weightsGradientPtr + idx2;

        for (size_t d = 0; d < params.k; d++) {
          T g1 = params.regLambda * w1[d] + kappa * w2[d] * v;
          T g2 = params.regLambda * w2[d] + kappa * w1[d] * v;

          wg1[d] += g1 * g1;
          wg2[d] += g2 * g2;

          w1[d] -= params.learningRate / sqrt(wg1[d]) * g1;
          w2[d] -= params.learningRate / sqrt(wg2[d]) * g2;
        }
      } else {
        for (size_t d = 0; d < alignFeat2; d++) {
          localt += w1[d] * w2[d] * v;
        }
      }

      return localt;
    },
    (T) 0.0,
        thrust::plus<T>());
  }

  return loss;
}

template<typename T>
void Trainer<T>::predict(T *predictions) {
  for (int i = 0; i < params.nGpus; i++) {
    log_verbose(this->params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->model.weights.size(), i);
    thrust::device_vector<T> localWeights(this->model.weights.begin(), this->model.weights.end());
    thrust::device_vector<T> localGradients(this->model.weights.size());

    int record = 0;
    while (trainDataBatcher[i]->hasNext()) {
      DatasetBatch<T> batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);

      T loss = 0;
      // TODO parallelize somehow
      while (batch.hasNext()) {
        Row<T> *row = batch.nextRow();

        T t = wTx(row, localWeights, localGradients, this->params);

        predictions[record++] = 1 / (1 + exp(-t));
      }
    }
  }
}

template<typename T>
// TODO return loss
T Trainer<T>::oneEpoch(bool update) {
  log_debug(this->params.verbose, "Computing an FFM epoch (update = %s)", update ? "true" : "false");

  T loss = 0;

  std::vector<thrust::device_vector<T>> allLocalWeights(params.nGpus);
  std::vector<thrust::device_vector<T>> allLocalGradients(params.nGpus);
  for (int i = 0; i < params.nGpus; i++) {
    log_verbose(this->params.verbose, "Copying weights of size %zu to GPU %d", this->model.weights.size(), i);

    // TODO do only once for all iterations?
    allLocalWeights[i].resize(this->model.weights.size());
    thrust::copy(this->model.weights.begin(), this->model.weights.end(), allLocalWeights[i].begin() );
    allLocalGradients[i].resize(this->model.gradients.size());
    thrust::copy(this->model.gradients.begin(), this->model.gradients.end(), allLocalGradients[i].begin() );

    while (trainDataBatcher[i]->hasNext()) {
      DatasetBatch<T> batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);

      // TODO parallelize somehow
      // TODO shuffle batch?
      while (batch.hasNext()) {
        Row<T> *row = batch.nextRow();

        T t = wTx(row, allLocalWeights[i], allLocalGradients[i], this->params);

        T expnyt = exp(-row->label * t);
        loss += log(1 + expnyt);

        if (update) {
          T kappa = -row->label * expnyt / (1 + expnyt);
          wTx(row, allLocalWeights[i], allLocalGradients[i], this->params, kappa, true);
        }
      }
    }
    trainDataBatcher[i]->reset();
  }

  if (params.nGpus != 1) {
    // TODO average local weights
    // TODO distribute gradients
  } else {
    thrust::copy(allLocalWeights[0].begin(), allLocalWeights[0].end(), this->model.weights.begin());
    thrust::copy(allLocalGradients[0].begin(), allLocalGradients[0].end(), this->model.gradients.begin());
  }

  log_debug(this->params.verbose, "Log loss = %f", loss / params.numRows);

  return loss / params.numRows;
}

template<typename T>
bool Trainer<T>::earlyStop() {
  // TODO implement
  return false;
}

template
class Trainer<float>;
template
class Trainer<double>;

}