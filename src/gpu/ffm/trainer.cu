/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include "batching_gpu.cuh"
#include "../utils/cuda.h"
#include "../../base/ffm/trainer.h"
#include "../../common/timer.h"

namespace ffm {

template<typename T>
Trainer<T>::Trainer(const Dataset<T> &dataset, Model<T> &model, Params &params)
    : trainDataBatcher(1), model(model), params(params) {
  // TODO delete in destructor
  DatasetBatcherGPU<T> *batcher = new DatasetBatcherGPU<T>(dataset, params);
  trainDataBatcher[0] = batcher;
}

template<typename T>
__global__ void wTxKernel(const size_t *features, const size_t *fields, const T *values, const T *r, const size_t *rowSizes,
                          T *weightsPtr, T *weightsGradientPtr, T *losses,
                          int alignFeat1, int alignFeat2, int k, float regLambda, float learningRate,
                          bool update, const int* labels = nullptr) {
  // TODO might need size_t
  int rowIdx = blockIdx.x;
  int rowOffset = rowSizes[rowIdx];
  int rowSize = rowSizes[rowIdx + 1] - rowSizes[rowIdx];

  if(threadIdx.x >= (1.0 + rowSize - 1.0)/2.0 * (rowSize-1)) {
    return;
  }

  // TODO might need size_t
  int n1 = rowOffset; // Node1 index
  int acc = 0;
  int currSize = rowSize - 1;
  int pos = currSize;
  while(threadIdx.x >= pos) {
    acc += currSize;
    currSize--;
    pos += currSize;
    n1++;
  }

  int n2 = threadIdx.x - acc + n1 + 1;

  int feature1 = features[n1];
  int field1 = fields[n1];
  T value1 = values[n1];

  int feature2 = features[n2];
  int field2 = fields[n2];
  T value2 = values[n2];
  T localt = 0;

  // TODO check if fields/features are in range of trained model

  int idx1 = feature1 * alignFeat1 + field2 * alignFeat2;
  int idx2 = feature2 * alignFeat1 + field1 * alignFeat2;
  T *w1 = weightsPtr + idx1;
  T *w2 = weightsPtr + idx2;

  T v = value1 * value2 * r[rowIdx];


  if (update) {
    // TODO shared in a block?
    T expnyt = exp(-labels[rowIdx] * losses[rowIdx]);
    T kappa = -labels[rowIdx] * expnyt / (1 + expnyt);

    T *wg1 = weightsGradientPtr + idx1;
    T *wg2 = weightsGradientPtr + idx2;

    for (int d = 0; d < k; d++) {
      T g1 = regLambda * w1[d] + kappa * w2[d] * v;
      T g2 = regLambda * w2[d] + kappa * w1[d] * v;

      wg1[d] += g1 * g1;
      wg2[d] += g2 * g2;

      w1[d] -= learningRate / sqrt(wg1[d]) * g1;
      w2[d] -= learningRate / sqrt(wg2[d]) * g2;
    }
  } else {
    for (int d = 0; d < alignFeat2; d++) {
      localt += w1[d] * w2[d] * v;
    }
    losses[rowIdx] = localt;
  }
}

template<typename T>
void Trainer<T>::predict(T *predictions) {
  for (int i = 0; i < params.nGpus; i++) {
    log_verbose(this->params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->model.weights.size(), i);
    thrust::device_vector<T> localWeights(this->model.weights.begin(), this->model.weights.end());
    thrust::device_vector<T> localGradients(this->model.weights.size());

    int record = 0;
    while (trainDataBatcher[i]->hasNext()) {
      DatasetBatch<T> *batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);

      size_t maxCalcs = batch->widestRow();
      size_t blocks = batch->numRows;
      size_t threads = maxCalcs;

      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      T* weightsPtr = thrust::raw_pointer_cast(localWeights.data());
      T* weightsGradientPtr = thrust::raw_pointer_cast(localGradients.data());

      size_t alignFeat1 = params.numFields * params.k;
      size_t alignFeat2 = params.k;

      wTxKernel << < blocks, threads >> > (batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
          weightsPtr, weightsGradientPtr, losses, alignFeat1, alignFeat2, params.k, params.regLambda, params.learningRate, false);

      thrust::copy(losses + record, losses + record + batch->numRows, predictions + record);


      record += batch->numRows;
    }

    std::transform (predictions, predictions + params.numRows, predictions, [&](T val){ return 1.0 / (1.0 + exp(-val)); });
  }
}

template<typename T>
T Trainer<T>::oneEpoch(bool update) {
  Timer timer;
  log_debug(this->params.verbose, "Computing an FFM epoch (update = %s)", update ? "true" : "false");

  T loss = 0;

  std::vector<thrust::device_vector<T>> allLocalWeights(params.nGpus);
  std::vector<thrust::device_vector<T>> allLocalGradients(params.nGpus);
  for (int i = 0; i < params.nGpus; i++) {
    log_verbose(this->params.verbose, "Copying weights of size %zu to GPU %d", this->model.weights.size(), i);

    /**
     * Copy Weights
     * */
    timer.tic();
    // TODO do only once for all iterations?
    allLocalWeights[i].resize(this->model.weights.size());
    thrust::copy(this->model.weights.begin(), this->model.weights.end(), allLocalWeights[i].begin() );

    allLocalGradients[i].resize(this->model.gradients.size());
    thrust::copy(this->model.gradients.begin(), this->model.gradients.end(), allLocalGradients[i].begin() );
    timer.toc();
    log_verbose(params.verbose, "Copying weights took %f.", timer.pop());

    while (trainDataBatcher[i]->hasNext()) {
      /**
       * Get batch
       */

      timer.tic();
      DatasetBatch<T> *batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);
timer.toc();
      log_verbose(params.verbose, "Getting batch took %f.", timer.pop());

      size_t alignFeat1 = params.numFields * params.k;
      size_t alignFeat2 = params.k;

      size_t maxCalcs = batch->widestRow();
      size_t threads = maxCalcs;
      size_t blocks = batch->numRows;

      /**
        * Alloc
        */
      // todo dealloc
      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      T* weightsPtr = thrust::raw_pointer_cast(allLocalWeights[i].data());
      T* weightsGradientPtr = thrust::raw_pointer_cast(allLocalGradients[i].data());

      timer.tic();
      wTxKernel<<<blocks, threads>>>(batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
          weightsPtr, weightsGradientPtr, losses, alignFeat1, alignFeat2, params.k, params.regLambda, params.learningRate, false);
      timer.toc();
      log_verbose(params.verbose, "wTx (update false) took %f.", timer.pop());

      timer.tic();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      timer.toc();
      log_verbose(params.verbose, "Cuda synchronize took %f.", timer.pop());

      if(update) {
        timer.tic();
        wTxKernel<<<blocks, threads>>>(batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
            weightsPtr, weightsGradientPtr, losses, alignFeat1, alignFeat2, params.k, params.regLambda, params.learningRate, true, batch->labels);
        timer.toc();
        log_verbose(params.verbose, "wTx (update true) took %f.", timer.pop());

      }

      timer.tic();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      timer.toc();
      log_verbose(params.verbose, "Cuda synchronize took %f.", timer.pop());

      timer.tic();
      int* labels = batch->labels;
      thrust::counting_iterator<int> counter(0);
      loss += thrust::transform_reduce(counter, counter + batch->numRows , [=]__device__(int idx) {
        return log(1 + exp(-labels[idx] * losses[idx]));
      },
      (T) 0.0, thrust::plus<T>());
      timer.toc();

      log_verbose(params.verbose, "Loss compute took %f.", timer.pop());

    }
    trainDataBatcher[i]->reset();
  }

  if (params.nGpus != 1) {
    // TODO average local weights
    // TODO distribute gradients
  } else {
    timer.tic();
    thrust::copy(allLocalWeights[0].begin(), allLocalWeights[0].end(), this->model.weights.begin());
    thrust::copy(allLocalGradients[0].begin(), allLocalGradients[0].end(), this->model.gradients.begin());
    timer.toc();
    log_verbose(params.verbose, "Copying weights back took %f.", timer.pop());

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