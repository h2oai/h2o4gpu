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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                        __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

namespace ffm {

template<typename T>
Trainer<T>::Trainer(const Dataset<T> &dataset, Model<T> &model, Params &params)
    : trainDataBatcher(1), model(model), params(params) {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

#if __CUDA_ARCH__ > 500
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 0));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 0));
#endif // CUDA 5.0
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0));
  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  // TODO delete in destructor
  DatasetBatcherGPU<T> *batcher = new DatasetBatcherGPU<T>(dataset, params);
  trainDataBatcher[0] = batcher;
}

template<typename T>
Trainer<T>::~Trainer() {
  CUDA_CHECK(cudaDeviceReset());
}

template<typename T>
__global__ void wTxKernel(const size_t *__restrict__ features, const size_t *__restrict__ fields, const T *__restrict__ values,
                          const T *__restrict__ r, const size_t *__restrict__ rowSizes,
                          T *__restrict__ weightsPtr, T *__restrict__ weightsGradientPtr,
                          T *__restrict__ losses, const int maxRowSize, const int rows,
                          const int alignFeat1, const int alignFeat2, const int k, const float regLambda, const float learningRate,
                          const bool update, const int* labels = nullptr) {
  __shared__ T kappa;

  int rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / maxRowSize;

  if(rowIdx > rows) return;

  int rowSize = rowSizes[rowIdx + 1] - rowSizes[rowIdx];

  int nodeIdx = (blockIdx.x * blockDim.x + threadIdx.x) % maxRowSize;

  if(nodeIdx >= rowSize) return;

  int n1 = rowSizes[rowIdx] + nodeIdx;
  T loss = 0.0;
  for(int i = 1; n1 + i < rowSizes[rowIdx + 1]; i++) {
    const int n2 = n1 + i;

    const int feature1 = features[n1];
    const int field1 = fields[n1];
    const T value1 = values[n1];

    const int feature2 = features[n2];
    const int field2 = fields[n2];
    const T value2 = values[n2];

    // TODO check if fields/features are in range of trained model
    const int idx1 = feature1 * alignFeat1 + field2 * alignFeat2;
    const int idx2 = feature2 * alignFeat1 + field1 * alignFeat2;
    T *w1 = weightsPtr + idx1;
    T *w2 = weightsPtr + idx2;

    const T v = value1 * value2 * r[rowIdx];

    if (update) {
      if (threadIdx.x == 0) {
        const int label = labels[rowIdx];
        const T expnyt = exp(-label * losses[rowIdx]);
        kappa = -label * expnyt / (1 + expnyt);
      }

      __syncthreads();

      T *wg1 = weightsGradientPtr + idx1;
      T *wg2 = weightsGradientPtr + idx2;

      for (int d = 0; d < k; d++) {
        const T w1d = w1[d];
        const T w2d = w2[d];
        const T g1 = regLambda * w1d + kappa * w2d * v;
        const T g2 = regLambda * w2d + kappa * w1d * v;

        const T wg1d = wg1[d] + g1 * g1;
        const T wg2d = wg2[d] + g2 * g2;
        wg1[d] = wg1d;
        wg2[d] = wg2d;

        w1[d] = w1d - learningRate / sqrt(wg1d) * g1;
        w2[d] = w2d - learningRate / sqrt(wg2d) * g2;
      }
    } else {
      for (int d = 0; d < alignFeat2; d++) {
        loss += w1[d] * w2[d] * v;
      }
    }
  }
  atomicAdd(losses + rowIdx, loss);
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

      size_t threads = 512; // TODO based on row number??
      size_t maxRowSize = batch->widestRow();
      size_t totalThreads = batch->numRows * maxRowSize;
      size_t blocks = std::ceil(totalThreads / threads);

      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      T* weightsPtr = thrust::raw_pointer_cast(localWeights.data());
      T* weightsGradientPtr = thrust::raw_pointer_cast(localGradients.data());

      size_t alignFeat1 = params.numFields * params.k;
      size_t alignFeat2 = params.k;

      wTxKernel << < blocks, threads >> > (batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
          weightsPtr, weightsGradientPtr, losses, maxRowSize, batch->numRows, alignFeat1, alignFeat2, params.k, params.regLambda, params.learningRate, false);

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

      size_t threads = 512; // TODO based on row number??
      size_t maxRowSize = batch->widestRow();
      size_t totalThreads = batch->numRows * maxRowSize;
      size_t blocks = std::ceil((double)totalThreads / threads);

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
          weightsPtr, weightsGradientPtr, losses, maxRowSize, batch->numRows, alignFeat1, alignFeat2, params.k, params.regLambda, params.learningRate, false);

      if(update) {
        wTxKernel<<<blocks, threads>>>(batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
            weightsPtr, weightsGradientPtr, losses, maxRowSize, batch->numRows, alignFeat1, alignFeat2, params.k, params.regLambda, params.learningRate, true, batch->labels);
      }

      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      timer.toc();
      log_verbose(params.verbose, "wTx took %f.", timer.pop());

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