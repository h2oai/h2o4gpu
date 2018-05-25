/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "batching_gpu.cuh"
#include "../utils/cuda.h"
#include "../../base/ffm/trainer.h"
#include "../../common/timer.h"
#include "model_gpu.cuh"

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

#define MAX_BLOCK_THREADS 128

namespace ffm {

template<typename T>
Trainer<T>::Trainer(Params &params) : params(params), trainDataBatcher(params.nGpus) {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

#if __CUDA_ARCH__ > 500
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 0));
#endif // CUDA 5.0
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0));
  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  this->model = new ModelGPU<T>(params);
}

template<typename T>
Trainer<T>::Trainer(const T* weights, Params &params) : params(params), trainDataBatcher(params.nGpus) {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

#if __CUDA_ARCH__ > 500
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 0));
#endif // CUDA 5.0
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0));
  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  this->model = new ModelGPU<T>(params, weights);
}

template<typename T>
void Trainer<T>::setDataset(const Dataset<T> &dataset) {
  DatasetBatcherGPU<T> *batcher = new DatasetBatcherGPU<T>(dataset, params);
  trainDataBatcher[0] = batcher;
}

template<typename T>
Trainer<T>::~Trainer() {
  delete trainDataBatcher[0];
  delete model;
  CUDA_CHECK(cudaDeviceReset());
}

__constant__ int cK[1];
__constant__ int cMaxRowSize[1];
__constant__ int cRows[1];
__constant__ int cAlignFeat[2];
__constant__ float cRegLambda[1];
__constant__ float cLearningRate[1];
__constant__ int cWeightsOffset[1];
__constant__ int cBatchOffset[1];


template<typename T>
__global__ void wTxKernel(const int *__restrict__ features, const int *__restrict__ fields, const T *__restrict__ values,
                          const T *__restrict__ r, const int *__restrict__ rowSizes,
                          T *__restrict__ weightsPtr, T *__restrict__ losses, const bool update, const int *__restrict__ labels = nullptr) {
  int rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / cMaxRowSize[0];

  if(rowIdx >= cRows[0]) return;

  int rowSize = rowSizes[rowIdx + 1] - rowSizes[rowIdx];

  int nodeIdx = (blockIdx.x * blockDim.x + threadIdx.x) % cMaxRowSize[0];

  if(nodeIdx >= rowSize) return;

  __shared__ int fieldFeature[MAX_BLOCK_THREADS * 2];
  __shared__ T vals[MAX_BLOCK_THREADS];
  __shared__ T scales[MAX_BLOCK_THREADS];

  __shared__ T kappas[MAX_BLOCK_THREADS];
  __shared__ T expnyts[MAX_BLOCK_THREADS];

  int n1 = rowSizes[rowIdx] + nodeIdx - cBatchOffset[0];

  fieldFeature[threadIdx.x * 2] = fields[n1];
  fieldFeature[threadIdx.x * 2 + 1] = features[n1];
  vals[threadIdx.x] = values[n1];
  scales[rowIdx % MAX_BLOCK_THREADS] = r[rowIdx];

  if(update) {
    expnyts[rowIdx % MAX_BLOCK_THREADS] = std::exp(-labels[rowIdx] * losses[rowIdx]);
    kappas[rowIdx % MAX_BLOCK_THREADS] = -labels[rowIdx] * expnyts[rowIdx % MAX_BLOCK_THREADS] / (1 + expnyts[rowIdx % MAX_BLOCK_THREADS]);
  }

  __syncthreads();

  T loss = 0.0;

  for(int i = 1; n1 + i < rowSizes[rowIdx + 1] - cBatchOffset[0]; i++) {
    // We cache some of the field:feature:values in shared memory, only as many "nodes" as there are threads
    // so we know for 100% the initial node will be cached (since we run 1 thread per each starting node)
    // but the subsequent nodes can be within the same block or they can spill to consequitive blocks
    // depending on the size of the row and number of threads in a block
    const int idx1 = fieldFeature[threadIdx.x * 2 + 1] * cAlignFeat[0] +
        (threadIdx.x + i < MAX_BLOCK_THREADS ? fieldFeature[(threadIdx.x + i) * 2] : fields[n1 + i]) * cAlignFeat[1];

    const int idx2 = (threadIdx.x + i < MAX_BLOCK_THREADS ? fieldFeature[(threadIdx.x + i) * 2 + 1] : features[n1 + i]) * cAlignFeat[0] +
        fieldFeature[threadIdx.x * 2] * cAlignFeat[1];

    const T v = vals[threadIdx.x] * (threadIdx.x + i < MAX_BLOCK_THREADS ? vals[threadIdx.x + i] : values[n1 + i]) * scales[rowIdx % MAX_BLOCK_THREADS];

    if (update) {
      for (int d = 0; d < cK[0] * cWeightsOffset[0]; d+=cWeightsOffset[0]) {
        T w1d = (weightsPtr + idx1)[d];
        T w2d = (weightsPtr + idx2)[d];
        const T g1 = cRegLambda[0] * w1d + kappas[rowIdx % MAX_BLOCK_THREADS] * w2d * v;
        const T g2 = cRegLambda[0] * w2d + kappas[rowIdx % MAX_BLOCK_THREADS] * w1d * v;

        const T w1gdup = (weightsPtr + idx1)[d+1] + g1 * g1;
        const T w2gdup = (weightsPtr + idx2)[d+1] + g2 * g2;

        (weightsPtr + idx1)[d] -= cLearningRate[0] / std::sqrt(w1gdup) * g1;
        (weightsPtr + idx2)[d] -= cLearningRate[0] / std::sqrt(w2gdup) * g2;

        (weightsPtr + idx1)[d+1] = w1gdup;
        (weightsPtr + idx2)[d+1] = w2gdup;

      }
    } else {
      for (int d = 0; d < cK[0] * cWeightsOffset[0]; d+=cWeightsOffset[0]) {
        loss += (weightsPtr + idx1)[d] * (weightsPtr + idx2)[d] * v;
      }
    }
  }
  if(!update) {
    atomicAdd(losses + rowIdx, loss);
  }
}

template<typename T>
void Trainer<T>::predict(T *predictions) {
  for (int i = 0; i < params.nGpus; i++) {

    int initialBatchOffset = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(cBatchOffset, &initialBatchOffset, sizeof(int)));
    int record = 0;
    while (trainDataBatcher[i]->hasNext()) {
      DatasetBatch<T> *batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);

      // TODO once per predictions and share
      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      int alignFeat1 = params.numFields * params.k;
      int alignFeat2 = params.k;

      int threads = MAX_BLOCK_THREADS;
      int maxRowSize = batch->widestRow();
      size_t totalThreads = batch->numRows * maxRowSize;
      int blocks = std::ceil((double)totalThreads / threads);

      T* weightsPtr = this->model->weightsRawPtr(i);

      CUDA_CHECK(cudaMemcpyToSymbol(cK, &params.k, sizeof(int)));
      CUDA_CHECK(cudaMemcpyToSymbol(cMaxRowSize, &maxRowSize, sizeof(int)));
      CUDA_CHECK(cudaMemcpyToSymbol(cRows, &batch->numRows, sizeof(int)));
      int alignTmp[2] = { alignFeat1, alignFeat2 };
      CUDA_CHECK(cudaMemcpyToSymbol(cAlignFeat, &alignTmp, 2 * sizeof(int)));
      CUDA_CHECK(cudaMemcpyToSymbol(cRegLambda, &params.regLambda, sizeof(float)));
      CUDA_CHECK(cudaMemcpyToSymbol(cLearningRate, &params.learningRate, sizeof(float)));
      int offset = 1;
      CUDA_CHECK(cudaMemcpyToSymbol(cWeightsOffset, &offset, sizeof(int)));

      wTxKernel << < blocks, threads >> > (batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
          weightsPtr, losses, false);

      CUDA_CHECK(cudaMemcpy(predictions + record, losses, batch->numRows * sizeof(T), cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());

      record += batch->numRows;

      thrust::fill(thrust::device, losses, losses + batch->numRows, 0.0);

      CUDA_CHECK(cudaMemcpyToSymbol(cBatchOffset, batch->rowPositions + batch->numRows, sizeof(int), 0, cudaMemcpyDeviceToDevice));

      delete batch;
      cudaFree(losses);
    }

    std::transform (predictions, predictions + params.numRows, predictions, [&](T val){ return 1.0 / (1.0 + std::exp(-val)); });
  }
}

template<typename T>
T Trainer<T>::oneEpoch(bool update) {
  Timer timer;
  log_debug(this->params.verbose, "Computing an FFM epoch (update = %s)", update ? "true" : "false");

  T loss = 0;

  for (int i = 0; i < params.nGpus; i++) {
    int initialBatchOffset = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(cBatchOffset, &initialBatchOffset, sizeof(int)));

    while (trainDataBatcher[i]->hasNext()) {
      /**
       * Get batch
       */

      timer.tic();
      DatasetBatch<T> *batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);
      timer.toc();
      log_verbose(params.verbose, "Getting batch took %f.", timer.pop());

      // todo once per trainer and dealloc
      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      int alignFeat1 = params.numFields * params.k * 2;
      int alignFeat2 = params.k * 2;

      int threads = MAX_BLOCK_THREADS;
      int maxRowSize = batch->widestRow();
      size_t totalThreads = batch->numRows * maxRowSize;
      int blocks = std::ceil((double)totalThreads / threads);

      /**
        * Alloc
        */
      T* weightsPtr = this->model->weightsRawPtr(i);

      CUDA_CHECK(cudaMemcpyToSymbol(cK, &params.k, sizeof(int)));
      CUDA_CHECK(cudaMemcpyToSymbol(cMaxRowSize, &maxRowSize, sizeof(int)));
      CUDA_CHECK(cudaMemcpyToSymbol(cRows, &batch->numRows, sizeof(int)));
      int alignTmp[2] = { alignFeat1, alignFeat2 };
      CUDA_CHECK(cudaMemcpyToSymbol(cAlignFeat, &alignTmp, 2 * sizeof(int)));
      CUDA_CHECK(cudaMemcpyToSymbol(cRegLambda, &params.regLambda, sizeof(float)));
      CUDA_CHECK(cudaMemcpyToSymbol(cLearningRate, &params.learningRate, sizeof(float)));
      int offset = 2;
      CUDA_CHECK(cudaMemcpyToSymbol(cWeightsOffset, &offset, sizeof(int)));

      timer.tic();

      wTxKernel << < blocks, threads>>> (batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
              weightsPtr, losses, false);

      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      timer.toc();
      log_verbose(params.verbose, "wTx (false) took %f.", timer.pop());

      timer.tic();

      if (update) {
        wTxKernel << < blocks, threads>>> (batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
                weightsPtr, losses, true, batch->labels);
      }

      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      timer.toc();
      log_verbose(params.verbose, "wTx (true) took %f.", timer.pop());

      timer.tic();
      int* labels = batch->labels;
      thrust::counting_iterator<int> counter(0);
      loss += thrust::transform_reduce(counter, counter + batch->numRows , [=]__device__(int idx) {
        return std::log(1 + std::exp(-labels[idx] * losses[idx]));
      },
      (T) 0.0, thrust::plus<T>());
      timer.toc();

      log_verbose(params.verbose, "Loss compute took %f.", timer.pop());

      thrust::fill(thrust::device, losses, losses + batch->numRows, 0.0);

      CUDA_CHECK(cudaMemcpyToSymbol(cBatchOffset, batch->rowPositions + batch->numRows, sizeof(int), 0, cudaMemcpyDeviceToDevice));

      delete batch;
      cudaFree(losses);
    }

    trainDataBatcher[i]->reset();
  }

  if (params.nGpus != 1) {
    // TODO average local weights
  } // Don't do anything for 1GPU cases

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