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

#define MAX_BLOCK_THREADS 128

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

__constant__ int cK[1];
__constant__ int cMaxRowSize[1];
__constant__ int cRows[1];
__constant__ int cAlignFeat[2];
__constant__ float cRegLambda[1];
__constant__ float cLearningRate[1];
__constant__ int cWeightsOffset[1];

template<typename T>
__global__ void wTxKernel(const int *__restrict__ features, const int *__restrict__ fields, const T *__restrict__ values,
                          const T *__restrict__ r, const int *__restrict__ rowSizes,
                          T *__restrict__ weightsPtr, T *__restrict__ losses, bool update, const int *__restrict__ labels = nullptr) {
  int rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / cMaxRowSize[0];

  if(rowIdx > cRows[0]) return;

  int rowSize = rowSizes[rowIdx + 1] - rowSizes[rowIdx];

  int nodeIdx = (blockIdx.x * blockDim.x + threadIdx.x) % cMaxRowSize[0];

  if(nodeIdx >= rowSize) return;

  __shared__ int fieldFeature[MAX_BLOCK_THREADS * 2];
  __shared__ T vals[MAX_BLOCK_THREADS];
  __shared__ T scales[MAX_BLOCK_THREADS];

  int n1 = rowSizes[rowIdx] + nodeIdx;

  fieldFeature[threadIdx.x * 2] = fields[n1];
  fieldFeature[threadIdx.x * 2 + 1] = features[n1];
  vals[threadIdx.x] = values[n1];
  scales[rowIdx % MAX_BLOCK_THREADS] = r[rowIdx];

  __syncthreads();

  T loss = 0.0;

  T expnyt = 0;
  T kappa = 0;
  if(update) {
    expnyt = std::exp(-labels[rowIdx] * losses[rowIdx]);
    kappa = -labels[rowIdx] * expnyt / (1 + expnyt);
  }

  for(int i = 1; n1 + i < rowSizes[rowIdx + 1]; i++) {
    const int n2 = n1 + i;

    // We cache some of the field:feature:values in shared memory, only as many "nodes" as there are threads
    // so we know for 100% the initial node will be cached (since we run 1 thread per each starting node)
    // but the subsequent nodes can be within the same block or they can spill to consequitive blocks
    // depending on the size of the row and number of threads in a block
    const int idx1 = fieldFeature[threadIdx.x * 2 + 1] * cAlignFeat[0] +
        (threadIdx.x + i < MAX_BLOCK_THREADS ? fieldFeature[i * 2] : fields[n2]) * cAlignFeat[1];
    const int idx2 = (threadIdx.x + i < MAX_BLOCK_THREADS ? fieldFeature[i * 2 + 1] : features[n2]) * cAlignFeat[0] +
        fieldFeature[threadIdx.x * 2] * cAlignFeat[1];
    T *w1 = weightsPtr + idx1;
    T *w2 = weightsPtr + idx2;

    const T v = vals[threadIdx.x] * (threadIdx.x + i < MAX_BLOCK_THREADS ? vals[i] : values[n2]) * scales[rowIdx % MAX_BLOCK_THREADS];

    if (update) {
      for (int d = 0; d < cK[0] * cWeightsOffset[0]; d+=cWeightsOffset[0]) {
        T w1d = w1[d];
        T w2d = w2[d];
        const T g1 = cRegLambda[0] * w1d + kappa * w2d * v;
        const T g2 = cRegLambda[0] * w2d + kappa * w1d * v;

        w1[d+1] += g1 * g1;
        w2[d+1] += g2 * g2;

        w1[d] -= cLearningRate[0] / std::sqrt(w1[d+1]) * g1;
        w2[d] -= cLearningRate[0] / std::sqrt(w2[d+1]) * g2;
      }
    } else {
      for (int d = 0; d < cK[0] * cWeightsOffset[0]; d+=cWeightsOffset[0]) {
        loss += w1[d] * w2[d] * v;
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
    log_verbose(this->params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->model.weights.size(), i);
    thrust::device_vector<T> localWeights(this->model.weights.begin(), this->model.weights.end());

    int record = 0;
    while (trainDataBatcher[i]->hasNext()) {
      DatasetBatch<T> *batch = trainDataBatcher[i]->nextBatch(this->params.batchSize);

      int threads = MAX_BLOCK_THREADS; // TODO based on row number??
      int maxRowSize = batch->widestRow();
      int totalThreads = batch->numRows * maxRowSize;
      int blocks = std::ceil(totalThreads / threads);

      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      T* weightsPtr = thrust::raw_pointer_cast(localWeights.data());

      int alignFeat1 = params.numFields * params.k;
      int alignFeat2 = params.k;

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
  for (int i = 0; i < params.nGpus; i++) {
    log_verbose(this->params.verbose, "Copying weights of size %zu to GPU %d", this->model.weights.size(), i);

    /**
     * Copy Weights
     * */
    timer.tic();
    // TODO do only once for all iterations?
    allLocalWeights[i].resize(this->model.weights.size());
    thrust::copy(this->model.weights.begin(), this->model.weights.end(), allLocalWeights[i].begin() );
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

      int alignFeat1 = params.numFields * params.k * 2;
      int alignFeat2 = params.k * 2;

      int threads = MAX_BLOCK_THREADS; // TODO based on row number??
      int maxRowSize = batch->widestRow();
      int totalThreads = batch->numRows * maxRowSize;
      int blocks = std::ceil((double)totalThreads / threads);

      /**
        * Alloc
        */
      // todo dealloc
      T *losses;
      cudaMalloc(&losses, batch->numRows * sizeof(T));

      T* weightsPtr = thrust::raw_pointer_cast(allLocalWeights[i].data());

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

      if (update) {
        wTxKernel << < blocks, threads>>> (batch->features, batch->fields, batch->values, batch->scales, batch->rowPositions,
                weightsPtr, losses, true, batch->labels);
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
  } else {
    timer.tic();
    thrust::copy(allLocalWeights[0].begin(), allLocalWeights[0].end(), this->model.weights.begin());
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