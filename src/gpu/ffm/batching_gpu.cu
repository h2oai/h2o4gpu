/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "batching_gpu.cuh"
#include "../utils/cuda.h"
#include "../../include/solver/ffm_api.h"

namespace ffm {

/**
 *
 * DatasetBatcherGPU Methods
 *
 */

template<typename T>
DatasetBatcherGPU<T>::DatasetBatcherGPU(Dataset<T> const &dataset, Params const &params)
    : DatasetBatcher<T>(dataset.numRows), params(params) {

  size_t requiredBytes = dataset.requiredBytes();
  size_t availableBytesFree = 0;
  size_t availableBytesTotal = 0;
  CUDA_CHECK(cudaMemGetInfo(&availableBytesFree, &availableBytesTotal));

  // If possible put all the data on the GPU in one go so we don't waste time sending it over and over for each epoch
  if (dataset.cpu && requiredBytes < (availableBytesFree * 0.75)) {
    log_debug(params.verbose,
              "Creating a dataset batcher requiring %zu bytes fully on GPU (available %zu bytes).",
              requiredBytes,
              availableBytesFree);
    this->onGPU = true;

    Dataset<T> datasetGpu;
    datasetGpu.cpu = false;
    datasetGpu.numRows = dataset.numRows;
    datasetGpu.numFields = dataset.numFields;
    datasetGpu.rowPositions = dataset.rowPositions;
    datasetGpu.labels = dataset.labels;
    datasetGpu.scales = dataset.scales;

    CUDA_CHECK(cudaMalloc(&datasetGpu.features, params.numNodes * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.features, dataset.features, params.numNodes * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&datasetGpu.fields, params.numNodes * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.fields, dataset.fields, params.numNodes * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&datasetGpu.values, params.numNodes * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.values, dataset.values, params.numNodes * sizeof(T), cudaMemcpyHostToDevice));

    this->dataset = datasetGpu;
  } else {
    log_debug(params.verbose,
              "Creating a dataset batcher requiring %zu bytes on CPU, batches will be transfered on demand (available %zu bytes).",
              requiredBytes,
              availableBytesFree);
    this->dataset = dataset;
  }
}

template<typename T>
DatasetBatch<T> DatasetBatcherGPU<T>::nextBatch(size_t batchSize) {
  log_debug(this->params.verbose, "Asked for batch of size %zu.", batchSize);
  size_t actualBatchSize = batchSize <= this->remaining() ? batchSize : this->remaining();

  if (this->onGPU) {
    // Take the whole thing as 1 batch if all is on GPU
    size_t actualBatchSize = this->remaining();

    log_debug(this->params.verbose,
              "Creating batch of size %zu (asked for %zu) directly on the GPU.",
              actualBatchSize,
              batchSize);

    DatasetBatchGPU<T> batch(this->dataset.features + this->pos, this->dataset.fields + this->pos, this->dataset.values + this->pos,
                             this->dataset.labels + this->pos, this->dataset.scales + this->pos, this->dataset.rowPositions + this->pos,
                             actualBatchSize);
    this->pos = this->pos + actualBatchSize;

    log_debug(this->params.verbose, "New batcher position %zu", this->pos);
    return batch;
  } else {
    log_debug(this->params.verbose,
              "Creating batch of size %zu (asked for %zu) from the CPU.",
              actualBatchSize,
              batchSize);
    // TODO copy batch to GPU
    DatasetBatchGPU<T> batch;
    this->pos = this->pos + actualBatchSize;

    return batch;
  }
}

template
class DatasetBatcher<float>;
template
class DatasetBatcher<double>;

template
class DatasetBatcherGPU<float>;
template
class DatasetBatcherGPU<double>;

template
class DatasetBatch<float>;
template
class DatasetBatch<double>;

}