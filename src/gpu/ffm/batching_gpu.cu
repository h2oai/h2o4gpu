/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "batching_gpu.cuh"
#include "../utils/cuda.h"
#include "../../include/solver/ffm_api.h"
#include "../../common/timer.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>

namespace ffm {


/**
 * DatasetBatchGPU Mathods
 */
template <typename T>
size_t DatasetBatchGPU<T>::widestRow() {
  this->rowPositions;

  Timer timer;

  timer.tic();

  thrust::device_vector<size_t> tmpRowSizes(this->numRows);
  thrust::device_vector<size_t> rowPositions(this->rowPositions, this->rowPositions + this->numRows);
  thrust::adjacent_difference(rowPositions.begin(), rowPositions.end(), tmpRowSizes.begin(), thrust::minus<size_t>());

  thrust::device_vector<size_t>::iterator iter = thrust::max_element(tmpRowSizes.begin(), tmpRowSizes.end());

  size_t max_value = *iter;

  size_t widest = (max_value/2.0) * (max_value - 1.0);

  timer.toc();

  // TODO fix
//  return widest;
  return 741;
}

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
    log_verbose(params.verbose,
              "Creating a dataset batcher requiring %zu bytes fully on GPU (available %zu bytes).",
              requiredBytes,
              availableBytesFree);
    this->onGPU = true;

    Dataset<T> datasetGpu;
    datasetGpu.cpu = false;
    datasetGpu.numRows = dataset.numRows;
    datasetGpu.numFields = dataset.numFields;

    CUDA_CHECK(cudaMalloc(&datasetGpu.rowPositions, (params.numRows + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.rowPositions, dataset.rowPositions, (params.numRows + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

    // No need for predict
    if(dataset.labels != nullptr) {
      CUDA_CHECK(cudaMalloc(&datasetGpu.labels, params.numRows * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(datasetGpu.labels, dataset.labels, params.numRows * sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&datasetGpu.scales, params.numRows * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.scales, dataset.scales, params.numRows * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&datasetGpu.features, params.numNodes * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.features, dataset.features, params.numNodes * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&datasetGpu.fields, params.numNodes * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.fields, dataset.fields, params.numNodes * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&datasetGpu.values, params.numNodes * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(datasetGpu.values, dataset.values, params.numNodes * sizeof(T), cudaMemcpyHostToDevice));

    this->dataset = datasetGpu;
  } else {
    log_verbose(params.verbose,
              "Creating a dataset batcher requiring %zu bytes on CPU, batches will be transfered on demand (available %zu bytes).",
              requiredBytes,
              availableBytesFree);
    this->dataset = dataset;
  }
}

template<typename T>
DatasetBatch<T> *DatasetBatcherGPU<T>::nextBatch(size_t batchSize) {
  log_verbose(this->params.verbose, "Asked for batch of size %zu.", batchSize);
  size_t actualBatchSize = batchSize <= this->remaining() ? batchSize : this->remaining();

  if (this->onGPU) {
    log_verbose(this->params.verbose,
              "Creating batch of size %zu (asked for %zu) directly on the GPU.",
              actualBatchSize,
              batchSize);

    // todo DELETE
    DatasetBatchGPU<T> *batch = new DatasetBatchGPU<T>(this->dataset.features + this->pos, this->dataset.fields + this->pos, this->dataset.values + this->pos,
                             this->dataset.labels + this->pos, this->dataset.scales + this->pos, this->dataset.rowPositions + this->pos,
                             actualBatchSize);
    this->pos = this->pos + actualBatchSize;

    return batch;
  } else {
    log_verbose(this->params.verbose,
              "Creating batch of size %zu (asked for %zu) from the CPU.",
              actualBatchSize,
              batchSize);
    // TODO copy batch to GPU
    DatasetBatchGPU<T> *batch = new DatasetBatchGPU<T>();
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