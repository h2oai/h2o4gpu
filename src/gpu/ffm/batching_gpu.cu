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
#include <thrust/host_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/extrema.h>

namespace ffm {


/**
 * DatasetBatchGPU Mathods
 */
template <typename T>
int DatasetBatchGPU<T>::widestRow() {
  thrust::device_vector<int> tmpRowSizes(this->numRows + 1);
  thrust::device_vector<int> rowPositions(this->rowPositions, this->rowPositions + this->numRows + 1);
  thrust::adjacent_difference(rowPositions.begin(), rowPositions.end(), tmpRowSizes.begin(), thrust::minus<int>());

  // Don't take into account 1st difference as it's equal to the first row's size
  thrust::device_vector<int>::iterator iter = thrust::max_element(tmpRowSizes.begin() + 1, tmpRowSizes.end());

  int max_value = *iter;

  return max_value;
}

/**
 *
 * DatasetBatcherGPU Methods
 *
 */
template<typename T>
Dataset<T> *toGpuDataset(Dataset<T> &dataset, Params const &params, int rowOffset, int nodeOffset, int rows, int numNodes) {
  Dataset<T> *datasetGpu = new Dataset<T>();
  datasetGpu->cpu = false;
  datasetGpu->numRows = dataset.numRows;
  datasetGpu->numFields = dataset.numFields;

  cudaStream_t streams[6];
  for(int i = 0; i < 6; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  // todo dealloc in destructor??
  CUDA_CHECK(cudaMalloc(&datasetGpu->rowPositions, (rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&datasetGpu->scales, rows * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&datasetGpu->features, numNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&datasetGpu->fields, numNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&datasetGpu->values, numNodes * sizeof(T)));
  if(dataset.labels != nullptr) {
    CUDA_CHECK(cudaMalloc(&datasetGpu->labels, rows * sizeof(int)));
  }

  // No need for predict
  cudaMemcpyAsync(datasetGpu->rowPositions, dataset.rowPositions + rowOffset, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice, streams[0]);
  if(dataset.labels != nullptr) {
    cudaMemcpyAsync(datasetGpu->labels, dataset.labels + rowOffset, rows * sizeof(int), cudaMemcpyHostToDevice, streams[1]);
  }
  cudaMemcpyAsync(datasetGpu->scales, dataset.scales + rowOffset, rows * sizeof(T), cudaMemcpyHostToDevice, streams[2]);
  cudaMemcpyAsync(datasetGpu->features, dataset.features + nodeOffset, numNodes * sizeof(int), cudaMemcpyHostToDevice, streams[3]);
  cudaMemcpyAsync(datasetGpu->fields, dataset.fields + nodeOffset, numNodes * sizeof(int), cudaMemcpyHostToDevice, streams[4]);
  cudaMemcpyAsync(datasetGpu->values, dataset.values + nodeOffset, numNodes * sizeof(T), cudaMemcpyHostToDevice, streams[5]);

  for(int i = 0; i < 6; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }

  return datasetGpu;
}


template<typename T>
DatasetBatcherGPU<T>::DatasetBatcherGPU(Dataset<T> &dataset, Params const &params, int rows, int nodes)
    : DatasetBatcher<T>(dataset.numRows), params(params) {

  if(dataset.empty()) {
    log_verbose(params.verbose, "Creating a batcher from an empty dataset.");
    return;
  }

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
    this->dataset = toGpuDataset(dataset, params, 0, 0, rows, nodes);
  } else {
    log_verbose(params.verbose,
              "Creating a dataset batcher requiring %zu bytes on CPU, batches will be transfered on demand (available %zu bytes).",
              requiredBytes,
              availableBytesFree);
    this->dataset = &dataset;
  }
}

template<typename T>
DatasetBatch<T> *DatasetBatcherGPU<T>::nextBatch(int batchSize) {
  int actualBatchSize = batchSize <= this->remaining() && batchSize > 0 ? batchSize : this->remaining();

  if (this->onGPU) {
    log_verbose(this->params.verbose,
              "Creating batch of size %d (asked for %d) directly on the GPU.",
              actualBatchSize,
              batchSize);


    int moveBy = 0;
    CUDA_CHECK(cudaMemcpy(&moveBy, &(this->dataset->rowPositions[this->pos]), sizeof(int), cudaMemcpyDeviceToHost));

    DatasetBatchGPU<T> *batch = new DatasetBatchGPU<T>(this->dataset->features + moveBy, this->dataset->fields + moveBy, this->dataset->values + moveBy,
                             this->dataset->labels + this->pos, this->dataset->scales + this->pos, this->dataset->rowPositions + this->pos,
                             actualBatchSize);
    this->pos = this->pos + actualBatchSize;

    return batch;
  } else {
    log_verbose(this->params.verbose,
              "Creating batch of size %d (asked for %d) from the CPU.",
              actualBatchSize,
              batchSize);
    int batchNumNodes = 0;
    for(int i = 0; i < actualBatchSize; i++) {
      batchNumNodes += this->dataset->rowPositions[this->pos + i + 1] - this->dataset->rowPositions[this->pos + i];
    }

    Dataset<T> *gpuDataset = toGpuDataset(*this->dataset, params, this->pos, this->dataset->rowPositions[this->pos], actualBatchSize, batchNumNodes);

    DatasetBatchGPU<T> *batch = new DatasetBatchGPU<T>(gpuDataset->features, gpuDataset->fields, gpuDataset->values,
                                                       gpuDataset->labels, gpuDataset->scales, gpuDataset->rowPositions,
                                                       actualBatchSize, true);
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