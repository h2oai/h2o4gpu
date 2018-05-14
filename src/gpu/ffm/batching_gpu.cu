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
    datasetGpu.rows = std::vector<Row<T>*>(dataset.numRows);
    datasetGpu.cpu = false;
    datasetGpu.numRows = dataset.numRows;
    datasetGpu.numFields = dataset.numFields;

#pragma omp parallel for
    for (int r = 0; r < dataset.numRows; r++) {
      std::vector<Node<T> *> dataVector(dataset.rows[r]->size);

#pragma omp parallel for
      for(int n = 0; n < dataset.rows[r]->size; n++) {
        CUDA_CHECK(cudaMalloc(&dataVector[n], sizeof(Node<T>)));
        CUDA_CHECK(cudaMemcpy(dataVector[n], dataset.rows[r]->data[n], sizeof(Node<T>), cudaMemcpyHostToDevice));
      }

      Row<T> *d_row = new Row<T>(dataset.rows[r]->label, dataset.rows[r]->scale, dataset.rows[r]->size, dataVector);
      datasetGpu.rows[r] = d_row;
    }

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
  // TODO take the whole thing as 1 batch if all is on GPU
  size_t actualBatchSize = batchSize <= this->remaining() ? batchSize : this->remaining();

  if (this->onGPU) {
    log_debug(this->params.verbose,
              "Creating batch of size %zu (asked for %zu) directly on the GPU.",
              actualBatchSize,
              batchSize);

    // TODO memory duplication??
    std::vector<Row<T>*> slice(this->dataset.rows.cbegin() + this->pos, this->dataset.rows.cbegin() + this->pos + actualBatchSize);

    DatasetBatchGPU<T> batch(slice, actualBatchSize);
    this->pos = this->pos + actualBatchSize;

    log_debug(this->params.verbose, "New position %zu", this->pos);
    return batch;
  } else {
    log_debug(this->params.verbose,
              "Creating batch of size %zu (asked for %zu) from the CPU.",
              actualBatchSize,
              batchSize);
    std::vector<Row<T> *> dRows;
    // TODO copy [this->dataset.rows + this->pos, this->dataset.rows + this->pos + actualBatchSize) to dRows
    DatasetBatchGPU<T> batch(dRows, actualBatchSize);
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