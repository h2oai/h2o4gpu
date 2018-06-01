/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../../base/ffm/batching.h"
#include "../../include/solver/ffm_api.h"

namespace ffm {

template<typename T>
class DatasetBatchGPU : public DatasetBatch<T> {
 public:
  DatasetBatchGPU() {}

  DatasetBatchGPU(int *features, int* fields, T* values,
                  int *labels, T *scales, int *rowPositions,
                  int numRows, bool dealloc = false) : DatasetBatch<T>(features, fields, values, labels, scales, rowPositions, numRows), dealloc(dealloc) {}


  ~DatasetBatchGPU() {
    if(dealloc) {
      cudaFree(this->features);
      cudaFree(this->fields);
      cudaFree(this->values);
      if(this->labels) {
        cudaFree(this->labels);
      }
      cudaFree(this->scales);
      cudaFree(this->rowPositions);
    }
  }

  bool dealloc = false;

  int widestRow() override;
};

template<typename T>
class DatasetBatcherGPU : public DatasetBatcher<T> {
 public:
  DatasetBatcherGPU() {}

  DatasetBatcherGPU(Dataset<T> &dataset, Params const &params, int rows, int nodes);

  DatasetBatcherGPU(const DatasetBatcherGPU &other) : DatasetBatcher<T>(other), params(other.params) {}

  DatasetBatcherGPU(DatasetBatcherGPU &&other) noexcept : DatasetBatcher<T>(other), params(other.params) {}

  ~DatasetBatcherGPU() {
    delete this->dataset;
  }

  DatasetBatcherGPU &operator=(const DatasetBatcherGPU &other) {
    DatasetBatcherGPU tmp(other);
    *this = std::move(tmp);
    return *this;
  }

  DatasetBatcherGPU &operator=(DatasetBatcherGPU &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->dataset = other.dataset;
    this->params = other.params;
    return *this;
  }

  DatasetBatch<T> *nextBatch(int batchSize) override;

 private:
  // If true means the GPU had enough memory to hold the whole data
  // Otherwise we keep the data on the CPU and lazily copy batches as required
  bool onGPU = false;

  Params params = Params{};

};

}