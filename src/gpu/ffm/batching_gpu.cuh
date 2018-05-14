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

  DatasetBatchGPU(const DatasetBatchGPU &other) : DatasetBatch<T>(other.rows, other.numRows) {}

  DatasetBatchGPU(DatasetBatchGPU &&other) noexcept : DatasetBatch<T>(other.rows, other.numRows) {}

  DatasetBatchGPU &operator=(const DatasetBatchGPU &other) {
    DatasetBatchGPU tmp(other);
    *this = std::move(tmp);
    return *this;
  }

  DatasetBatchGPU &operator=(DatasetBatchGPU &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->rows = other.rows;
    this->numRows = other.numRows;
    return *this;
  }

  DatasetBatchGPU<T>::DatasetBatchGPU(std::vector<Row<T> *> rows, size_t numRows) : DatasetBatch<T>(rows, numRows) {}

};

template<typename T>
class DatasetBatcherGPU : public DatasetBatcher<T> {
 public:
  DatasetBatcherGPU() {}

  ~DatasetBatcherGPU() {
    // TODO implement
  }

  DatasetBatcherGPU(const DatasetBatcherGPU &other) : DatasetBatcher<T>(other), params(other.params) {}

  DatasetBatcherGPU(DatasetBatcherGPU &&other) noexcept : DatasetBatcher<T>(other), params(other.params) {
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

  DatasetBatcherGPU(Dataset<T> const &dataset, Params const &params);

  DatasetBatch<T> nextBatch(size_t batchSize) override;

 private:
  // If true means the GPU had enough memory to hold the whole data
  // Otherwise we keep the data on the CPU and lazily copy batches as required
  bool onGPU = false;

  Params params = Params{};

};

}