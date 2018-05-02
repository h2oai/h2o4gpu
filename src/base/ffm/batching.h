/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../../include/data/ffm/data.h"
#include "../../common/logger.h"

namespace ffm {

template<typename T>
class DatasetBatch {

 public:
  DatasetBatch() {}

  DatasetBatch(const DatasetBatch &other) : rows(other.rows), numRows(other.numRows) {}

  DatasetBatch(DatasetBatch &&other) noexcept : rows(other.rows), numRows(other.numRows) {}

  DatasetBatch &operator=(const DatasetBatch &other) {
    DatasetBatch tmp(other);
    *this = std::move(tmp);
    return *this;
  }

  DatasetBatch &operator=(DatasetBatch &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    rows = other.rows;
    numRows = other.numRows;
    return *this;
  }

  ~DatasetBatch() {
    // TODO implement!!!
  }

  DatasetBatch(std::vector<Row<T> *> rows, size_t numRows) : rows(rows), numRows(numRows) {}

  std::vector<Row<T> *> rows;

  // Current position in the batch
  size_t pos = 0;

  // Actual samples in the batch
  size_t numRows;

  bool hasNext() const {
    return pos < numRows;
  }

  Row<T> *nextRow() {
    return rows[pos++];
  }

};

template<typename T>
class DatasetBatcher {
 public:
  DatasetBatcher() {}

  DatasetBatcher(size_t numRows) : numRows(numRows) {}

  bool hasNext() const {
    return pos < numRows;
  }

  size_t remaining() {
    return numRows - pos;
  }

  void reset() {
    pos = 0;
  }

  virtual DatasetBatch<T> nextBatch(size_t batchSize) {}

 protected:
  Dataset<T> dataset;
  size_t pos = 0;
  size_t numRows = 0;

};

}