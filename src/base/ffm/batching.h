/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../../include/data/ffm/data.h"
#include "../../include/solver/ffm_api.h"
#include "../../common/logger.h"

namespace ffm {

template<typename T>
class DatasetBatch {

 public:
  DatasetBatch() {}

  DatasetBatch(const DatasetBatch &other) : features(other.features), fields(other.fields), values(other.values),
                                            labels(other.labels), scales(other.scales),
                                            rowPositions(other.rowPositions), numRows(other.numRows){}

  DatasetBatch(DatasetBatch &&other) noexcept : features(other.features), fields(other.fields), values(other.values),
                                                labels(other.labels), scales(other.scales),
                                                rowPositions(other.rowPositions), numRows(other.numRows){}

  DatasetBatch(
      int *features, int* fields, T* values,
      int *labels, T *scales,
      int *rowPositions, int numRows) : features(features), fields(fields), values(values),
                                              labels(labels), scales(scales),
                                              rowPositions(rowPositions), numRows(numRows) {}

  // Starting position for each row. Of size nr_rows + 1
  int* rowPositions;

  // feature:field:value for all the data points in all the rows
  int* features;
  int* fields;
  T* values;

  // label and scale for each row
  int* labels = nullptr;
  T* scales;

  // Current position in the batch
  int pos = 0;

  // Actual samples in the batch
  int numRows;

  int remaining() {
    return numRows - pos;
  }

  bool hasNext() const {
    return pos < numRows;
  }

  virtual int widestRow() { return 0.0; }
};

template<typename T>
class DatasetBatcher {
 public:
  DatasetBatcher() {}

  DatasetBatcher(int numRows) : numRows(numRows) {}

  DatasetBatcher(Dataset<T> &dataset, Params const &params, int rows) : dataset(&dataset), numRows(rows) {}

  bool hasNext() const {
    return pos < numRows;
  }

  int remaining() {
    return numRows - pos;
  }

  void reset() {
    pos = 0;
  }

  virtual DatasetBatch<T> *nextBatch(int batchSize) {
    int actualBatchSize = batchSize <= this->remaining() && batchSize > 0 ? batchSize : this->remaining();

    int moveBy = this->dataset->rowPositions[this->pos];
    DatasetBatch<T> *batch = new DatasetBatch<T>(this->dataset->features + moveBy,
                                                 this->dataset->fields + moveBy,
                                                 this->dataset->values + moveBy,
                                                 this->dataset->labels + this->pos,
                                                 this->dataset->scales + this->pos,
                                                 this->dataset->rowPositions + this->pos,
                                                 actualBatchSize);
    this->pos = this->pos + actualBatchSize;

    return batch;
  }

  bool empty() {
    return numRows <= 0;
  }

 protected:
  Dataset<T> *dataset;
  int pos = 0;
  int numRows = 0;

};

}