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

  DatasetBatch(const DatasetBatch &other) : features(other.features), fields(other.fields), values(other.values),
                                            labels(other.labels), scales(other.scales),
                                            rowPositions(other.rowPositions), numRows(other.numRows){}

  DatasetBatch(DatasetBatch &&other) noexcept : features(other.features), fields(other.fields), values(other.values),
                                                labels(other.labels), scales(other.scales),
                                                rowPositions(other.rowPositions), numRows(other.numRows){}

  DatasetBatch(
      size_t *features, size_t* fields, T* values,
      int *labels, T *scales,
      size_t *rowPositions, size_t numRows) : features(features), fields(fields), values(values),
                                              labels(labels), scales(scales),
                                              rowPositions(rowPositions), numRows(numRows) {}

  // Starting position for each row. Of size nr_rows + 1
  size_t* rowPositions;

  // feature:field:value for all the data points in all the rows
  size_t* features;
  size_t* fields;
  T* values;

  // label and scale for each row
  int* labels = nullptr;
  T* scales;

  // Current position in the batch
  size_t pos = 0;

  // Actual samples in the batch
  size_t numRows;

  size_t remaining() {
    return numRows - pos;
  }

  bool hasNext() const {
    return pos < numRows;
  }

  Row<T> *rowAt(size_t rowPos) {
    Row<T>* row;
    if(nullptr != labels) {
      // TODO where to deallocate??
      row = new Row<T>(features + rowPos, fields + rowPos, values + rowPos, labels[rowPos], scales[rowPos], rowPositions[rowPos + 1] - rowPositions[rowPos]);
    } else {
      // Predictions don't need labels
      row = new Row<T>(features + rowPos, fields + rowPos, values + rowPos, -1, scales[rowPos], rowPositions[rowPos + 1] - rowPositions[rowPos]);
    }
    return row;
  }

  Row<T> *nextRow() {
    Row<T>* row;
    if(nullptr != labels) {
      // TODO where to deallocate??
      row = new Row<T>(features + pos, fields + pos, values + pos, labels[pos], scales[pos], rowPositions[pos + 1] - rowPositions[pos]);
    } else {
      // Predictions don't need labels
      row = new Row<T>(features + pos, fields + pos, values + pos, -1, scales[pos], rowPositions[pos + 1] - rowPositions[pos]);
    }
    pos++;
    return row;
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