/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "stddef.h"
#include "../../../common/logger.h"
#include <vector>
#include <cstdlib>
#include <cstring>

namespace ffm {

template<typename T>
class Row {

 public:
  Row(size_t *features, size_t* fields, T* values, int label, T scale, size_t size) : features(features), fields(fields), values(values),
                                                                             label(label), scale(scale), size(size) {}

  // feature:field:value for all the data points in all the rows
  size_t* features;
  size_t* fields;
  T* values;

  int label;
  T scale;
  size_t size;

};

template<typename T>
class Dataset {
 public:
  Dataset() {}

  Dataset(int numFields, int numFeatures, size_t numRows, size_t numNodes, size_t *features, size_t* fields, T* values, int *labels, T *scales, size_t *rowPositions) :
      numRows(numRows),
      numNodes(numNodes),
      numFields(numFields),
      numFeatures(numFeatures),
      features(features),
      fields(fields),
      values(values),
      labels(labels),
      scales(scales),
      rowPositions(rowPositions){}

  // Number of rows in the dataset
  size_t numRows;
  // Total number of nodes in all the rows
  size_t numNodes;
  // Total number of fields
  int numFields = 0;
  // Total number of features
  int numFeatures = 0;

  // Starting position for each row. Of size nr_rows + 1
  size_t* rowPositions;

  // feature:field:value for all the data points in all the rows
  size_t* features;
  size_t* fields;
  T* values;

  // label and scale for each row
  int* labels;
  T* scales;

  // Whether resides on CPU or GPU
  bool cpu = true;

  // Total size in bytes required to hold this structure
  size_t requiredBytes() const {
    return numNodes * 2 * sizeof(size_t) + numRows * sizeof(T) + numRows * sizeof(int) + numRows * sizeof(T);
  }
};

}