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
  Row(int *features, int* fields, T* values, int label, T scale, int size) : features(features), fields(fields), values(values),
                                                                             label(label), scale(scale), size(size) {}

  // feature:field:value for all the data points in all the rows
  int* features;
  int* fields;
  T* values;

  int label;
  T scale;
  int size;

};

template<typename T>
class Dataset {
 public:
  Dataset() {}

  Dataset(int numFields, int numFeatures, int numRows, int numNodes, int *features, int* fields, T* values, int *labels, T *scales, int *rowPositions) :
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
  int numRows;
  // Total number of nodes in all the rows
  int numNodes;
  // Total number of fields
  int numFields = 0;
  // Total number of features
  int numFeatures = 0;

  // Starting position for each row. Of size nr_rows + 1
  int* rowPositions;

  // feature:field:value for all the data points in all the rows
  int* features;
  int* fields;
  T* values;

  // label and scale for each row
  int* labels;
  T* scales;

  // Whether resides on CPU or GPU
  bool cpu = true;

  // Total size in bytes required to hold this structure
  int requiredBytes() const {
    return numNodes * 2 * sizeof(int) + numRows * sizeof(T) + numRows * sizeof(int) + numRows * sizeof(T);
  }
};

}