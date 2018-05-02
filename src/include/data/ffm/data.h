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
class Node {

 public:
  Node() {}

  ~Node() {
    // TODO implement - watch out for SWIG calling this
  }

  size_t featureIdx = 0;
  size_t fieldIdx = 0;
  T value = 0;
};

/**
 * Usually FFM data is presented in a modified LIBSVM format
 * label field1:feat1:val1 field2:feat2:val2 ...
 *
 * This data structure represents a single row in such format.
 */
template<typename T>
class Row {

 public:
  Row() {}

  Row(int label, T scale, size_t size, std::vector<Node<T> *> &data) : label(label), scale(scale), size(size), data(data) {}

  Row(int label, T scale, size_t size, Node<T> *data) : label(label), scale(scale), size(size) {
    this->data = std::vector<Node<T>*>(size);
    for(int i = 0; i < size; i++) {
      this->data[i] = &data[i];
    }
  }

  Row(int label, T scale, size_t size, Node<T> *data[]) : label(label), scale(scale), size(size) {
    for(int i = 0; i < size; i++) {
      this->data[i] = data[i];
    }
  }


  ~Row() {
    // TODO implement - watch out for SWIG calling this
  }

  std::vector<Node<T> *> data;
  int label;
  T scale;
  size_t size;

};

template<typename T>
class Dataset {
 public:
  Dataset() {}

  Dataset(int numFields, int numFeatures, size_t numRows, size_t numNodes, std::vector<Row<T> *> rows) :
      numRows(numRows),
      numNodes(numNodes),
      numFields(numFields),
      numFeatures(numFeatures) {
      this->rows = rows;
  }

  ~Dataset() {
    // TODO implement
  }

  // Number of rows in the dataset
  size_t numRows;
  // Total number of nodes in all the rows
  size_t numNodes;
  // Total number of fields
  int numFields = 0;
  // Total number of features
  int numFeatures = 0;

  std::vector<Row<T> *> rows;

  // Whether resides on CPU or GPU
  bool cpu = true;

  // Total size in bytes required to hold this structure
  size_t requiredBytes() const {
    return numNodes * sizeof(Node<T>) + numRows * sizeof(int) + numRows * sizeof(T) + 3 * sizeof(size_t)
        + 2 * sizeof(int);
  }
};

}