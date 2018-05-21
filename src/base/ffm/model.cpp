/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "model.h"
#include <random>
#include <algorithm>
#include <string.h>

namespace ffm {

template<typename T>
Model<T>::Model(Params &params) : weights(2 * params.numFeatures * params.numFields * params.k) {
  this->numFeatures = params.numFeatures;
  this->numFields = params.numFields;
  this->k = params.k;
  this->normalize = params.normalize;

  T coef = 1.0f / sqrt(this->k);

  std::default_random_engine generator(params.seed);
  std::uniform_real_distribution<T> distribution(0.0, 1.0);

  for (int i = 0; i < weights.size(); i += 2) {
    this->weights[i] = coef * distribution(generator);
    this->weights[i + 1] = 1.0;
  }

}

template<typename T>
Model<T>::Model(Params &params, const T *_weights) : weights(params.numFeatures * params.numFields * params.k) {
  this->numFeatures = params.numFeatures;
  this->numFields = params.numFields;
  this->k = params.k;
  this->normalize = params.normalize;

  for (int i = 0; i < this->weights.size(); i++) {
    // TODO memcpy?
    this->weights[i] = _weights[i];
  }
}

template<typename T>
void Model<T>::copyTo(T *dstWeights) {
  // Copy only weights
  std::copy_if( this->weights.begin(), this->weights.end(), dstWeights,
                            []( int x ) { return x % 2; } );
};

template<typename T>
T* Model<T>::weightsRawPtr(int i) {
  return this->weights.data();
}


template
class Model<float>;
template
class Model<double>;

}
