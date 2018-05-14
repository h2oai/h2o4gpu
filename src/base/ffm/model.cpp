/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "model.h"
#include <random>
#include <string.h>

namespace ffm {

template<typename T>
Model<T>::Model(Params &params) : weights(params.numFeatures * params.numFields * params.k), gradients(params.numFeatures * params.numFields * params.k) {
  this->numFeatures = params.numFeatures;
  this->numFields = params.numFields;
  this->k = params.k;
  this->normalize = params.normalize;

  T coef = 1.0f / sqrt(this->k);

  std::default_random_engine generator(params.seed);
  std::uniform_real_distribution<T> distribution(0.0, 1.0);

  for (int i = 0; i < weights.size(); i++) {
    this->weights[i] = coef * distribution(generator);
    this->gradients[i] = 1.0;
  }

}

template<typename T>
Model<T>::Model(Params &params, T *weights) : weights(params.numFeatures * params.numFields * params.k) {
  this->numFeatures = params.numFeatures;
  this->numFields = params.numFields;
  this->k = params.k;
  this->normalize = params.normalize;

  for (int i = 0; i < this->weights.size(); i++) {
    this->weights[i] = weights[i];
  }
}

template<typename T>
void Model<T>::copyTo(T *dstWeights) {
  memcpy(dstWeights, this->weights.data(), this->weights.size() * sizeof(T));
};

template
class Model<float>;
template
class Model<double>;

}
