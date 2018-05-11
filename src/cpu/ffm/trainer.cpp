/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "../../base/ffm/trainer.h"

namespace ffm {

template<typename T>
Trainer<T>::Trainer(const Dataset<T> &dataset, Model<T> &model, Params const &params)
    : trainDataBatcher(1), model(model), params(params) {
  // TODO implement
}

template<typename T>
void Trainer<T>::predict(T *predictions) {
  // TODO implement
}

template<typename T>
T Trainer<T>::oneEpoch(bool update) {
  // TODO implement
}

template<typename T>
bool Trainer<T>::earlyStop() {
  // TODO implement
  return false;
}

template class Trainer<float>;
template class Trainer<double>;

}