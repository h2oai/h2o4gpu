/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include "../../base/ffm/model.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

namespace ffm {

template<typename T>
class ModelGPU : public Model<T> {
 public:
  ModelGPU(Params &params) : Model<T>(params), localWeights(params.nGpus) {
#pragma omp for
    for(int i = 0; i < params.nGpus; i++) {
      log_verbose(params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->weights.size(), i);
      localWeights[i] = new thrust::device_vector<T>(this->weights.size());
      thrust::copy(this->weights.begin(), this->weights.end(), this->localWeights[i]->begin());
    }
  }

  ModelGPU(Params &params, const T *weights) : Model<T>(params, weights), localWeights(params.nGpus) {
#pragma omp for
    for(int i = 0; i < params.nGpus; i++) {
      log_verbose(params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->weights.size(), i);
      localWeights[i] = new thrust::device_vector<T>(this->weights.size());
      thrust::copy(this->weights.begin(), this->weights.end(), this->localWeights[i]->begin());
    }

  }

  ~ModelGPU() {
#pragma omp for
    for(int i = 0; i < localWeights.size(); i++) {
      delete localWeights[i];
    }
  }

  std::vector<thrust::device_vector<T>*> localWeights;

  void copyTo(T *dstWeights) override {
    // TODO probably can be done with counting iterator
    thrust::device_vector<int> indices(this->localWeights[0]->size());
    thrust::sequence(indices.begin(), indices.end(), 0, 1);

    thrust::copy_if(
        thrust::raw_pointer_cast(this->localWeights[0]->data()),
        thrust::raw_pointer_cast(this->localWeights[0]->data()) + this->localWeights[0]->size(),
        thrust::raw_pointer_cast(indices.data()),
        dstWeights,
        [=] __device__(int idx) { return idx % 2 == 0; });
  };

  T* weightsRawPtr(int i) override {
    return thrust::raw_pointer_cast(this->localWeights[i]->data());
  }

};

}
