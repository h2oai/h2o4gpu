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
#pragma omp parallel for
    for(int i = 0; i < params.nGpus; i++) {
      log_verbose(params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->weights.size(), i);
      localWeights[i] = new thrust::device_vector<T>(this->weights.size());
      thrust::copy(this->weights.begin(), this->weights.end(), this->localWeights[i]->begin());
    }
  }

  ModelGPU(Params &params, const T *weights) : Model<T>(params, weights), localWeights(params.nGpus) {
#pragma omp parallel for
    for(int i = 0; i < params.nGpus; i++) {
      log_verbose(params.verbose, "Copying weights of size %zu to GPU %d for predictions", this->weights.size(), i);
      localWeights[i] = new thrust::device_vector<T>(this->weights.size());
      thrust::copy(this->weights.begin(), this->weights.end(), this->localWeights[i]->begin());
    }

  }

  ~ModelGPU() {
    for(int i = 0; i < localWeights.size(); i++) {
      delete localWeights[i];
    }
  }

  std::vector<thrust::device_vector<T>*> localWeights;

  void copyTo(T *dstWeights) override {
    // TODO probably can be done with counting iterator
    thrust::device_vector<int> indices(this->localWeights[0]->size());
    thrust::sequence(indices.begin(), indices.end(), 0, 1);
    thrust::device_vector<T> onlyWeights(indices.size() / 2);

    thrust::copy_if(
        this->localWeights[0]->begin(),
        this->localWeights[0]->end(),
        indices.begin(),
        onlyWeights.begin(),
        [=] __device__(int idx) { return idx % 2 == 0; });

    CUDA_CHECK(cudaMemcpy(dstWeights, thrust::raw_pointer_cast(onlyWeights.data()), onlyWeights.size() * sizeof(T), cudaMemcpyDeviceToHost));
  };

  T* weightsRawPtr(int i) override {
    return thrust::raw_pointer_cast(this->localWeights[i]->data());
  }

};

}
