/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include "array.cuh"

namespace H2O4GPU{
namespace KMeans {

template <typename T>
class KmeansInitBase {
 public:
  virtual ~KmeansInitBase() {}
  virtual Array::CUDAArray<T> operator()(Array::CUDAArray<T>& data) = 0;
};

template <typename T>
struct KmeansLlInit : public KmeansInitBase<T> {
 private:
  double over_sample;
  int seed;

 public:
  KmeansLlInit () : over_sample (2.0), seed (0) {}
  virtual ~KmeansLlInit () override {}

  Array::CUDAArray<T> sample_centroids(Array::CUDAArray<T>& data,
                                Array::CUDAArray<T>& prob);

  Array::CUDAArray<T> operator()(Array::CUDAArray<T>& data) override;
};

}  // namespace Kmeans
}  // namespace H2O4GPU
