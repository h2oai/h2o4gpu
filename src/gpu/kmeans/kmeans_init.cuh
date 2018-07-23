/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef KMEANS_INIT_H_
#define KMEANS_INIT_H_


#include <memory>

#include "KmMatrix/KmConfig.h"
#include "KmMatrix/KmMatrix.hpp"
#include "KmMatrix/utils.cuh"
#include "KmMatrix/Generator.hpp"
#include "KmMatrix/Generator.cuh"
#include "KmMatrix/GpuInfo.cuh"

namespace H2O4GPU{
namespace KMeans {

/*
 * Base class used for all K-Means initialization algorithms.
 */
template <typename T>
class KmeansInitBase {
 public:
  virtual ~KmeansInitBase() {}
  /*
   * Select k centroids from data.
   *
   * @param data data points stored in row major matrix.
   * @param k number of centroids.
   */
  virtual KmMatrix<T> operator()(KmMatrix<T>& data, size_t k) = 0;
};

/*
 * Each instance of KmeansLlInit corresponds to one dataset, if a new data set
 * is used, users need to create a new instance.
 *
 * k-means|| algorithm based on the paper:
 * <a href="http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf">
 * Scalable K-Means++
 * </a>
 *
 * @tparam Data type, supported types are float and double.
 */
template <typename T>
struct KmeansLlInit : public KmeansInitBase<T> {
 private:
  T over_sample_;
  int seed_;
  int k_;

  std::unique_ptr<GeneratorBase<T>> generator_;

  // Buffer like variables
  // store the self dot product of each data point
  KmMatrix<T> data_dot_;
  // store distances between each data point and centroids
  KmMatrix<T> distance_pairs_;

  KmMatrix<int> weight_centroids(KmMatrix<T>& centroids);
  KmMatrix<T> probability(KmMatrix<T>& data, KmMatrix<T>& centroids);
 public:
  // sample_centroids should not be part of the interface, but following error
  // is generated when put in private section:
  // The enclosing parent function ("sample_centroids") for an extended
  // __device__ lambda cannot have private or protected access within its class
  KmMatrix<T> sample_centroids(KmMatrix<T>& data, KmMatrix<T>& centroids);

  /*
   * Initialize KmeansLlInit algorithm, with default:
   *  over_sample = 1.5,
   *  seed = 0,
   */
  KmeansLlInit () :
      over_sample_ (1.5f), seed_ (-1), k_ (0),
      generator_ (new UniformGenerator<T>) {}

  /*
   * Initialize KmeansLlInit algorithm, with default:
   *  seed = 0,
   *
   * @param over_sample over_sample rate.
   *    \f$p_x = over_sample \times \frac{d^2(x, C)}{\Phi_X (C)}\f$
   *    Note that when \f$over_sample != 1\f$, the probability for each data
   *    point doesn't add to 1.
   */
  KmeansLlInit (T _over_sample) :
      over_sample_ (_over_sample), seed_ (-1), k_ (0),
      generator_ (new UniformGenerator<T>) {}

  /*
   * Initialize KmeansLlInit algorithm.
   *
   * @param seed Seed used to generate threshold for sampling centroids.
   * @param over_sample over_sample rate.
   *    \f$p_x = over_sample \times \frac{d^2(x, C)}{\Phi_X (C)}\f$
   */
  KmeansLlInit (int _seed, T _over_sample) :
      seed_(_seed), k_(0),
      generator_ (new UniformGenerator<T>(seed_)) {}

  /*
   * Initialize KmeansLlInit algorithm.
   *
   * @param gen Unique pointer to a generator used to generate threshold for
   *    sampling centroids.
   * @param over_sample over_sample rate.
   *    \f$p_x = over_sample \times \frac{d^2(x, C)}{\Phi_X (C)}\f$
   */
  KmeansLlInit (std::unique_ptr<GeneratorBase<T>>& _gen, T _over_sample) :
      generator_(std::move(_gen)), over_sample_ (1.5f), seed_ (-1), k_(0) {}

  virtual ~KmeansLlInit () override {}

  /*
   * Select k centroids from data.
   * @param data data points stored in row major matrix.
   * @param k number of centroids.
   */
  KmMatrix<T> operator()(KmMatrix<T>& data, size_t k) override;
};


// FIXME: Make kmeans++ a derived class of KmeansInitBase

}  // namespace Kmeans
}  // namespace H2O4GPU

#endif  // KMEANS_INIT_H_