/*!
 * Copyright (c) 2017 H2O.ai
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#pragma once
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#endif

#include <sstream>
#include <stdio.h>
#include "timer.h"

namespace h2o4gpukmeans {

static const std::string H2O4GPUKMEANS_VERSION = "0.0.3";

template<typename M>
class H2O4GPUKMeans {
 private:
  // Data
  const M *_A;
  int _k;
  int _n;
  int _d;
 public:
  H2O4GPUKMeans(const M *A, int k, int n, int d);
};

template<typename M>
class H2O4GPUKMeansCPU {
 private:
  // Data
  const M *_A;
  int _k;
  int _n;
  int _d;
 public:
  H2O4GPUKMeansCPU(const M *A, int k, int n, int d);
};

template<typename T>
int makePtr_dense(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t rows, size_t cols,
                  const char ord, int k, int max_iterations, int init_from_data, int init_data,
                  T threshold, const T *srcdata, const int *srclabels, const T *centroids,
                  void **pred_centroids, void **pred_labels);

template<typename T>
int makePtr_dense_cpu(int dopredict, int verbose, int seed, int cpu_id, int n_cpu, size_t rows, size_t cols,
                      const char ord, int k, int max_iterations, int init_from_data,
                      int init_data, T threshold, const T *srcdata, const int *srclabels, const T *centroids,
                      void **pred_centroids, void **pred_labels);

template<typename T>
int kmeans_transform(int verbose,
                     int gpu_id, int n_gpu,
                     size_t m, size_t n, const char ord, int k,
                     const T *srcdata, const T *centroids,
                     void **preds);

}  // namespace h2o4gpukmeans
