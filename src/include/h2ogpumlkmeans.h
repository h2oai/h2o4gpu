#pragma once
#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
#endif

#include <sstream>
#include <stdio.h>
#include "timer.h"

namespace h2ogpumlkmeans {

  static const std::string H2OGPUMLKMEANS_VERSION = "0.0.1";

  template <typename M>
  class H2OGPUMLKMeans {
    private:
      // Data
      const M* _A;
      int _k;
      int _n;
      int _d;
    public:
      H2OGPUMLKMeans(const M *A, int k, int n, int d);
  };
  template <typename M>
  class H2OGPUMLKMeansCPU {
    private:
      // Data
      const M* _A;
      int _k;
      int _n;
      int _d;
    public:
      H2OGPUMLKMeansCPU(const M *A, int k, int n, int d);
  };

  template <typename T>
    int makePtr_dense(int verbose, int seed, int gpu_id, int n_gpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, T threshold, const T* srcdata, const int*srclabels, void ** res);
  template <typename T>
    int makePtr_dense_cpu(int verbose, int seed, int cpu_id, int n_cpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, T threshold, const T* srcdata, const int*srclabels, void ** res);

}  // namespace h2ogpumlkmeans
