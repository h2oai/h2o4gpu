#ifndef __ELASTIC_NET_PTR_H__
#define __ELASTIC_NET_PTR_H__

#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>

#include "matrix/matrix_dense.h"
#include "h2oaiglm.h"
#include "timer.h"
#include <omp.h>

namespace h2oaiglm {


template<typename T>
T getRMSE(size_t len, const T *v1, const T *v2) {
  double rmse = 0;
  for (size_t i = 0; i < len; ++i) {
    double d = v1[i] - v2[i];
    rmse += d * d;
  }
  rmse /= (double) len;
  return static_cast<T>(std::sqrt(rmse));
 }

template<typename T>
  T getRMSE(const T*weights, size_t len, const T *v1, const T *v2) {
  double weightsum=0;
  for (size_t i = 0; i < len; ++i) {
    weightsum += weights[i];
  }

  double rmse = 0;
  for (size_t i = 0; i < len; ++i) {
    double d = v1[i] - v2[i];
    rmse += d * d * weights[i];
  }

  rmse /= weightsum;
  return static_cast<T>(std::sqrt(rmse));
 }


// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <h2oaiglm>/matlab/examples/lasso_path.m for detailed description.
// m and n are training data size


   template<typename T>
     double ElasticNetptr(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                          size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                          double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                          void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);

   
template <typename T>
  int makePtr_dense(int sharedA, int me, int wDev, size_t m, size_t n, size_t mValid, const char ord, const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight, void **_data, void **_datay, void **_vdata, void **_vdatay, void **_weight);
  
#ifdef __cplusplus
    extern "C" {
#endif
      double elastic_net_ptr_double(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                  size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                  double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                                    void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);
      double elastic_net_ptr_float(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                   size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                   double lambda_min_ratio, int nLambdas, int nFolds, int nAlphas,
                                   void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);

#ifdef __cplusplus
    }
#endif

}

#endif
