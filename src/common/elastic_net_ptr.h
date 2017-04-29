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
  

// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <h2oaiglm>/matlab/examples/lasso_path.m for detailed description.
// m and n are training data size
    template<typename T>
      double ElasticNetptr(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                           size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                           double lambda_min_ratio, int nLambdas, int nAlphas,
                           void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);

    template<typename T>
      int makePtr(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                  T *trainX, T *trainY, T *validX, T *validY, T *weight,  //CPU
                  void **a, void **b, void **c, void **d, void **e)  // GPU
    {
      // TODO: See if ok to save memory even between input data and data inside Asource_
      int sharedAlocal=-abs(sharedA); // passes pointer in and out (sharedA=0 would copy data to new memory), but doesn't do any Equilibration
      //      int sharedAlocal=0;
      //int sharedAlocal=1;
      h2oaiglm::MatrixDense<T> Asource_(sharedAlocal, sourceme, sourceDev, ord, mTrain, n, mValid, trainX, trainY, validX, validY, weight);
      *a = reinterpret_cast<void *>(Asource_._data);
      *b = reinterpret_cast<void *>(Asource_._datay);
      *c = reinterpret_cast<void *>(Asource_._vdata);
      *d = reinterpret_cast<void *>(Asource_._vdatay);
      *e = reinterpret_cast<void *>(Asource_._weight);
      fprintf(stderr,"pointer %p\n",*a);
      fprintf(stderr,"pointer %p\n",*b);
      fprintf(stderr,"pointer %p\n",*c);
      fprintf(stderr,"pointer %p\n",*d);
      fprintf(stderr,"pointer %p\n",*e);
      return 0;
    }


#ifdef __cplusplus
    extern "C" {
#endif

      int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                          double* trainX, double* trainY, double* validX, double* validY, double *weight,
                          void**a, void**b, void**c, void**d, void **e);
      int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                         float* trainX, float* trainY, float* validX, float* validY, float *weight,
                         void**a, void**b, void**c, void**d, void **e);

      double elastic_net_ptr_double(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                  size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                  double lambda_min_ratio, int nLambdas, int nAlphas,
                                    void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);
      double elastic_net_ptr_float(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                                   size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                   double lambda_min_ratio, int nLambdas, int nAlphas,
                                   void *trainXptr, void *trainYptr, void *validXptr, void *validYptr, void *weightptr);

#ifdef __cplusplus
    }
#endif

}

#endif
