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
#include "pogs.h"
#include "timer.h"
#include <omp.h>

namespace pogs {


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
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
// m and n are training data size
    template<typename T>
      double ElasticNetptr(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, const char ord,
                         size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                         double lambda_min_ratio, int nLambdas, int nAlphas,
                         void *trainXptr, void *trainYptr, void *validXptr, void *validYptr);

    template<typename T>
      int makePtr(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid,
                T *trainX, T *trainY, T *validX, T *validY,  //CPU
                void **a, void **b, void **c, void **d)  // GPU
    {
      // TODO: See if ok to save memory even between input data and data inside Asource_
      int sharedAlocal=-abs(sharedA); // passes pointer in and out (sharedA=0 would copy data to new memory), but doesn't do any Equilibration
      //      int sharedAlocal=0;
      //int sharedAlocal=1;
      pogs::MatrixDense<T> Asource_(sharedAlocal, sourceme, sourceDev, 'r', mTrain, n, mValid, trainX, trainY, validX, validY);
      *a = reinterpret_cast<void *>(Asource_._data);
      *b = reinterpret_cast<void *>(Asource_._datay);
      *c = reinterpret_cast<void *>(Asource_._vdata);
      *d = reinterpret_cast<void *>(Asource_._vdatay);
      fprintf(stderr,"pointer %p\n",*a);
      fprintf(stderr,"pointer %p\n",*b);
      fprintf(stderr,"pointer %p\n",*c);
      fprintf(stderr,"pointer %p\n",*d);
      return 0;
    }


#ifdef __cplusplus
    extern "C" {
#endif

      int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid,
                        double* trainX, double* trainY, double* validX, double* validY,
                        void**a, void**b, void**c, void**d);
    int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid,
                       float* trainX, float* trainY, float* validX, float* validY,
                       void**a, void**b, void**c, void**d);

      double elastic_net_ptr_double(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, int ord,
                                  size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                                  double lambda_min_ratio, int nLambdas, int nAlphas,
                                  void *trainXptr, void *trainYptr, void *validXptr, void *validYptr);
      double elastic_net_ptr_float(int sourceDev, int datatype, int sharedA, int nThreads, int nGPUs, int ord,
                                 size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
                         double lambda_min_ratio, int nLambdas, int nAlphas,
                         void *trainXptr, void *trainYptr, void *validXptr, void *validYptr);

#ifdef __cplusplus
    }
#endif

}

#endif
