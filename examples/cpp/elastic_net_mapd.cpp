#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>

#include "../common/elastic_net_mapd.h"
#include "timer.h"

using namespace std;

template<typename T>
void fillData(std::vector<T>& trainX, std::vector<T>& trainY,
              std::vector<T>& validX, std::vector<T>& validY,
              size_t m, size_t n, size_t mValid, int intercept) {
// allocate matrix problem to solve
  std::vector <T> A(m * n);
  std::vector <T> b(m);

  cout << "START FILL DATA\n" << endl;
  double t0 = timer<double>();

// choose to generate or read-in data
  int generate = 0;

#include "readorgen.c"

  double t1 = timer<double>();
  cout << "END FILL DATA. Took " << t1 - t0 << " secs" << endl;

  cout << "START TRAIN/VALID SPLIT" << endl;
// Split A/b into train/valid, via head/tail
  size_t mTrain = m - mValid;

// If intercept == 1, add one extra column at the end, all constant 1s
  n += intercept;

// Alloc
  trainX.resize(mTrain * n);
  trainY.resize(mTrain);

  for (int i = 0; i < mTrain; ++i) { //rows
    trainY[i] = b[i];
//      cout << "y[" << i << "] = " << b[i] << endl;
    for (int j = 0; j < n - intercept; ++j) { //cols
      trainX[i * n + j] = A[i * (n-intercept) + j];
//        cout << "X[" << i << ", " << j << "] = " << A[i*n+j] << endl;
    }
    if (intercept) {
      trainX[i * n + n - 1] = 1;
    }
  }
  if (mValid > 0) {
    validX.resize(mValid * n);
    validY.resize(mValid);
    for (int i = 0; i < mValid; ++i) { //rows
      validY[i] = b[mTrain + i];
      for (int j = 0; j < n - intercept; ++j) { //cols
        validX[i * n + j] = A[(mTrain + i) * (n-intercept) + j];
      }
      if (intercept) {
        validX[i * n + n - 1] = 1;
      }
    }
  }
  cout << "END TRAIN/VALID SPLIT" << endl;
  fflush(stdout);
}

// m and n are full data set size before splitting
template <typename T>
double ElasticNet(size_t m, size_t n, int nGPUs, int nLambdas, int nAlphas, int intercept, int standardize, double validFraction) {
  if (validFraction<0 or validFraction>=1) {
    cerr << "validFraction must be in [0, 1)\n";
    exit(-1);
  }
  if (intercept!=0 and intercept!=1) {
    cerr << "intercept must be a boolean: 0 or 1\n";
    exit(-1);
  }

  // read data and do train-valid split
  std::vector<T> trainX, trainY, validX, validY;
  size_t mValid = static_cast<size_t>(m * validFraction);
  size_t mTrain = m - mValid;
  fillData(trainX, trainY, validX, validY, m, n, mValid, intercept);
  n+=intercept;
  cout << "Rows in training data: " << trainY.size() << endl;

//  // DEBUG START
//  for (int i=0;i<m;++i) {
//    for (int j=0;j<n;++j) {
//      cout << trainX[i*n+j] << " ";
//    }
//    cout << " -> " << trainY[i] << "\n";
//    cout << "\n";
//  }
//  // DEBUG END

  // Training mean and stddev
  T meanTrainY0 = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
  T sdTrainY0 = std::sqrt(pogs::getVarV(trainY, meanTrainY0));
  T meanTrainYn;
  T sdTrainYn;
  cout << "Mean trainY: " << meanTrainY0 << endl;
  cout << "StdDev trainY: " << sdTrainY0 << endl;
  if(standardize){
    // standardize the response for training data
    for (size_t i=0; i<trainY.size(); ++i) {
      trainY[i] -= meanTrainY0;
      trainY[i] /= sdTrainY0;
    }
    meanTrainYn = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
    sdTrainYn = std::sqrt(pogs::getVarV(trainY, meanTrainYn));
  }

  // Validation mean and stddev
  T meanValidY0;
  T sdValidY0;
  T meanValidYn;
  T sdValidYn;
  if (!validY.empty()) {
    cout << "Rows in validation data: " << validY.size() << endl;
    meanValidY0 = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
    sdValidY0 = std::sqrt(pogs::getVarV(validY, meanValidY0));
    cout << "Mean validY: " << meanValidY0 << endl;
    cout << "StdDev validY: " << sdValidY0 << endl;
    if (standardize) {
      // standardize the response the same way as for training data ("apply fitted transform during scoring")
      for (size_t i=0; i<validY.size(); ++i) {
        validY[i] -= meanValidY0;
        validY[i] /= sdValidY0;
      }
      meanValidYn = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
      sdValidYn = std::sqrt(pogs::getVarV(validY, meanValidYn));
      cout << "new Mean validY: " << meanValidYn << endl;
      cout << "new StdDev validY: " << sdValidYn << endl;
    }
  }
    

  // TODO: compute on the GPU - inside of ElasticNetPtr
  // set lambda max 0 (i.e. base lambda_max)
  T lambda_max0 = static_cast<T>(0);
  for (unsigned int j = 0; j < n-intercept; ++j) { //col
    T u = 0;
    T weights = static_cast<T>(1.0); //TODO: Add per-obs weights
    if (intercept) weights/=mTrain;
    for (unsigned int i = 0; i < mTrain; ++i) { //row
      u += weights * trainX[i * n + j] * (trainY[i] - intercept*meanTrainYn);
    }
    lambda_max0 = static_cast<T>(std::max(lambda_max0, std::abs(u)));
  }
  cout << "lambda_max0 " << lambda_max0 << endl;
  // set lambda_min_ratio
  T lambda_min_ratio = 1E-5; //(m<n ? static_cast<T>(0.01) : static_cast<T>(0.0001));
  cout << "lambda_min_ratio " << lambda_min_ratio << endl;

#define DOWARMSTART 0 // leads to poor usage of GPUs even on local 4 GPU system (all 4 at about 30-50%).  Really bad on AWS 16 GPU system.  // But, if terminate program, disable these, then pogs runs normally at high GPU usage.  So these leave the device in a bad state.
#define DOP2PCHECK 0 // This doesn't seem to lead to any difference in 4 GPU system.  It's not nccl, only cuda.
#define DOBWCHECK 0

#if(DOWARMSTART)
#ifdef HAVECUDA
  // warm-up GPUs
  extern int warmstart(int N, int nGPUs);
  warmstart(1000000,nGPUs);
#endif
#endif

#if(DOP2PCHECK)
#ifdef HAVECUDA
  // do peer-to-peer bandwidth check
  extern int p2pbwcheck(void);
  p2pbwcheck();
#endif
#endif

#if(DOBWCHECK)
#ifdef HAVECUDA
  // do bandwidth check
  extern int bwcheck(void);
  bwcheck();
#endif
#endif
  // Push data from CPU to GPU
  void* a;
  void* b;
  void* c;
  void* d;
  int sourceDev=0; //index of first GPU to own data
  pogs::makePtr(sourceDev, mTrain, n, mValid, trainX.data(), trainY.data(), validX.data(), validY.data(), &a, &b, &c, &d);

  int datatype = 1;
  return pogs::ElasticNetptr<T>(sourceDev, datatype, nGPUs, 'r', mTrain, n, mValid, intercept, standardize, lambda_max0, lambda_min_ratio, nLambdas, nAlphas, sdTrainY0, meanTrainY0, a, b, c, d);
}

template double ElasticNet<double>(size_t m, size_t n, int, int, int, int, int, double);
template double ElasticNet<float>(size_t m, size_t n, int, int, int, int, int, double);

