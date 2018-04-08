#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include "reader.h"
#include <random>
#include <stdlib.h>
#include "../../src/include/matrix/matrix_dense.h"
#include "timer.h"
#include "../../src/common/elastic_net_ptr.h"

using namespace std;


template<typename T>
T getVarV(std::vector <T> &v, T mean) {
  double var = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    var += (v[i] - mean) * (v[i] - mean);
  }
  return static_cast<T>(var / (v.size() - 1));
}



// m and n are full data set size before splitting
template <typename T>
double ElasticNet(const std::vector<T>&A, const std::vector<T>&b, const std::vector<T>&w, int sharedA, int nThreads, int nGPUs, int nLambdas, int nFolds, int nAlphas, int intercept, int standardize, double validFraction) {
  if (sharedA<0 or sharedA>2) {
    cerr << "sharedA must be in [0,2]\n";
    exit(-1);
  }
  if (nThreads<1) {
    cerr << "nThreads must be in [1,\\infty]\n";
    exit(-1);
  }
  if (nGPUs<0 or nGPUs>nThreads) {
    cerr << "nGPUs must be in [0,nThreads]\n";
    exit(-1);
  }
  if (nLambdas<2) {
    cerr << "nLambdas must be in [2,\\infty]\n";
    exit(-1);
  }
  if (nAlphas<1) {
    cerr << "nAlphas must be in [1,\\infty]\n";
    exit(-1);
  }
  if (intercept!=0 and intercept!=1) {
    cerr << "intercept must be a boolean: 0 or 1\n";
    exit(-1);
  }
  if (standardize!=0 and standardize!=1) {
    cerr << "standardize must be a boolean: 0 or 1\n";
    exit(-1);
  }
  if (validFraction<0 or validFraction>=1) {
    cerr << "validFraction must be in [0, 1)\n";
    exit(-1);
  }

  system("rm -f rmse.txt ; touch rmse.txt ; rm -f varimp.txt ; touch varimp.txt");

  // read data and do train-valid split
  std::vector<T> trainX, trainY, trainW, validX, validY, validW;
  splitData(A, b, w, trainX, trainY, trainW, validX, validY, validW, validFraction, intercept);
  size_t mTrain = trainY.size();
  size_t mValid = validY.size();
  size_t n=trainX.size()/mTrain;
  cout << "Rows in training data: " << mTrain << endl;
  cout << "Rows in validation data: " << mValid << endl;
  
  if (nFolds<0 or nFolds>mTrain) {
    cerr << "nFolds must be in [0,mTrain]\n";
    exit(-1);
  }
  cout << "Cols in training data: " << n << endl;
  
  // set weights
  if(0){
    for(unsigned int i=0; i<mTrain;i++) trainW[i]/=static_cast<T>(mTrain);
    for(unsigned int i=0; i<mValid;i++) validW[i]/=static_cast<T>(mValid);
  }
  else{ // FIXME: should be able to scale by constant weights and be same result/outcome, but currently appears to solve very different problem.
    for(unsigned int i=0; i<mTrain;i++) trainW[i]/=1.0;
    for(unsigned int i=0; i<mValid;i++) validW[i]/=1.0;
  }

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
  T sdTrainY0 = std::sqrt(getVarV(trainY, meanTrainY0));
  T meanTrainYn = meanTrainY0;
  T sdTrainYn = sdTrainY0;
  cout << "Mean trainY: " << meanTrainY0 << endl;
  cout << "StdDev trainY: " << sdTrainY0 << endl;
  if(standardize){
    // standardize the response for training data
    for (size_t i=0; i<trainY.size(); ++i) {
      trainY[i] -= meanTrainY0;
      trainY[i] /= sdTrainY0;
    }
    meanTrainYn = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
    sdTrainYn = std::sqrt(getVarV(trainY, meanTrainYn));
  }

  // Validation mean and stddev
  T meanValidY0 = meanTrainY0;
  T sdValidY0 = sdTrainY0;
  T meanValidYn = meanValidY0;
  T sdValidYn = sdValidY0;
  if (!validY.empty()) {
    meanValidY0 = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
    sdValidY0 = std::sqrt(getVarV(validY, meanValidY0));
    cout << "Rows in validation data: " << validY.size() << endl;
    cout << "Mean validY: " << meanValidY0 << endl;
    cout << "StdDev validY: " << sdValidY0 << endl;
    if (standardize) {
      // standardize the response the same way as for training data ("apply fitted transform during scoring")
      for (size_t i=0; i<validY.size(); ++i) {
        validY[i] -= meanTrainY0;
        validY[i] /= sdTrainY0;
      }
    }
    meanValidYn = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
    sdValidYn = std::sqrt(getVarV(validY, meanValidYn));
  }

  T alpha_min = 0.0;
  T alpha_max = 1.0;

  T lambda_max = 0; // let algo set this
  // set lambda_min_ratio
  T lambda_min_ratio = 1E-9; //(m<n ? static_cast<T>(0.01) : static_cast<T>(0.0001));
  cout << "lambda_min_ratio " << lambda_min_ratio << endl;

#define DOWARMSTART 0 // leads to poor usage of GPUs even on local 4 GPU system (all 4 at about 30-50%).  Really bad on AWS 16 GPU system.  // But, if terminate program, disable these, then h2o4gpu runs normally at high GPU usage.  So these leave the device in a bad state.
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
  T* aa;
  T* bb;
  T* cc;
  T* dd;
  T* ee;
  int sourceme=0; // index of first thread to own data
  int sourceDev=0; //index of first GPU to own data
  const char ord='r'; // normal C-order
  // only need train weight
  //  extern int makePtr_dense<T>(int sharedA, int me, int wDev, size_t m, size_t n, size_t mValid, const char ord,
  //                           const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight,
  //                           void **_data, void **_datay, void **_vdata, void **_vdatay, void **_weight);
  h2o4gpu::makePtr_dense(sharedA, sourceme, sourceDev, mTrain, n, mValid, ord, trainX.data(), trainY.data(), validX.data(), validY.data(), trainW.data(), &aa, &bb, &cc, &dd, &ee); // //static_cast<T*>(NULL)


  int datatype = 1;
  int givefullpath=0;
  T *Xvsalphalambda=NULL;
  T *Xvsalpha=NULL;
  T *validPredsvsalphalambda=NULL;
  T *validPredsvsalpha=NULL;
  size_t countfull=0;
  size_t countshort=0;
  size_t countmore=0;
  int dopredict=0;
  const char family='e';
  double tol = 1E-2;
  double tolseekfactor = 1E-1;
  int lambdastopearly=1;
  int glmstopearly=1;
  double glmstopearlyrmsefraction=1.0;
  int maxiterations=5000;
  int verbose=0;
  T *alphas = NULL;
  T *lambdas = NULL;
  int gpu_id = 0;
  int totalnGPUs = nGPUs; // not really right TODO: Should have elasticNetptr figure out total number of GPUs
  double time = h2o4gpu::ElasticNetptr<T>(family, dopredict, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs, ord, mTrain, n, mValid, intercept, standardize, lambda_max, lambda_min_ratio, nLambdas, nFolds, nAlphas, alpha_min, alpha_max, alphas, lambdas, tol, tolseekfactor, lambdastopearly, glmstopearly, glmstopearlyrmsefraction, maxiterations, verbose, aa, bb, cc, dd, ee, givefullpath, &Xvsalphalambda, &Xvsalpha, &validPredsvsalphalambda, &validPredsvsalpha, &countfull, &countshort, &countmore);

  // print out some things about Xvsalphalambda and Xvsalpha
  printf("countfull=%d countshort=%d countmore=%d\n",countfull,countshort,countmore); fflush(stdout);

  // print out some things about validPredsvsalphalambda and validPredsvsalpha

  // do some prediction, say on validX already have

  return(time);

  
  
}

template double ElasticNet<double>(const std::vector<double>&A, const std::vector<double>&b, const std::vector<double>&w, int, int, int, int, int, int, int, int, double);
template double ElasticNet<float>(const std::vector<float>&A, const std::vector<float>&b, const std::vector<float>&w, int, int, int, int, int, int, int, int, double);



