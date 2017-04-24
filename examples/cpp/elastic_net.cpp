#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include "reader.h"
#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if(USEMKL==1)
#include <mkl.h>
#endif

using namespace pogs;
using namespace std;

template <typename T>
T getRMSE(const T *v1, std::vector<T> *v2) {
  double rmse = 0;
  size_t len = v2->size();
  for (size_t i = 0; i < len; ++i) {
    double d = v1[i] - (*v2)[i];
    rmse += d*d;
  }
  rmse /= (double)len;
  return static_cast<T>(std::sqrt(rmse));
}

template <typename T>
T getVar(std::vector<T>& v, T mean) {
  double var = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    var += (v[i]-mean) * (v[i]-mean);
  }
  return static_cast<T>(var/(v.size()-1));
}

// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double ElasticNet(const std::vector<T>&A, const std::vector<T>&b, int sharedA, int nThreads, int nGPUs, int nLambdas, int nAlphas, int intercept, int standardize, double validFraction) {
  int nlambda = nLambdas;
  if (nlambda <= 1) {
    cerr << "Must use nlambda > 1\n";
    exit(-1);
  }
  // number of openmp threads = number of cuda devices to use


#ifdef _OPENMP
  int omt=omp_get_max_threads();
#define MIN(a,b) ((a)<(b)?(a):(b))
  omp_set_num_threads(MIN(omt,nGPUs)); // not necessary, but most useful mode so far
  int nth=omp_get_max_threads();
  nGPUs=nth; // openmp threads = cuda devices used
#ifdef DEBUG
  cout << "Number of original threads=" << omt << " Number of threads for cuda=" << nth << endl;
#endif

  if(nAlphas % nGPUs!=0){
    fprintf(stderr,"NOTE: Number of alpha's not evenly divisible by number of GPUs, so not efficint use of GPUs.\n"); fflush(stderr);
  }
#endif


  // read data and do train-valid split
  std::vector<T> trainX, trainY, validX, validY;
  splitData(A, b, trainX, trainY, validX, validY, validFraction, intercept);
  size_t m = trainY.size();
  size_t mTrain = trainY.size();
  size_t mValid = validY.size();
  size_t n=trainX.size()/mTrain;
  cout << "Rows in training data: " << mTrain << endl;
  cout << "Rows in validation data: " << mValid << endl;
  cout << "Cols in training data: " << n << endl;

  // Training mean and stddev
  T meanTrainY0 = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
  T sdTrainY0 = std::sqrt(getVar(trainY, meanTrainY0));
  T meanTrainYn = meanTrainY0;
  T sdTrainYn = sdTrainY0;
  cout << "Mean trainY: " << meanTrainY0 << endl;
  cout << "StdDev trainY: " << sdTrainY0 << endl;
  // standardize the response for training data
  if(standardize){
    for (size_t i=0; i<trainY.size(); ++i) {
      trainY[i] -= meanTrainY0;
      trainY[i] /= sdTrainY0;
    }
    meanTrainYn = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
    sdTrainYn = std::sqrt(getVar(trainY, meanTrainYn));
    cout << "new Mean trainY: " << meanTrainYn << endl;
    cout << "new StdDev trainY: " << sdTrainYn << endl;
  }


  // Validation mean and stddev
  if (!validY.empty()) {
    cout << "Rows in validation data: " << validY.size() << endl;
    T meanValidY0 = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
    T sdValidY0 = std::sqrt(getVar(validY, meanValidY0));
    T meanValidYn = meanValidY0;
    T sdValidYn = sdValidY0;
    cout << "Mean validY: " << meanValidY0 << endl;
    cout << "StdDev validY: " << sdValidY0 << endl;

    if(standardize){
      // standardize the response the same way as for training data ("apply fitted transform during scoring")
      for (size_t i=0; i<validY.size(); ++i) {
        validY[i] -= meanTrainY0;
        validY[i] /= sdTrainY0;
      }
      meanValidYn = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
      sdValidYn = std::sqrt(getVar(validY, meanValidYn));
      cout << "new Mean validY: " << meanValidYn << endl;
      cout << "new StdDev validY: " << sdValidYn << endl;
    }
  }

  // set lambda max 0 (i.e. base lambda_max)
  T lambda_max0 = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    T weights = static_cast<T>(1.0/mTrain); //TODO: Add per-obs weights
    for (unsigned int i = 0; i < mTrain; ++i) {
      u += weights * trainX[i * n + j] * (trainY[i] - intercept*meanTrainYn);
    }
    lambda_max0 = std::max(lambda_max0, std::abs(u));
  }
  cout << "lambda_max0 " << lambda_max0 << endl;
  // set lambda_min_ratio
  T lambda_min_ratio = 1E-9; //(m<n ? static_cast<T>(0.01) : static_cast<T>(0.0001));
  cout << "lambda_min_ratio " << lambda_min_ratio << endl;


  


  // for source, create class objects that creates cuda memory, cpu memory, etc.
  int sourceDev=0;
  pogs::MatrixDense<T> Asource_(sharedA, sourceDev, 'r', mTrain, n, trainX.data());
  // now can always access A_(sourceDev) to get pointer from within other MatrixDense calls
  



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
  
/*
  fprintf(stdout,"waiting for AWS GPUs to go live.");
  fflush(stdout);
  sleep(200);
  fprintf(stdout,"AWS ready - starting with GLM.");
  fflush(stdout);
*/


  
  double t = timer<double>();
#pragma omp parallel proc_bind(master)
  {
#ifdef _OPENMP
    int me = omp_get_thread_num();
#if(USEMKL==1)
    //https://software.intel.com/en-us/node/522115
    int physicalcores=omt;///2; // asssume hyperthreading Intel processor (doens't improve much to ensure physical cores used0
    // set number of mkl threads per openmp thread so that not oversubscribing cores
    int mklperthread=max(1,(physicalcores % nThreads==0 ? physicalcores/nThreads : physicalcores/nThreads+1));
    //mkl_set_num_threads_local(mklperthread);
    mkl_set_num_threads_local(mklperthread);
    //But see (hyperthreading threads not good for MKL): https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/288645
#endif
#else
    int me=0;
#endif
    
    // choose GPU device ID for each thread
    int wDev = me%nGPUs;

    char filename[100];
    sprintf(filename,"me%d.txt",me);
    FILE * fil = fopen(filename,"wt");
    if(fil==NULL){
      cerr << "Cannot open filename=" << filename << endl;
      exit(0);
    }

    double t0 = timer<double>();
    fprintf(fil,"Moving data to the GPU. Starting at %21.15g\n", t0);
    fflush(fil);
    // create class objects that creates cuda memory, cpu memory, etc.
#pragma omp barrier
    pogs::MatrixDense<T> A_(sharedA, me, wDev, Asource_); // not setup for nThread!=nGPUs
#pragma omp barrier // this is the required barrier
    pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(wDev, A_);
#pragma omp barrier
    double t1 = timer<double>();
    fprintf(fil,"Done moving data to the GPU. Stopping at %21.15g\n", t1);
    fprintf(fil,"Done moving data to the GPU. Took %g secs\n", t1-t0);
    fflush(fil);

    pogs_data.SetnDev(1); // set how many cuda devices to use internally in pogs
//    pogs_data.SetRelTol(1e-4); // set how many cuda devices to use internally in pogs
//    pogs_data.SetAbsTol(1e-4); // set how many cuda devices to use internally in pogs
    //pogs_data.SetAdaptiveRho(false); // trying
    //pogs_data.SetEquil(false); // trying
    //pogs_data.SetRho(1E-4);
    //    pogs_data.SetVerbose(5);
    //pogs_data.SetMaxIter(200);

    fprintf(fil,"BEGIN SOLVE\n");
    fflush(fil);
    int a;
#pragma omp for
    for (a = 0; a < nAlphas; ++a) { //alpha search
      const T alpha = nAlphas == 1 ? 1 : static_cast<T>(a)/static_cast<T>(nAlphas>1 ? nAlphas-1 : 1);
      T lambda_max = lambda_max0/std::max(static_cast<T>(1e-2), alpha); // same as H2O
      if (alpha==1 && mTrain>10000) {
        lambda_max *= 2;
        lambda_min_ratio /= 2;
      }
      const T lambda_min = lambda_min_ratio * static_cast<T>(lambda_max); // like pogs.R
      fprintf(fil, "lambda_max: %f\n", lambda_max);
      fprintf(fil, "lambda_min: %f\n", lambda_min);
      fflush(fil);

      // setup f,g as functions of alpha
      std::vector<FunctionObj<T> > f;
      std::vector<FunctionObj<T> > g;
      f.reserve(mTrain);
      g.reserve(n);
      // minimize ||Ax-b||_2^2 + \alpha\lambda||x||_1 + (1/2)(1-alpha)*lambda x^2
      T weights = static_cast<T>(1.0/(static_cast<T>(mTrain))); // like pogs.R
      T penalty_factor = static_cast<T>(1.0); // like pogs.R
      for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kSquare, 1.0, trainY[j], weights); // pogs.R
      for (unsigned int j = 0; j < n; ++j) g.emplace_back(kAbs);

      fprintf(fil,"alpha%f\n", alpha);
      for (int i = 0; i < nlambda; ++i){

        // starts at lambda_max and goes down to 1E-2 lambda_max in exponential spacing
        //T lambda = std::exp((std::log(lambda_max) * ((float)nlambda - 1.0f - (float)i) + lambda_min * (float)i) / ((float)nlambda - 1.0f));
        T lambda = std::exp((std::log(lambda_max) * ((float)nlambda - 1.0f - (float)i) +
                             std::log(lambda_min) * (float)i) / ((float)nlambda - 1.0f));
        fprintf(fil,"lambda %d = %f\n", i, lambda);

        // assign lambda
        for (unsigned int j = 0; j < n; ++j) {
          g[j].c = static_cast<T>(alpha*lambda*penalty_factor); //for L1
          g[j].e = static_cast<T>((1.0-alpha)*lambda*penalty_factor); //for L2
        }

        // Solve
        fprintf(fil, "Starting to solve at %21.15g\n", timer<double>());
        fflush(fil);
        pogs_data.Solve(f, g);

        size_t dof = 0;
        for (size_t i=0; i<n; ++i) {
          if (std::abs(pogs_data.GetX()[i]) > 1e-8) {
            dof++;
          }
        }

        std::vector<T> trainPreds(trainY.size());
        for (size_t i=0; i<trainY.size(); ++i) {
          for (size_t j=0; j<n; ++j) {
            trainPreds[i]+=pogs_data.GetX()[j]*trainX[i*n+j]; //add predictions
          }
        }
        // score
        double trainRMSE;
        if(standardize) trainRMSE=sdTrainY0*getRMSE(&trainPreds[0], &trainY);
        else trainRMSE=getRMSE(&trainPreds[0], &trainY);

        if(standardize){
          for (size_t i=0; i<trainY.size(); ++i) {
            // reverse standardization
            trainPreds[i]*=sdTrainY0; //scale
            trainPreds[i]+=meanTrainY0; //intercept
            //assert(trainPreds[i] == pogs_data.GetY()[i]); //FIXME: CHECK
          }
        }
          
        double validRMSE = -1;
        if (!validY.empty()) {
          std::vector<T> validPreds(validY.size());
          for (size_t i=0; i<validY.size(); ++i) {
            for (size_t j=0; j<n; ++j) {
              validPreds[i]+=pogs_data.GetX()[j]*validX[i*n+j]; //add predictions
            }
          }
          if(standardize) validRMSE = sdTrainY0*getRMSE(&validPreds[0], &validY);
          else validRMSE = getRMSE(&validPreds[0], &validY);
          if(standardize){
            for (size_t i=0; i<validY.size(); ++i) {
              // reverse (fitted) standardization
              validPreds[i]*=sdTrainY0; //scale
              validPreds[i]+=meanTrainY0; //intercept
            }
          }
        }
        fprintf(fil,   "me: %d a: %d alpha: %g i: %d lambda: %g dof: %d trainRMSE: %f validRMSE: %f\n",me,a,alpha,i,lambda,dof,trainRMSE,validRMSE);fflush(fil);
        fprintf(stdout,"me: %d a: %d alpha: %g i: %d lambda: %g dof: %d trainRMSE: %f validRMSE: %f\n",me,a,alpha,i,lambda,dof,trainRMSE,validRMSE);fflush(stdout);
      }// over lambda
    }// over alpha
    if(fil!=NULL) fclose(fil);
  } // end parallel region

  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 1 m %d n %d twall %g\n",(int)mTrain,(int)n,tf-t);
  return tf-t;
}
  
template double ElasticNet<double>(const std::vector<double> &A, const std::vector<double> &b, int, int, int, int, int, int, int, double);
template double ElasticNet<float>(const std::vector<float> &A, const std::vector<float> &b, int, int, int, int, int, int, int, double);
 
