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
double ElasticNet(size_t m, size_t n, int nGPUs, int nLambdas, int nAlphas, int intercept, int standardize, double validFraction) {

  int nlambda = nLambdas;
  if (nlambda <= 1) {
    cerr << "Must use nlambda > 1\n";
    exit(-1);
  }
  // number of openmp threads = number of cuda devices to use


#ifdef _OPENMP
  int omt=omp_get_max_threads();
#define MIN(a,b) ((a)<(b)?(a):(b))
  omp_set_num_threads(MIN(omt,nGPUs));
  int nth=omp_get_max_threads();
  nGPUs=nth; // openmp threads = cuda devices used
  cout << "Number of original threads=" << omt << " Number of threads for cuda=" << nth << endl;

#else
#error Need OpenMP
#endif

  std::vector <T> trainX;
  std::vector <T> validX;
  std::vector <T> trainY;
  std::vector <T> validY;
  size_t mTrain = 0;
  {
    // allocate matrix problem to solve
    std::vector <T> A(m * n);
    std::vector <T> b(m);

    cout << "START FILL DATA\n" << endl;
    double t0 = timer<double>();

    // choose to generate or read-in data
    int generate=0;    
#include "readorgen.c"

    double t1 = timer<double>();
    cout << "END FILL DATA. Took " << t1-t0 << " secs" << endl;

    cout << "START TRAIN/VALID SPLIT" << endl;
    // Split A/b into train/valid, via head/tail
    size_t mValid = static_cast<size_t>(m * validFraction);
    mTrain = m - mValid;

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
    if (mValid>0) {
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
  cout << "Rows in training data: " << trainY.size() << endl;

  // Training mean and stddev
  T meanTrainY = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
  T sdTrainY = std::sqrt(getVar(trainY, meanTrainY));
  cout << "Mean trainY: " << meanTrainY << endl;
  cout << "StdDev trainY: " << sdTrainY << endl;
  // standardize the response for training data
  if(standardize){
    for (size_t i=0; i<trainY.size(); ++i) {
      trainY[i] -= meanTrainY;
      trainY[i] /= sdTrainY;
    }
    meanTrainY = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
    sdTrainY = std::sqrt(getVar(trainY, meanTrainY));
  }


  // Validation mean and stddev
  if (!validY.empty()) {
    cout << "Rows in validation data: " << validY.size() << endl;
    T meanValidY = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
    cout << "Mean validY: " << meanValidY << endl;
    cout << "StdDev validY: " << std::sqrt(getVar(validY, meanValidY)) << endl;

    if(standardize){
      // standardize the response the same way as for training data ("apply fitted transform during scoring")
      for (size_t i=0; i<validY.size(); ++i) {
        validY[i] -= meanTrainY;
        validY[i] /= sdTrainY;
      }
      meanValidY = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
      cout << "Mean validY: " << meanValidY << endl;
      cout << "StdDev validY: " << std::sqrt(getVar(validY, meanValidY)) << endl;
    }
  }

  T weights = static_cast<T>(1.0/(static_cast<T>(m))); // like pogs.R
  cout << "weights " << weights << endl;

  // set lambda max 0 (i.e. base lambda_max)
  T lambda_max0 = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    T weights = static_cast<T>(1.0); //TODO: Add per-obs weights
    if (intercept) weights/=mTrain;
    for (unsigned int i = 0; i < mTrain; ++i) {
      u += weights * trainX[i * n + j] * (trainY[i] - intercept*meanTrainY);
    }
    lambda_max0 = std::max(lambda_max0, std::abs(u));
  }
  cout << "lambda_max0 " << lambda_max0 << endl;
  // set lambda_min_ratio
  T lambda_min_ratio = 1E-5; //(m<n ? static_cast<T>(0.01) : static_cast<T>(0.0001));
  cout << "lambda_min_ratio " << lambda_min_ratio << endl;


  


  // for source, create class objects that creates cuda memory, cpu memory, etc.
  int sourceDev=0;
  pogs::MatrixDense<T> Asource_(sourceDev, 'r', mTrain, n, trainX.data());
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
  double t0 = 0;
  double t1 = 0;
#pragma omp parallel 
  {
    int me = omp_get_thread_num();

    char filename[100];
    sprintf(filename,"me%d.txt",me);
    FILE * fil = fopen(filename,"wt");
    if(fil==NULL){
      cerr << "Cannot open filename=" << filename << endl;
      exit(0);
    }

    t0 = timer<double>();
    fprintf(fil,"Moving data to the GPU. Starting at %21.15g\n", t0);
    fflush(fil);
    // create class objects that creates cuda memory, cpu memory, etc.
    pogs::MatrixDense<T> A_(me, Asource_);
    pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(me, A_);
    t1 = timer<double>();
    fprintf(fil,"Done moving data to the GPU. Stopping at %21.15g\n", t1);
    fprintf(fil,"Done moving data to the GPU. Took %g secs\n", t1-t0);
    fflush(fil);

    pogs_data.SetnDev(1); // set how many cuda devices to use internally in pogs
//    pogs_data.SetRelTol(1e-4); // set how many cuda devices to use internally in pogs
//    pogs_data.SetAbsTol(1e-4); // set how many cuda devices to use internally in pogs
    //pogs_data.SetAdaptiveRho(false); // trying
    //pogs_data.SetEquil(false); // trying
    //pogs_data.SetRho(1E-4);
    pogs_data.SetVerbose(5);
    //pogs_data.SetMaxIter(200);

    int N=nAlphas; // number of alpha's
    if(N % nGPUs!=0){
      fprintf(stderr,"NOTE: Number of alpha's not evenly divisible by number of GPUs, so not efficint use of GPUs.\n"); fflush(stderr);
    }
    fprintf(fil,"BEGIN SOLVE\n");
    fflush(fil);
    int a;
#pragma omp for
    for (a = 0; a < N; ++a) { //alpha search
      const T alpha = N == 1 ? 0.5 : static_cast<T>(a)/static_cast<T>(N>1 ? N-1 : 1);
      T lambda_max = lambda_max0/std::max(static_cast<T>(1e-3), alpha); // same as H2O
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
          if(standardize){
            // reverse standardization
            trainPreds[i]*=sdTrainY; //scale
            trainPreds[i]+=meanTrainY; //intercept
            //assert(trainPreds[i] == pogs_data.GetY()[i]); //FIXME: CHECK
          }
        }
        double trainRMSE = getRMSE(&trainPreds[0], &trainY);

        double validRMSE = -1;
        if (!validY.empty()) {
          std::vector<T> validPreds(validY.size());
          for (size_t i=0; i<validY.size(); ++i) {
            for (size_t j=0; j<n; ++j) {
              validPreds[i]+=pogs_data.GetX()[j]*validX[i*n+j]; //add predictions
            }
            if(standardize){
              // reverse (fitted) standardization
              validPreds[i]*=sdTrainY; //scale
              validPreds[i]+=meanTrainY; //intercept
            }
          }
          validRMSE = getRMSE(&validPreds[0], &validY);
        }
        fprintf(fil,   "me: %d a: %d alpha: %g i: %d lambda: %g dof: %d trainRMSE: %f validRMSE: %f\n",me,a,alpha,i,lambda,dof,trainRMSE,validRMSE);fflush(fil);
        fprintf(stdout,"me: %d a: %d alpha: %g i: %d lambda: %g dof: %d trainRMSE: %f validRMSE: %f\n",me,a,alpha,i,lambda,dof,trainRMSE,validRMSE);fflush(stdout);
      }// over lambda
    }// over alpha
    if(fil!=NULL) fclose(fil);
  } // end parallel region

  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 1 m %d n %d twall %g tsolve %g\n",(int)mTrain,(int)n,tf-t,tf-t1);
  return tf-t1;
}
  
template double ElasticNet<double>(size_t m, size_t n, int, int, int, int, int, double);
template double ElasticNet<float>(size_t m, size_t n, int, int, int, int, int, double);
 
