#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>

#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"
#include <omp.h>

using namespace pogs;

template <typename T>
T getRMSE(const T *v1, std::vector<T> *v2) {
  T rmse = 0;
  size_t len = v2->size();
  for (size_t i = 0; i < len; ++i) {
    T d = v1[i] - (*v2)[i];
    rmse += d*d;
  }
  rmse /= (T)len;
  return std::sqrt(rmse);
}

template <typename T>
double getSD(std::vector<T>& v, T mean) {
  double sd = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    sd += (v[i]-mean) * (v[i]-mean);
  }
  return sqrt(sd/v.size()-1);
}

// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double ElasticNet(size_t m, size_t n, int nGPUs, int nLambdas, int nAlphas, double validFraction) {
  int nlambda = nLambdas;
  if (nlambda <= 1) {
    fprintf(stderr, "Must use nlambda > 1\n");
    exit(-1);
  }
  if (static_cast<size_t>(validFraction*m) <= 1) {
    fprintf(stderr, "Must use larger validFraction\n");
    exit(-1);
  }
  // number of openmp threads = number of cuda devices to use


#ifdef _OPENMP
  int nopenmpthreads0=omp_get_max_threads();
#define MIN(a,b) ((a)<(b)?(a):(b))
  omp_set_num_threads(MIN(nopenmpthreads0,nGPUs));
  int nopenmpthreads=omp_get_max_threads();
  nGPUs = nopenmpthreads; // openmp threads = cuda devices used
  fprintf(stdout,"Number of original threads=%d.  Number of threads for cuda=%d\n",nopenmpthreads0,nopenmpthreads);


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

    fprintf(stdout, "START FILL DATA\n");
    fflush(stdout);
    double t0 = timer<double>();

#include "readorgen.c"

    double t1 = timer<double>();
    fprintf(stdout, "END FILL DATA. Took %g secs\n", t1-t0);
    fflush(stdout);

    fprintf(stdout, "START TRAIN/VALID SPLIT\n");
    fflush(stdout);
    // Split A/b into train/valid, via head/tail
    size_t mValid = static_cast<size_t>(m * validFraction);
    mTrain = m - mValid;

    // Alloc
    trainX.resize(mTrain * n);
    validX.resize(mValid * n);
    trainY.resize(mTrain);
    validY.resize(mValid);
    for (int i = 0; i < mTrain; ++i) { //rows
      trainY[i] = b[i];
      for (int j = 0; j < n; ++j) { //cols
        trainX[i * n + j] = A[i * n + j];
      }
    }
    for (int i = 0; i < mValid; ++i) { //rows
      validY[i] = b[mTrain + i];
      for (int j = 0; j < n; ++j) { //cols
        validX[i * n + j] = A[(mTrain + i) * n + j];
      }
    }
    fprintf(stdout, "END TRAIN/VALID SPLIT\n");
    fflush(stdout);
  }

  fprintf(stdout, "Rows in training data: %d\n", (int)trainY.size());
  fprintf(stdout, "Rows in validation data: %d\n", (int)validY.size());


  // Training mean and stddev
  T meanTrainY = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
  fprintf(stdout,"Mean trainY: %f\n", meanTrainY);
  fprintf(stdout,"StDev trainY: %f\n", getSD(trainY, meanTrainY));
  for (size_t i=0; i<trainY.size(); ++i) trainY[i] -= meanTrainY;
  meanTrainY = std::accumulate(begin(trainY), end(trainY), T(0)) / trainY.size();
  fprintf(stdout,"Mean trainY: %f\n", meanTrainY);
  fprintf(stdout,"StDev trainY: %f\n", getSD(trainY, meanTrainY));

  // Validation mean and stddev //TODO: refactor
  T meanValidY = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
  fprintf(stdout,"Mean validY: %f\n", meanValidY);
  fprintf(stdout,"StDev validY: %f\n", getSD(validY, meanValidY));
  for (size_t i=0; i<validY.size(); ++i) validY[i] -= meanValidY;
  meanValidY = std::accumulate(begin(validY), end(validY), T(0)) / validY.size();
  fprintf(stdout,"Mean validY: %f\n", meanValidY);
  fprintf(stdout,"StDev validY: %f\n", getSD(validY, meanValidY));

  T weights = static_cast<T>(1.0/(static_cast<T>(m))); // like pogs.R
  fprintf(stdout,"weights %f\n", weights);

  // set lambda max 0 (i.e. base lambda_max)
  T lambda_max0 = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < mTrain; ++i) {
      u += trainX[i + j * mTrain] * (trainY[i] - meanTrainY);
    }
    //lambda_max0 = weights * static_cast<T>(std::max(lambda_max0, std::abs(u)));
    lambda_max0 = std::max(lambda_max0, std::abs(u));
  }
  fprintf(stdout,"lambda_max0 %f\n", lambda_max0);
  // set lambda_min_ratio
  T lambda_min_ratio = 1e-7; //(m<n ? static_cast<T>(0.01) : static_cast<T>(0.0001));
  fprintf(stdout,"lambda_min_ratio %f\n", lambda_min_ratio);

//#ifdef HAVECUDA
//  // warm-up GPUs
//  extern int warmstart(int N, int nGPUs);
//  warmstart(1000000,nGPUs);
//#endif
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
      fprintf(stderr,"Cannot open filename=%s\n",filename);
      exit(0);
    }

    t0 = timer<double>();
    fprintf(fil,"Moving data to the GPU. Starting at %21.15g\n", t0);
    fflush(fil);
    // create class objects that creates cuda memory, cpu memory, etc.
    pogs::MatrixDense<T> A_(me, 'r', mTrain, n, trainX.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(me, A_);
    t1 = timer<double>();
    fprintf(fil,"Done moving data to the GPU. Stopping at %21.15g\n", t1);
    fprintf(fil,"Done moving data to the GPU. Took %g secs\n", t1-t0);
    fflush(fil);

    pogs_data.SetnDev(1); // set how many cuda devices to use internally in pogs
    //pogs_data.SetRelTol(1e-4); // set how many cuda devices to use internally in pogs
    //pogs_data.SetAdaptiveRho(false); // trying
    //pogs_data.SetEquil(false); // trying
    //pogs_data.SetRho(1E-4);
    //pogs_data.SetVerbose(4);
    //pogs_data.SetMaxIter(1u);

    int N=nAlphas; // number of alpha's
    if(N % nGPUs!=0){
      fprintf(stderr,"NOTE: Number of alpha's not evenly divisible by number of GPUs, so not efficint use of GPUs.\n"); fflush(stderr);
    }
    fprintf(fil,"BEGIN SOLVE\n");
    fflush(fil);
    int a;
#pragma omp for
    for (a = 0; a < N; ++a) { //alpha search
      const T alpha = static_cast<T>(a)/static_cast<T>(N>1 ? N-1 : 1);
      T lambda_max = 10*lambda_max0;///(alpha+static_cast<T>(1e-3f)); // actual lambda_max like pogs.R
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
      T penalty_factor = static_cast<T>(1.0); // like pogs.R
      for (unsigned int j = 0; j < mTrain; ++j) f.emplace_back(kSquare, 1.0, trainY[j]);//, weights); // pogs.R
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
        fprintf(fil, "c/e: %f %f\n", g[0].c, g[0].e);

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

        double trainRMSE = getRMSE(pogs_data.GetY(), &trainY);
        std::vector<T> validPreds(validY.size());
        for (size_t i=0; i<validY.size(); ++i) {
          validPreds[i]=meanValidY;
          for (size_t j=0; j<n; ++j) {
            validPreds[i]+=pogs_data.GetX()[j]*validX[i*n+j];
          }
        }
        double validRMSE = getRMSE(&validPreds[0], &validY);
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

template double ElasticNet<double>(size_t m, size_t n, int, int, int, double);
template double ElasticNet<float>(size_t m, size_t n, int, int, int, double);

