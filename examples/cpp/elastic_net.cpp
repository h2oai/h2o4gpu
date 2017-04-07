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

// Elastic Net
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda \alpha ||x||_1 + \lambda 1-\alpha ||x||_2
//
// for many values of \lambda and multiple values of \alpha
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double ElasticNet(size_t m, size_t n, int nGPUs, int nLambdas, int nAlphas) {
  int nlambda = nLambdas;
  if (nlambda <= 1) {
    fprintf(stderr,"Must use nlambda > 1\n");
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

  // allocate matrix problem to solve
  std::vector<T> A(m * n);
  std::vector<T> b(m);

  fprintf(stdout,"START FILL DATA\n");
  double t0 = timer<double>();
#include "readorgen.c"
  double t1 = timer<double>();
  fprintf(stdout,"END FILL DATA\n");

  double mean_y = 0;
  for (unsigned int i = 0; i < m; ++i) {
    mean_y += b[i];
  }
  mean_y /= (double)m;
  const T mean_b = static_cast<T>(mean_y);
  fprintf(stdout,"Mean b: %f\n", mean_b);
  T weights = static_cast<T>(1.0/(static_cast<T>(m))); // like pogs.R
  fprintf(stdout,"weights %f\n", weights);

  // set lambda max 0 (i.e. base lambda_max)
  //T lambda_max0 = static_cast<T>(0);
  T lambda_max = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i) {
      //u += A[i * n + j] * b[i];
      //u += A[i + j * m] * (b[i] - mean_b);
      u += A[i + j * m] * b[i];
    }
    //lambda_max0 = weights * static_cast<T>(std::max(lambda_max0, std::abs(u)));
    lambda_max = std::max(lambda_max, std::abs(u));
  }
  //fprintf(stdout,"lambda_max0 %f\n", lambda_max0);
  fprintf(stdout,"lambda_max %f\n", lambda_max);
  // set lambda_min_ratio
  T lambda_min_ratio = 1e-2f; //(m<n ? static_cast<T>(0.01) : static_cast<T>(0.001)); // like pogs.R
  fprintf(stdout,"lambda_min_ratio %f\n", lambda_min_ratio);


  double t = timer<double>();
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

    double t0 = timer<double>();
    fprintf(fil,"Moving data to the GPU. Starting at %21.15g\n", t0);
    // create class objects that creates cuda memory, cpu memory, etc.
    pogs::MatrixDense<T> A_(me, 'r', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(me, A_);
    double t1 = timer<double>();
    fprintf(fil,"Done moving data to the GPU. Stopping at %21.15g\n", t1);
    fprintf(fil,"Done moving data to the GPU. Took %g secs\n", t1-t0);

    pogs_data.SetnDev(1); // set how many cuda devices to use internally in pogs
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

      //const T lambda_max = lambda_max0/(alpha+static_cast<T>(1e-3f)); // actual lambda_max like pogs.R
      // set lambda_min
      const T lambda_min = lambda_min_ratio * static_cast<T>(lambda_max); // like pogs.R
      fprintf(fil, "lambda_max: %f\n", lambda_max);
      fprintf(fil, "lambda_min: %f\n", lambda_min);
      fflush(fil);

      // setup f,g as functions of alpha
      std::vector<FunctionObj<T> > f;
      std::vector<FunctionObj<T> > g;
      f.reserve(m);
      g.reserve(n);
      // minimize ||Ax-b||_2^2 + \alpha\lambda||x||_1 + (1/2)(1-alpha)*lambda x^2
      T penalty_factor = static_cast<T>(1.0); // like pogs.R
      for (unsigned int j = 0; j < m; ++j) f.emplace_back(kSquare, 1.0, b[j]);//, weights); // pogs.R
      for (unsigned int j = 0; j < n; ++j) g.emplace_back(kAbs);

      fprintf(fil,"alpha%f\n", alpha);
      for (int i = 0; i < nlambda; ++i){

        // starts at lambda_max and goes down to 1E-2 lambda_max in exponential spacing
        //T lambda = std::exp((std::log(lambda_max) * ((float)nlambda - 1.0f - (float)i) + lambda_min * (float)i) / ((float)nlambda - 1.0f));
        T lambda = std::exp((std::log(lambda_max) * ((float)nlambda - 1.0f - (float)i) +
                             static_cast<T>(1e-2) * std::log(lambda_max) * (float)i) / ((float)nlambda - 1.0f));
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

        fprintf(fil,"me=%d a=%d alpha=%g i=%d lambda=%g trainRMSE: %f\n",me,a,alpha,i,lambda,getRMSE(pogs_data.GetY(), &b));fflush(fil);
      }// over lambda
    }// over alpha
    if(fil!=NULL) fclose(fil);
  } // end parallel region

  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 1 m %d n %d tfd %g ts %g\n",(int)m,(int)n,t1-t0,tf-t);

  return tf-t;
}

template double ElasticNet<double>(size_t m, size_t n, int, int, int);
template double ElasticNet<float>(size_t m, size_t n, int, int, int);

