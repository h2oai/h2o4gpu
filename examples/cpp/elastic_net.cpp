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
double LassoPath(size_t m, size_t n) {
  int nlambda = 50;
  // number of openmp threads = number of cuda devices to use
  int nGPUs=1;

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


  // set lambda max
  T lambda_max = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      //u += A[i * n + j] * b[i];
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  double t = timer<double>();
#pragma omp parallel 
{
  int me = omp_get_thread_num();
  // create class objects that creates cuda memory, cpu memory, etc.
  pogs::MatrixDense<T> A_(me, 'r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(me, A_);

  pogs_data.SetnDev(2); // set how many cuda devices to use internally in pogs
  //pogs_data.SetAdaptiveRho(false); // trying
  //pogs_data.SetEquil(false); // trying
  //pogs_data.SetRho(1E-4);
  //pogs_data.SetVerbose(4);
  //pogs_data.SetMaxIter(1u);


  int N=2;
  fprintf(stdout,"BEGIN SOLVE\n"); 
#pragma omp parallel for
  for (int a = 0; a <= N; ++a) { //alpha search
    double alpha = (double)a/N;

    // setup f,g as functions of alpha
    std::vector<FunctionObj<T> > f;
    std::vector<FunctionObj<T> > g;
    f.reserve(m);
    g.reserve(n);
    for (unsigned int j = 0; j < m; ++j) f.emplace_back(kSquare, 1.0, b[j]);
    for (unsigned int j = 0; j < n; ++j) g.emplace_back(kAbs);

    for (int i = 0; i < nlambda; ++i){

      // starts at lambda_max and goes down to 1E-2 lambda_max in exponential spacing
      T lambda = std::exp((std::log(lambda_max) * ((float)nlambda - 1.0f - (float)i) +
            static_cast<T>(1e-2) * std::log(lambda_max) * (float)i) / ((float)nlambda - 1.0f));

      fprintf(stderr,"me=%d a=%d alpha=%g i=%d lambda=%g\n",me,a,alpha,i,lambda);

      // assign lambda
      for (unsigned int j = 0; j < n; ++j) {
        g[j].c = static_cast<float>(alpha*lambda); //for L1
        g[j].e = static_cast<float>((1-alpha)*lambda); //for L2
      }

      // Solve
      pogs_data.Solve(f, g);

      fprintf(stderr,"train RMSE: %f\n", getRMSE(pogs_data.GetY(), &b));
    }// over lambda
  }// over alpha
}

  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 1 m %d n %d tfd %g ts %g\n",(int)m,(int)n,t1-t0,tf-t);

  return tf-t;
}

template double LassoPath<double>(size_t m, size_t n);
template double LassoPath<float>(size_t m, size_t n);

