#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
double Lasso(size_t m, size_t n) {
  std::vector<T> A(m * n);
  std::vector<T> b(m);


  fprintf(stdout,"START FILL DATA\n");
  double t0 = timer<double>();
  int generate=0;
#include "readorgen.c"
  double t1 = timer<double>();
  fprintf(stdout,"END FILL DATA\n");
  printf("Time to create data: %f\n", t1-t0);



  ////////////////////
  // set lambda for regularization
  ////////////////////
  T lambda_max = static_cast<T>(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(max : lambda_max)
#endif
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      //u += A[i * n + j] * b[i];
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  ////////////////////
  // setup pogs
  ////////////////////
  fprintf(stderr,"MatrixDense\n"); fflush(stderr);
  pogs::MatrixDense<T> A_('r', m, n, A.data());
  fprintf(stderr,"PogsDirect\n"); fflush(stderr);
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);
  fprintf(stderr,"f\n"); fflush(stderr);
  std::vector<FunctionObj<T> > f;
  fprintf(stderr,"g\n"); fflush(stderr);
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs, static_cast<T>(0.2) * lambda_max);

  double t = timer<double>();


  ////////////////////
  // Solve with pogs
  ////////////////////
  fprintf(stdout,"BEGIN SOLVE\n");
  if(0==1){ // debug
    //    pogs_data.SetAdaptiveRho(false); // trying
    //    pogs_data.SetRho(1.0);
    //  pogs_data.SetMaxIter(5u);
    //    pogs_data.SetMaxIter(1u);
    //    pogs_data.SetVerbose(4);
  }
  else if(1==1){ // debug
    //    pogs_data.SetAdaptiveRho(false); // trying
    //    pogs_data.SetEquil(false); // trying
    //    pogs_data.SetRho(1E-4);
    //    fprintf(stderr,"sets\n"); fflush(stderr);
    //    pogs_data.SetVerbose(4);
    //    pogs_data.SetMaxIter(5u);
  }
  else if(1==0){
    //    pogs_data.SetVerbose(4);
  }
#ifdef __CUDACC__
  //  cudaProfilerStart();
#endif
  fprintf(stderr,"Solve\n"); fflush(stderr);
  pogs_data.Solve(f, g);
#ifdef __CUDACC__
  //  cudaProfilerStop();
#endif
  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 0 m %d n %d tfd %g ts %g\n",(int)m,(int)n,t1-t0,tf-t);

  return tf-t;
}

template double Lasso<double>(size_t m, size_t n);
template double Lasso<float>(size_t m, size_t n);

