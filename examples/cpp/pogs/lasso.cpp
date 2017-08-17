#include <stdio.h>
#include <random>
#include "matrix/matrix_dense.h"
#include "h2ogpumlglm.h"
#include "timer.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <h2ogpuml>/matlab/examples/lasso.m for detailed description.
template <typename T>
double Lasso(const std::vector<T>& A, const std::vector<T>& b) {
  size_t m=b.size();
  size_t n=A.size()/m;

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
  // setup h2ogpuml
  ////////////////////
  fprintf(stderr,"MatrixDense\n"); fflush(stderr);
  h2ogpuml::MatrixDense<T> A_('r', m, n, A.data());
  fprintf(stderr,"H2OGPUMLDirect\n"); fflush(stderr);
  h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > h2ogpuml_data(A_);
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
  // Solve with h2ogpuml
  ////////////////////
  fprintf(stdout,"BEGIN SOLVE\n");
  if(0==1){ // debug
    //    h2ogpuml_data.SetAdaptiveRho(false); // trying
    //    h2ogpuml_data.SetRho(1.0);
    //  h2ogpuml_data.SetMaxIter(5u);
    //    h2ogpuml_data.SetMaxIter(1u);
    //    h2ogpuml_data.SetVerbose(4);
  }
  else if(1==1){ // debug
    //    h2ogpuml_data.SetAdaptiveRho(false); // trying
    //    h2ogpuml_data.SetEquil(false); // trying
    //    h2ogpuml_data.SetRho(1E-4);
    //    fprintf(stderr,"sets\n"); fflush(stderr);
    //    h2ogpuml_data.SetVerbose(4);
    //    h2ogpuml_data.SetMaxIter(5u);
  }
  else if(1==0){
    //    h2ogpuml_data.SetVerbose(4);
  }
#ifdef __CUDACC__
  //  cudaProfilerStart();
#endif
  fprintf(stderr,"Solve\n"); fflush(stderr);
  h2ogpuml_data.Solve(f, g);
#ifdef __CUDACC__
  //  cudaProfilerStop();
#endif
  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 0 m %d n %d ts %g\n",(int)m,(int)n,tf-t);

  return tf-t;
}

template double Lasso<double>(const std::vector<double>& A, const std::vector<double>& b);
template double Lasso<float>(const std::vector<float>& A, const std::vector<float>& b);

