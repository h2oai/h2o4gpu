#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <random>
#include <vector>
#include "reader.h"
#include "matrix/matrix_dense.h"
#include "h2ogpuml.h"
#include "timer.h"

using namespace h2ogpuml;

template <typename T>
T MaxDiff(std::vector<T> *v1, std::vector<T> *v2) {
  T max_diff = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max : max_diff)
#endif
  for (unsigned int i = 0; i < v1->size(); ++i)
    max_diff = std::max(max_diff, std::abs((*v1)[i] - (*v2)[i]));
  return max_diff;
}

template <typename T>
T Asum(std::vector<T> *v) {
  T asum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : asum)
#endif
  for (unsigned int i = 0; i < v->size(); ++i)
    asum += std::abs((*v)[i]);
  return asum;
}

// LassoPath
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// for 100 values of \lambda.
// See <h2ogpuml>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double LassoPath(size_t m, size_t n) {
  unsigned int nlambda = 100;
  std::vector<T> A(m * n);
  std::vector<T> b(m);
  std::vector<T> w(m);
  std::vector<T> x_last(n, std::numeric_limits<T>::max());


  fprintf(stdout,"BEGIN FILL DATA\n");
  double t0 = timer<double>();
  int generate=0;
  fillData(m, n, "train.txt", &A[0], &b[0], &w[0]);
  double t1 = timer<double>();
  fprintf(stdout,"END FILL DATA\n");


  
  T lambda_max = static_cast<T>(0);
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      //u += A[i * n + j] * b[i];
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  // Set up h2ogpuml datastructure.
  h2ogpuml::MatrixDense<T> A_('r', m, n, A.data());
  h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > h2ogpuml_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;
  f.reserve(m);
  g.reserve(n);

  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs);

  // Set up h2ogpuml datastructure.
  h2ogpuml::MatrixDense<T> A2_('r', m, n, A.data());
  h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > h2ogpuml_data2(A2_);
  std::vector<FunctionObj<T> > f2;
  std::vector<FunctionObj<T> > g2;
  f2.reserve(m);
  g2.reserve(n);

  for (unsigned int i = 0; i < m; ++i)
    f2.emplace_back(kSquare, static_cast<T>(1), b[i]);

  for (unsigned int i = 0; i < n; ++i)
    g2.emplace_back(kAbs);




  

  fprintf(stdout,"BEGIN SOLVE\n");
  double t = timer<double>();
    // starts at lambda_max and goes down to 1E-2 lambda_max in exponential spacing
  for (unsigned int i = 0; i < nlambda; ++i) {
    T lambda = std::exp((std::log(lambda_max) * (nlambda - 1 - i) +
			 static_cast<T>(1e-2) * std::log(lambda_max) * i) / (nlambda - 1));

    if(i%2==0){
      for (unsigned int i = 0; i < n; ++i) g[i].c = lambda;
      h2ogpuml_data.Solve(f, g);
    }
    else{
      for (unsigned int i = 0; i < n; ++i) g2[i].c = lambda;
      h2ogpuml_data2.Solve(f2, g2);
    }
  }




  double tf = timer<double>();
  fprintf(stdout,"END SOLVE: type 1 m %d n %d tfd %g ts %g\n",m,n,t1-t0,tf-t);

  return tf-t;
}

template double LassoPath<double>(size_t m, size_t n);
template double LassoPath<float>(size_t m, size_t n);

