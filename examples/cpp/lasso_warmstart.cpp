#include <random>
#include <vector>

#include "pogs.h"
#include "timer.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
double LassoWarmstart(size_t m, size_t n) {
  std::vector<T> A(m * n);
  std::vector<T> b(m);
  std::vector<T> x(n);
  std::vector<T> y(m);
  std::vector<T> mu(n);
  std::vector<T> nu(m);
  std::vector<T> x12(n);
  std::vector<T> y12(m);
  std::vector<T> mu12(n);
  std::vector<T> nu12(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = n_dist(generator);

  std::vector<T> x_true(n);
  for (unsigned int i = 0; i < n; ++i)
    x_true[i] = u_dist(generator) < 0.8 ? 0 : n_dist(generator) / n;

#pragma omp parallel for
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * x_true[j];
      // b[i] += A[i + j * m] * x_true[j];

  for (unsigned int i = 0; i < m; ++i)
    b[i] += static_cast<T>(0.5) * n_dist(generator);

  T lambda_max = static_cast<T>(0);
#pragma omp parallel for reduction(max : lambda_max)
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      //u += A[i * n + j] * b[i];
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  Dense<T, ROW> A_(A.data());
  PogsData<T, Dense<T, ROW>> pogs_data(A_, m, n);

  pogs_data.x = x.data();
  pogs_data.y = y.data();
  pogs_data.nu = nu.data();
  pogs_data.mu = mu.data();  
  pogs_data.x12 = x12.data();
  pogs_data.y12 = y12.data();
  pogs_data.mu12 = mu12.data();
  pogs_data.nu12 = nu12.data();

  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);

  double t = timer<double>();

  AllocDenseFactors(&pogs_data);
  Printf("FIRST RUN\n");
  Pogs(&pogs_data);
  Printf("rho final=%0.2f\n",pogs_data.rho);

  Printf("SECOND RUN, warm start by continuation\n");
  Pogs(&pogs_data);
  Printf("rho final=%0.2f\n",pogs_data.rho);

  Printf("THIRD RUN, warm start by variable feed\n");
  pogs_data.warm_start=true;
  Pogs(&pogs_data);

  Printf("rho final=%0.2f\n",pogs_data.rho);

  std::vector<T> x_(n);
  std::vector<T> y_(m);
  std::vector<T> mu_(n);
  std::vector<T> nu_(m);
  std::vector<T> x12_(n);
  std::vector<T> y12_(m);
  std::vector<T> mu12_(n);
  std::vector<T> nu12_(m);

  // build clone
  PogsData<T, Dense<T, ROW>> pogs_clone(A_,m,n);
  pogs_clone.x = x_.data();
  pogs_clone.y = y_.data();
  pogs_clone.nu = nu_.data();
  pogs_clone.mu = mu_.data();  
  pogs_clone.x12 = x12_.data();
  pogs_clone.y12 = y12_.data();
  pogs_clone.mu12 = mu12_.data();
  pogs_clone.nu12 = nu12_.data();

  pogs_clone.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_clone.f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  pogs_clone.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_clone.g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);

  // run clone
  AllocDenseFactors(&pogs_clone);  
  Printf("FOURTH RUN, cold start clone with previous rho, rho=%0.2f\n",pogs_data.rho);  
  pogs_clone.rho=pogs_data.rho;
  pogs_clone.warm_start=false;
  Pogs(&pogs_clone);  

  // std::vector<T> x_(n);
  // std::vector<T> y_(m);
  // std::vector<T> mu_(n);
  // std::vector<T> nu_(m);
  // std::vector<T> x12_(n);
  // std::vector<T> y12_(m);
  // std::vector<T> mu12_(n);
  // std::vector<T> nu12_(m);

  // build clone
  PogsData<T, Dense<T, ROW>> pogs_clone2(A_,m,n);
  pogs_clone2.x = x_.data();
  pogs_clone2.y = y_.data();
  pogs_clone2.nu = nu_.data();
  pogs_clone2.mu = mu_.data();
  pogs_clone2.x12 = x12_.data();
  pogs_clone2.y12 = y12_.data();
  pogs_clone2.mu12 = mu12_.data();
  pogs_clone2.nu12 = nu12_.data();

  pogs_clone2.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_clone2.f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  pogs_clone2.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_clone2.g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);

  // run clone
  AllocDenseFactors(&pogs_clone2);
  Printf("FIFTH RUN, warm start clone with previous rho and primal/dual vars, rho=%0.2f\n",pogs_data.rho);  
  pogs_clone2.rho=pogs_data.rho;
  pogs_clone2.warm_start=true;
  Pogs(&pogs_clone2);



  FreeDenseFactors(&pogs_data);
  FreeDenseFactors(&pogs_clone);
  FreeDenseFactors(&pogs_clone2);



  return timer<double>() - t;
}

template double LassoWarmstart<double>(size_t m, size_t n);
template double LassoWarmstart<float>(size_t m, size_t n);

