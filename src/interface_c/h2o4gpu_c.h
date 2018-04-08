/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#pragma once

#include <stddef.h>
#include <omp.h>
#include <vector>
#include "h2o4gpu_c_api.h"

// TODO: arg for projector? or do we leave dense-direct and sparse-indirect pairings fixed?
// TODO: get primal and dual variables and rho
// TODO: set primal and dual variables and rho
// TODO: methods for warmstart with x,nu,rho,options
// TODO: pass in void *
// TODO: h2o4gpu shutdown
// TODO: julia finalizer

// Wrapper for H2O4GPU, a solver for convex problems in the form
//   min. \sum_i f(y_i) + g(x_i)
//   s.t.  y = Ax,
//  where 
//   f_i(y_i) = c_i * h_i(a_i * y_i - b_i) + d_i * y_i + e_i * x_i^2,
//   g_i(x_i) = c_i * h_i(a_i * x_i - b_i) + d_i * x_i + e_i * x_i^2.
//
// Input arguments (real_t is either double or float)
// - ORD ord           : Specifies row/colum major ordering of matrix A.
// - size_t m          : First dimensions of matrix A.
// - size_t n          : Second dimensions of matrix A.
// - real_t *A         : Pointer to matrix A.
// - real_t *f_a-f_e   : Pointer to array of a_i-e_i's in function f_i(y_i).
// - FUNCTION *f_h     : Pointer to array of h_i's in function f_i(y_i).
// - real_t *g_a-g_e   : Pointer to array of a_i-e_i's in function g_i(x_i).
// - FUNCTION *g_h     : Pointer to array of h_i's in function g_i(x_i).
// - real_t rho        : Initial value for rho parameter.
// - real_t abs_tol    : Absolute tolerance (recommended 1e-4).
// - real_t rel_tol    : Relative tolerance (recommended 1e-3).
// - uint max_iter     : Maximum number of iterations (recommended 1e3-2e3).
// - int quiet         : Output to screen if quiet = 0.
// - int adaptive_rho  : No adaptive rho update if adaptive_rho = 0.
// - int equil         : No equilibration if equil = 0.
// - int gap_stop      : Additionally use the gap as a stopping criteria.
// - int nDev          : Choose number of cuda devices
// - int wDev          : Choose which cuda device(s)
//
// Output arguments (real_t is either double or float)
// - real_t *x         : Array for solution vector x.
// - real_t *y         : Array for solution vector y.
// - real_t *nu        : Array for dual vector nu.
// - real_t *optval    : Pointer to single real for f(y^*) + g(x^*).
// - uint final_iter   : # of iterations at termination
//
// Author: H2O.ai
//

// created and managed locally
struct H2O4GPUWork{
    size_t m,n;
    bool directbit, densebit, rowmajorbit;
    void *h2o4gpu_data, *f, *g;

    H2O4GPUWork(size_t m_, size_t n_, bool direct_, bool dense_, bool rowmajor_, void *h2o4gpu_data_, void *f_, void *g_){
      m=m_;n=n_;
      directbit=direct_; densebit=dense_; rowmajorbit=rowmajor_;
      h2o4gpu_data= h2o4gpu_data_; f=f_; g=g_;
    }
};


bool VerifyH2O4GPUWork(void * work);

// Dense
template <typename T>
void * H2O4GPUInit(int wDev, size_t m, size_t n, const T *A, const char ord);

// Sparse 
template <typename T>
void * H2O4GPUInit(int wDev, size_t m, size_t n, size_t nnz, const T *nzvals, const int *nzindices, const int *pointers, const char ord);

template <typename T>
void H2O4GPUFunctionUpdate(size_t m, std::vector<FunctionObj<T> > *f, const T *f_a, const T *f_b, const T *f_c, 
                          const T *f_d, const T *f_e, const FUNCTION *f_h);

template <typename T>
void H2O4GPURun(h2o4gpu::H2O4GPUDirect<T, h2o4gpu::MatrixDense<T> > &h2o4gpu_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
  const H2O4GPUSettings<T> *settings, H2O4GPUInfo<T> *info, H2O4GPUSolution<T> *solution);


template <typename T>
void H2O4GPURun(h2o4gpu::H2O4GPUDirect<T, h2o4gpu::MatrixSparse<T> > &h2o4gpu_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const H2O4GPUSettings<T> *settings, H2O4GPUInfo<T> *info, H2O4GPUSolution<T> *solution);

template<typename T>
void H2O4GPURun(h2o4gpu::H2O4GPUIndirect<T, h2o4gpu::MatrixDense<T> > &h2o4gpu_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const H2O4GPUSettings<T> *settings, H2O4GPUInfo<T> *info, H2O4GPUSolution<T> *solution);

template<typename T>
void H2O4GPURun(h2o4gpu::H2O4GPUIndirect<T, h2o4gpu::MatrixSparse<T> > &h2o4gpu_data, const std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const H2O4GPUSettings<T> *settings, H2O4GPUInfo<T> *info, H2O4GPUSolution<T> *solution);

template<typename T>
int H2O4GPURun(void *work, const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e, const FUNCTION *f_h,
            const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e, const FUNCTION *g_h,
            void *settings, void *info, void *solution);

template<typename T>
void H2O4GPUShutdown(void * work);

template <typename T>
int makePtr(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid,
            T* trainX, T * trainY, T* validX, T* validY,  //CPU
            void**a, void**b, void**c, void**d); // GPU