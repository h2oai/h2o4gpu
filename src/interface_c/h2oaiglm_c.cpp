#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "h2oaiglm.h"
#include "h2oaiglm_c.h"
#include <iostream>   //std::cout

#include <nvToolsExt.h>

bool VerifyPogsWork(void * work){
  if (!work) { return false; }
  PogsWork * p_work = static_cast<PogsWork *>(work);
  if (!(p_work->h2oaiglm_data) || !(p_work->f) || !(p_work->g)){ return false; }
  else { return true; } 
}

//Dense Direct
template <typename T>
void * PogsInit(int wDev, size_t m, size_t n, const T *A, const char ord){
    
    bool directbit = true, densebit = true, rowmajorbit = ord == 'r';
    // bool directbit = true, densebit = true, rowmajorbit = O == ROW_MAJ;

    int sharedA=0; // force for now
    // char ord = rowmajorbit ? 'r' : 'c';
    h2oaiglm::MatrixDense<T> A_(sharedA,wDev,ord,m,n,A);
    h2oaiglm::PogsDirect<T,h2oaiglm::MatrixDense<T> > *h2oaiglm_data;    
    std::vector<FunctionObj<T> > *f, *g;
    PogsWork * work;



    // create h2oaiglm function vectors
    f = new std::vector<FunctionObj<T> >;
    g = new std::vector<FunctionObj<T> >;


    f->reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   
 
    g->reserve(n);
    for (unsigned int j = 0; j < n; ++j)
      g->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   


    //create new h2oaiglm_data object
    h2oaiglm_data = new h2oaiglm::PogsDirect<T,h2oaiglm::MatrixDense<T> >(wDev, A_);

    // create new h2oaiglm work struct
    work = new PogsWork(m,n,directbit,densebit,rowmajorbit, static_cast<void *>(h2oaiglm_data), static_cast<void *>(f), static_cast<void *>(g));


    return static_cast<void *>(work);

}

//Sparse Indirect
template <typename T>
void * PogsInit(int wDev, size_t m, size_t n, size_t nnz, const T *nzvals, const int *nzindices, const int *pointers, const char ord){
    
    // bool directbit = false, densebit = false, rowmajorbit = O == ROW_MAJ;
    bool directbit = false, densebit = false, rowmajorbit = ord == 'r';

    // char ord = rowmajorbit ? 'r' : 'c';
    h2oaiglm::MatrixSparse<T> A_(wDev, ord, static_cast<int>(m), static_cast<int>(n), static_cast<int>(nnz), nzvals, pointers, nzindices);
    h2oaiglm::PogsIndirect<T,h2oaiglm::MatrixSparse<T> > *h2oaiglm_data;    
    std::vector<FunctionObj<T> > *f, *g;
    PogsWork * work;


    // create h2oaiglm function vectors
    f = new std::vector<FunctionObj<T> >;
    g = new std::vector<FunctionObj<T> >;

    f->reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   
 
    g->reserve(n);
    for (unsigned int j = 0; j < n; ++j)
      g->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   

    //create h2oaiglm_data object
    h2oaiglm_data = new h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> >(wDev, A_);

    // create new h2oaiglm work struct
    work = new PogsWork(m,n,directbit,densebit,rowmajorbit, static_cast<void *>(h2oaiglm_data), static_cast<void *>(f), static_cast<void *>(g));
    return static_cast<void *>(work);
}


template <typename T>
void PogsFunctionUpdate(size_t m, std::vector<FunctionObj<T> > *f, const T *f_a, const T *f_b, const T *f_c, 
                            const T *f_d, const T *f_e, const FUNCTION *f_h){


  for (unsigned int i = 0; i < m; ++i)
    f->at(i).a= f_a[i];
  for (unsigned int i = 0; i < m; ++i)
    f->at(i).b= f_b[i];
  for (unsigned int i = 0; i < m; ++i)
    f->at(i).c= f_c[i];
  for (unsigned int i = 0; i < m; ++i)
    f->at(i).d= f_d[i];
  for (unsigned int i = 0; i < m; ++i)
    f->at(i).e= f_e[i];
  for (unsigned int i = 0; i < m; ++i)
    f->at(i).h= static_cast<Function>(f_h[i]);
}

template <typename T>
void PogsRun(h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > &h2oaiglm_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, \
              const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution){


  // Set parameters.
  h2oaiglm_data.SetRho(settings->rho);
  h2oaiglm_data.SetAbsTol(settings->abs_tol);
  h2oaiglm_data.SetRelTol(settings->rel_tol);
  h2oaiglm_data.SetMaxIter(settings->max_iters);
  h2oaiglm_data.SetVerbose(settings->verbose);
  h2oaiglm_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2oaiglm_data.SetEquil(static_cast<bool>(settings->equil));
  h2oaiglm_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2oaiglm_data.SetnDev(static_cast<int>(settings->nDev));
  h2oaiglm_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2oaiglm_data.SetInitX(solution->x);
    h2oaiglm_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2oaiglm_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);
  
  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,2);
  info->obj = h2oaiglm_data.GetOptval();
  info->iter = h2oaiglm_data.GetFinalIter();
  info->rho = h2oaiglm_data.GetRho();
  info->solvetime = h2oaiglm_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2oaiglm_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2oaiglm_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2oaiglm_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2oaiglm_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,2);
}

template<typename T>
void PogsRun(h2oaiglm::PogsDirect<T, h2oaiglm::MatrixSparse<T> > &h2oaiglm_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, \
                const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution){
  // Set parameters.
  h2oaiglm_data.SetRho(settings->rho);
  h2oaiglm_data.SetAbsTol(settings->abs_tol);
  h2oaiglm_data.SetRelTol(settings->rel_tol);
  h2oaiglm_data.SetMaxIter(settings->max_iters);
  h2oaiglm_data.SetVerbose(settings->verbose);
  h2oaiglm_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2oaiglm_data.SetEquil(static_cast<bool>(settings->equil));
  h2oaiglm_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2oaiglm_data.SetnDev(static_cast<int>(settings->nDev));
  h2oaiglm_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2oaiglm_data.SetInitX(solution->x);
    h2oaiglm_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2oaiglm_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);

  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,2);
  info->obj = h2oaiglm_data.GetOptval();
  info->iter = h2oaiglm_data.GetFinalIter();
  info->rho = h2oaiglm_data.GetRho();
  info->solvetime = h2oaiglm_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2oaiglm_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2oaiglm_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2oaiglm_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2oaiglm_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,2);
}

template<typename T>
void PogsRun(h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixDense<T> > &h2oaiglm_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution){
  // Set parameters.
  h2oaiglm_data.SetRho(settings->rho);
  h2oaiglm_data.SetAbsTol(settings->abs_tol);
  h2oaiglm_data.SetRelTol(settings->rel_tol);
  h2oaiglm_data.SetMaxIter(settings->max_iters);
  h2oaiglm_data.SetVerbose(settings->verbose);
  h2oaiglm_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2oaiglm_data.SetEquil(static_cast<bool>(settings->equil));
  h2oaiglm_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2oaiglm_data.SetnDev(static_cast<int>(settings->nDev));
  h2oaiglm_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2oaiglm_data.SetInitX(solution->x);
    h2oaiglm_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2oaiglm_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);

  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,2);
  info->obj = h2oaiglm_data.GetOptval();
  info->iter = h2oaiglm_data.GetFinalIter();
  info->rho = h2oaiglm_data.GetRho();
  info->solvetime = h2oaiglm_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2oaiglm_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2oaiglm_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2oaiglm_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2oaiglm_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,2);
}

template<typename T>
void PogsRun(h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> > &h2oaiglm_data, const std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution){
  // Set parameters.
  h2oaiglm_data.SetRho(settings->rho);
  h2oaiglm_data.SetAbsTol(settings->abs_tol);
  h2oaiglm_data.SetRelTol(settings->rel_tol);
  h2oaiglm_data.SetMaxIter(settings->max_iters);
  h2oaiglm_data.SetVerbose(settings->verbose);
  h2oaiglm_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2oaiglm_data.SetEquil(static_cast<bool>(settings->equil));
  h2oaiglm_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2oaiglm_data.SetnDev(static_cast<int>(settings->nDev));
  h2oaiglm_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2oaiglm_data.SetInitX(solution->x);
    h2oaiglm_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2oaiglm_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);

  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,1);
  info->obj = h2oaiglm_data.GetOptval();
  info->iter = h2oaiglm_data.GetFinalIter();
  info->rho = h2oaiglm_data.GetRho();
  info->solvetime = h2oaiglm_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2oaiglm_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2oaiglm_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2oaiglm_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2oaiglm_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,1);
}


template <typename T>
int PogsRun(void *work, const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e, const FUNCTION *f_h,
            const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e, const FUNCTION *g_h,
            void *settings_, void *info_, void *solution_){

  if (!VerifyPogsWork(work)) { return static_cast <int>(H2OAIGLM_ERROR); }


  const PogsSettings<T> * settings = static_cast<PogsSettings<T> *>(settings_);
  PogsInfo<T> * info = static_cast<PogsInfo<T> *>(info_);
  PogsSolution<T> * solution = static_cast<PogsSolution<T> *>(solution_);  
  PogsWork * p_work = static_cast<PogsWork *>(work);


  size_t m = p_work->m;
  size_t n = p_work->n;
  std::vector<FunctionObj<T> > *f = static_cast<std::vector<FunctionObj<T> > *>(p_work->f);
  std::vector<FunctionObj<T> > *g = static_cast<std::vector<FunctionObj<T> > *>(p_work->g);

  // Update f and g
  PogsFunctionUpdate(m, f, f_a, f_b, f_c, f_d, f_e, f_h);
  PogsFunctionUpdate(n, g, g_a, g_b, g_c, g_d, g_e, g_h);

  // Run
  if (p_work->densebit){
    if (p_work->directbit){
      h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > *h2oaiglm_data = static_cast< h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > *>(p_work->h2oaiglm_data);
      PogsRun(*h2oaiglm_data, f, g, settings, info, solution);
    }else{
      h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixDense<T> > *h2oaiglm_data = static_cast< h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixDense<T> > *>(p_work->h2oaiglm_data);
      PogsRun(*h2oaiglm_data, f, g, settings, info, solution);
    }
  }else{
    h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> > *h2oaiglm_data = static_cast< h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> > *>(p_work->h2oaiglm_data);
    PogsRun(*h2oaiglm_data, f, g, settings, info, solution);      
  }
  return info->status;
} 

template<typename T>
void PogsShutdown(void * work){
  PogsWork * p_work = static_cast<PogsWork *>(work);

  std::vector<FunctionObj<T> > *f, *g;
  f = static_cast<std::vector<FunctionObj<T> > *>(p_work->f);
  g = static_cast<std::vector<FunctionObj<T> > *>(p_work->g);

  delete f;
  delete g;

  // Run
  if (p_work->densebit){
    if (p_work->directbit){
      h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > *h2oaiglm_data = static_cast< h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > *>(p_work->h2oaiglm_data);
      delete h2oaiglm_data;
    }else{
      h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixDense<T> > *h2oaiglm_data = static_cast< h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixDense<T> > *>(p_work->h2oaiglm_data);
      delete h2oaiglm_data;
    }
  }else{
    h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> > *h2oaiglm_data = static_cast< h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> > *>(p_work->h2oaiglm_data);
    delete h2oaiglm_data;
  }

  delete p_work;
}


extern "C" {


void * h2oaiglm_init_dense_single(int wDev, enum ORD ord, size_t m, size_t n, const float *A){
    return ord == COL_MAJ ? PogsInit<float>(wDev, m,n,A,'c') : PogsInit<float>(wDev, m,n,A,'r');
}
void * h2oaiglm_init_dense_double(int wDev, enum ORD ord, size_t m, size_t n, const double *A){
    return ord == COL_MAJ ? PogsInit<double>(wDev, m,n,A,'c') : PogsInit<double>(wDev, m,n,A,'r');
}
void * h2oaiglm_init_sparse_single(int wDev, enum ORD ord, size_t m, size_t n, size_t nnz, const float *nzvals, const int *indices, const int *pointers){
  return ord == COL_MAJ ? PogsInit<float>(wDev, m,n,nnz,nzvals,indices,pointers,'c') : PogsInit<float>(wDev, m,n,nnz,nzvals,indices,pointers,'r');


}
void * h2oaiglm_init_sparse_double(int wDev, enum ORD ord, size_t m, size_t n, size_t nnz, const double *nzvals, const int *indices, const int *pointers){
  return ord == COL_MAJ ? PogsInit<double>(wDev, m,n,nnz,nzvals,indices,pointers,'c') : PogsInit<double>(wDev, m,n,nnz,nzvals,indices,pointers,'r');
}

int h2oaiglm_solve_single(void *work, PogsSettingsS *settings, PogsSolutionS *solution, PogsInfoS *info,
                      const float *f_a, const float *f_b, const float *f_c,const float *f_d, const float *f_e, const enum FUNCTION *f_h,
                      const float *g_a, const float *g_b, const float *g_c,const float *g_d, const float *g_e, const enum FUNCTION *g_h){
  return PogsRun<float>(work, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h, settings, info, solution);
}
int h2oaiglm_solve_double(void *work, PogsSettingsD *settings, PogsSolutionD *solution, PogsInfoD *info,
                      const double *f_a, const double *f_b, const double *f_c,const double *f_d, const double *f_e, const enum FUNCTION *f_h,
                      const double *g_a, const double *g_b, const double *g_c,const double *g_d, const double *g_e, const enum FUNCTION *g_h){
  return PogsRun<double>(work, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h, settings, info, solution);
}

void h2oaiglm_finish_single(void * work){ return PogsShutdown<float>(work); }
void h2oaiglm_finish_double(void * work){ return PogsShutdown<double>(work); }

}


