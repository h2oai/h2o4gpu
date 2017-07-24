#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "h2ogpumlglm.h"
#include "h2ogpuml_c.h"
#include <iostream>   //std::cout

#include <nvToolsExt.h>

bool VerifyH2OGPUMLWork(void * work){
  if (!work) { return false; }
  H2OGPUMLWork * p_work = static_cast<H2OGPUMLWork *>(work);
  if (!(p_work->h2ogpuml_data) || !(p_work->f) || !(p_work->g)){ return false; }
  else { return true; } 
}

//Dense Direct
template <typename T>
void * H2OGPUMLInit(int wDev, size_t m, size_t n, const T *A, const char ord){
    
    bool directbit = true, densebit = true, rowmajorbit = ord == 'r';
    // bool directbit = true, densebit = true, rowmajorbit = O == ROW_MAJ;

    int sharedA=0; // force for now
    // char ord = rowmajorbit ? 'r' : 'c';
    h2ogpuml::MatrixDense<T> A_(sharedA,wDev,ord,m,n,A);
    h2ogpuml::H2OGPUMLDirect<T,h2ogpuml::MatrixDense<T> > *h2ogpuml_data;    
    std::vector<FunctionObj<T> > *f, *g;
    H2OGPUMLWork * work;



    // create h2ogpuml function vectors
    f = new std::vector<FunctionObj<T> >;
    g = new std::vector<FunctionObj<T> >;


    f->reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   
 
    g->reserve(n);
    for (unsigned int j = 0; j < n; ++j)
      g->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   


    //create new h2ogpuml_data object
    h2ogpuml_data = new h2ogpuml::H2OGPUMLDirect<T,h2ogpuml::MatrixDense<T> >(A_);

    // create new h2ogpuml work struct
    work = new H2OGPUMLWork(m,n,directbit,densebit,rowmajorbit, static_cast<void *>(h2ogpuml_data), static_cast<void *>(f), static_cast<void *>(g));


    return static_cast<void *>(work);

}

//Sparse Indirect
template <typename T>
void * H2OGPUMLInit(int wDev, size_t m, size_t n, size_t nnz, const T *nzvals, const int *nzindices, const int *pointers, const char ord){
    
    // bool directbit = false, densebit = false, rowmajorbit = O == ROW_MAJ;
    bool directbit = false, densebit = false, rowmajorbit = ord == 'r';

    // char ord = rowmajorbit ? 'r' : 'c';
    h2ogpuml::MatrixSparse<T> A_(wDev, ord, static_cast<h2ogpuml::H2OGPUML_INT>(m), static_cast<h2ogpuml::H2OGPUML_INT>(n), static_cast<h2ogpuml::H2OGPUML_INT>(nnz), nzvals, pointers, nzindices);
    h2ogpuml::H2OGPUMLIndirect<T,h2ogpuml::MatrixSparse<T> > *h2ogpuml_data;    
    std::vector<FunctionObj<T> > *f, *g;
    H2OGPUMLWork * work;


    // create h2ogpuml function vectors
    f = new std::vector<FunctionObj<T> >;
    g = new std::vector<FunctionObj<T> >;

    f->reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   
 
    g->reserve(n);
    for (unsigned int j = 0; j < n; ++j)
      g->emplace_back(static_cast<Function>(kZero), static_cast<T>(1), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));   

    //create h2ogpuml_data object
    h2ogpuml_data = new h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T> >(A_);

    // create new h2ogpuml work struct
    work = new H2OGPUMLWork(m,n,directbit,densebit,rowmajorbit, static_cast<void *>(h2ogpuml_data), static_cast<void *>(f), static_cast<void *>(g));
    return static_cast<void *>(work);
}


template <typename T>
void H2OGPUMLFunctionUpdate(size_t m, std::vector<FunctionObj<T> > *f, const T *f_a, const T *f_b, const T *f_c, 
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
void H2OGPUMLRun(h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > &h2ogpuml_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, \
              const H2OGPUMLSettings<T> *settings, H2OGPUMLInfo<T> *info, H2OGPUMLSolution<T> *solution){


  // Set parameters.
  h2ogpuml_data.SetRho(settings->rho);
  h2ogpuml_data.SetAbsTol(settings->abs_tol);
  h2ogpuml_data.SetRelTol(settings->rel_tol);
  h2ogpuml_data.SetMaxIter(settings->max_iters);
  h2ogpuml_data.SetVerbose(settings->verbose);
  h2ogpuml_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2ogpuml_data.SetEquil(static_cast<bool>(settings->equil));
  h2ogpuml_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2ogpuml_data.SetnDev(static_cast<int>(settings->nDev));
  h2ogpuml_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2ogpuml_data.SetInitX(solution->x);
    h2ogpuml_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2ogpuml_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);
  
  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,2);
  info->obj = h2ogpuml_data.GetOptval();
  info->iter = h2ogpuml_data.GetFinalIter();
  info->rho = h2ogpuml_data.GetRho();
  info->solvetime = h2ogpuml_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2ogpuml_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2ogpuml_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2ogpuml_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2ogpuml_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,2);
}

template<typename T>
void H2OGPUMLRun(h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixSparse<T> > &h2ogpuml_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, \
                const H2OGPUMLSettings<T> *settings, H2OGPUMLInfo<T> *info, H2OGPUMLSolution<T> *solution){
  // Set parameters.
  h2ogpuml_data.SetRho(settings->rho);
  h2ogpuml_data.SetAbsTol(settings->abs_tol);
  h2ogpuml_data.SetRelTol(settings->rel_tol);
  h2ogpuml_data.SetMaxIter(settings->max_iters);
  h2ogpuml_data.SetVerbose(settings->verbose);
  h2ogpuml_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2ogpuml_data.SetEquil(static_cast<bool>(settings->equil));
  h2ogpuml_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2ogpuml_data.SetnDev(static_cast<int>(settings->nDev));
  h2ogpuml_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2ogpuml_data.SetInitX(solution->x);
    h2ogpuml_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2ogpuml_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);

  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,2);
  info->obj = h2ogpuml_data.GetOptval();
  info->iter = h2ogpuml_data.GetFinalIter();
  info->rho = h2ogpuml_data.GetRho();
  info->solvetime = h2ogpuml_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2ogpuml_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2ogpuml_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2ogpuml_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2ogpuml_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,2);
}

template<typename T>
void H2OGPUMLRun(h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixDense<T> > &h2ogpuml_data, std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const H2OGPUMLSettings<T> *settings, H2OGPUMLInfo<T> *info, H2OGPUMLSolution<T> *solution){
  // Set parameters.
  h2ogpuml_data.SetRho(settings->rho);
  h2ogpuml_data.SetAbsTol(settings->abs_tol);
  h2ogpuml_data.SetRelTol(settings->rel_tol);
  h2ogpuml_data.SetMaxIter(settings->max_iters);
  h2ogpuml_data.SetVerbose(settings->verbose);
  h2ogpuml_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2ogpuml_data.SetEquil(static_cast<bool>(settings->equil));
  h2ogpuml_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2ogpuml_data.SetnDev(static_cast<int>(settings->nDev));
  h2ogpuml_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2ogpuml_data.SetInitX(solution->x);
    h2ogpuml_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2ogpuml_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);

  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,2);
  info->obj = h2ogpuml_data.GetOptval();
  info->iter = h2ogpuml_data.GetFinalIter();
  info->rho = h2ogpuml_data.GetRho();
  info->solvetime = h2ogpuml_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2ogpuml_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2ogpuml_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2ogpuml_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2ogpuml_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,2);
}

template<typename T>
void H2OGPUMLRun(h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T> > &h2ogpuml_data, const std::vector<FunctionObj<T> > *f, std::vector<FunctionObj<T> > *g, 
              const H2OGPUMLSettings<T> *settings, H2OGPUMLInfo<T> *info, H2OGPUMLSolution<T> *solution){
  // Set parameters.
  h2ogpuml_data.SetRho(settings->rho);
  h2ogpuml_data.SetAbsTol(settings->abs_tol);
  h2ogpuml_data.SetRelTol(settings->rel_tol);
  h2ogpuml_data.SetMaxIter(settings->max_iters);
  h2ogpuml_data.SetVerbose(settings->verbose);
  h2ogpuml_data.SetAdaptiveRho(static_cast<bool>(settings->adaptive_rho));
  h2ogpuml_data.SetEquil(static_cast<bool>(settings->equil));
  h2ogpuml_data.SetGapStop(static_cast<bool>(settings->gap_stop));
  h2ogpuml_data.SetnDev(static_cast<int>(settings->nDev));
  h2ogpuml_data.SetwDev(static_cast<int>(settings->wDev));

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    h2ogpuml_data.SetInitX(solution->x);
    h2ogpuml_data.SetInitLambda(solution->nu);
  }

  // Solve.
  PUSH_RANGE("Solve",Solve,1);
  info->status = h2ogpuml_data.Solve(*f, *g);
  POP_RANGE("Solve",Solve,1);

  // Retrieve solver output & state
  PUSH_RANGE("Get",Get,1);
  info->obj = h2ogpuml_data.GetOptval();
  info->iter = h2ogpuml_data.GetFinalIter();
  info->rho = h2ogpuml_data.GetRho();
  info->solvetime = h2ogpuml_data.GetTime();

  size_t m = f->size();
  size_t n = g->size();

  memcpy(solution->x, h2ogpuml_data.GetX(), n * sizeof(T));
  memcpy(solution->y, h2ogpuml_data.GetY(), m * sizeof(T));
  memcpy(solution->mu, h2ogpuml_data.GetMu(), n * sizeof(T));  
  memcpy(solution->nu, h2ogpuml_data.GetLambda(), m * sizeof(T));
  POP_RANGE("Get",Get,1);
}


template <typename T>
int H2OGPUMLRun(void *work, const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e, const FUNCTION *f_h,
            const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e, const FUNCTION *g_h,
            void *settings_, void *info_, void *solution_){

  if (!VerifyH2OGPUMLWork(work)) { return static_cast <int>(H2OGPUML_ERROR); }


  const H2OGPUMLSettings<T> * settings = static_cast<H2OGPUMLSettings<T> *>(settings_);
  H2OGPUMLInfo<T> * info = static_cast<H2OGPUMLInfo<T> *>(info_);
  H2OGPUMLSolution<T> * solution = static_cast<H2OGPUMLSolution<T> *>(solution_);  
  H2OGPUMLWork * p_work = static_cast<H2OGPUMLWork *>(work);


  size_t m = p_work->m;
  size_t n = p_work->n;
  std::vector<FunctionObj<T> > *f = static_cast<std::vector<FunctionObj<T> > *>(p_work->f);
  std::vector<FunctionObj<T> > *g = static_cast<std::vector<FunctionObj<T> > *>(p_work->g);

  // Update f and g
  H2OGPUMLFunctionUpdate(m, f, f_a, f_b, f_c, f_d, f_e, f_h);
  H2OGPUMLFunctionUpdate(n, g, g_a, g_b, g_c, g_d, g_e, g_h);

  // Run
  if (p_work->densebit){
    if (p_work->directbit){
      h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > *h2ogpuml_data = static_cast< h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > *>(p_work->h2ogpuml_data);
      H2OGPUMLRun(*h2ogpuml_data, f, g, settings, info, solution);
    }else{
      h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixDense<T> > *h2ogpuml_data = static_cast< h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixDense<T> > *>(p_work->h2ogpuml_data);
      H2OGPUMLRun(*h2ogpuml_data, f, g, settings, info, solution);
    }
  }else{
    h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T> > *h2ogpuml_data = static_cast< h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T> > *>(p_work->h2ogpuml_data);
    H2OGPUMLRun(*h2ogpuml_data, f, g, settings, info, solution);      
  }
  return info->status;
} 

template<typename T>
void H2OGPUMLShutdown(void * work){
  H2OGPUMLWork * p_work = static_cast<H2OGPUMLWork *>(work);

  std::vector<FunctionObj<T> > *f, *g;
  f = static_cast<std::vector<FunctionObj<T> > *>(p_work->f);
  g = static_cast<std::vector<FunctionObj<T> > *>(p_work->g);

  delete f;
  delete g;

  // Run
  if (p_work->densebit){
    if (p_work->directbit){
      h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > *h2ogpuml_data = static_cast< h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > *>(p_work->h2ogpuml_data);
      delete h2ogpuml_data;
    }else{
      h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixDense<T> > *h2ogpuml_data = static_cast< h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixDense<T> > *>(p_work->h2ogpuml_data);
      delete h2ogpuml_data;
    }
  }else{
    h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T> > *h2ogpuml_data = static_cast< h2ogpuml::H2OGPUMLIndirect<T, h2ogpuml::MatrixSparse<T> > *>(p_work->h2ogpuml_data);
    delete h2ogpuml_data;
  }

  delete p_work;
}


extern "C" {


void * h2ogpuml_init_dense_single(int wDev, enum ORD ord, size_t m, size_t n, const float *A){
    return ord == COL_MAJ ? H2OGPUMLInit<float>(wDev, m,n,A,'c') : H2OGPUMLInit<float>(wDev, m,n,A,'r');
}
void * h2ogpuml_init_dense_double(int wDev, enum ORD ord, size_t m, size_t n, const double *A){
    return ord == COL_MAJ ? H2OGPUMLInit<double>(wDev, m,n,A,'c') : H2OGPUMLInit<double>(wDev, m,n,A,'r');
}
void * h2ogpuml_init_sparse_single(int wDev, enum ORD ord, size_t m, size_t n, size_t nnz, const float *nzvals, const int *indices, const int *pointers){
  return ord == COL_MAJ ? H2OGPUMLInit<float>(wDev, m,n,nnz,nzvals,indices,pointers,'c') : H2OGPUMLInit<float>(wDev, m,n,nnz,nzvals,indices,pointers,'r');


}
void * h2ogpuml_init_sparse_double(int wDev, enum ORD ord, size_t m, size_t n, size_t nnz, const double *nzvals, const int *indices, const int *pointers){
  return ord == COL_MAJ ? H2OGPUMLInit<double>(wDev, m,n,nnz,nzvals,indices,pointers,'c') : H2OGPUMLInit<double>(wDev, m,n,nnz,nzvals,indices,pointers,'r');
}

int h2ogpuml_solve_single(void *work, H2OGPUMLSettingsS *settings, H2OGPUMLSolutionS *solution, H2OGPUMLInfoS *info,
                      const float *f_a, const float *f_b, const float *f_c,const float *f_d, const float *f_e, const enum FUNCTION *f_h,
                      const float *g_a, const float *g_b, const float *g_c,const float *g_d, const float *g_e, const enum FUNCTION *g_h){
  return H2OGPUMLRun<float>(work, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h, settings, info, solution);
}
int h2ogpuml_solve_double(void *work, H2OGPUMLSettingsD *settings, H2OGPUMLSolutionD *solution, H2OGPUMLInfoD *info,
                      const double *f_a, const double *f_b, const double *f_c,const double *f_d, const double *f_e, const enum FUNCTION *f_h,
                      const double *g_a, const double *g_b, const double *g_c,const double *g_d, const double *g_e, const enum FUNCTION *g_h){
  return H2OGPUMLRun<double>(work, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h, settings, info, solution);
}

void h2ogpuml_finish_single(void * work){ return H2OGPUMLShutdown<float>(work); }
void h2ogpuml_finish_double(void * work){ return H2OGPUMLShutdown<double>(work); }

}


