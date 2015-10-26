#include <cstring> // memcpy
#include <vector>
#include "pogs.h"
#include "pogs_c.h"

// to be created and managed by caller
template <typename T>
struct PogsSettings{
  T rho, abs_tol, rel_tol;
  unsigned int max_iters, verbose;
  int adaptive_rho, gap_stop, warm_start;
};

template <typename T>
struct PogsInfo{
    unsigned int iter;
    int status;
    T obj, rho, solvetime;
};

template <typename T>
struct PogsSolution{
    T *x, *y, *mu, *nu; 
};

// created and managed locally
struct PogsWork{
    size_t m,n;
    bool densebit, rowmajorbit;
    void *pogs_data, *x, *y, *mu, *nu;

    PogsWork(size_t m_, size_t n_, bool dense_, bool rowmajor_, void *pogs_data_, void *x_, void *y_, void *mu_, void *nu_){
      m=m_;n=n_;
      densebit=dense_; rowmajorbit=rowmajor_;
      pogs_data= pogs_data_;
      x=x_; y=y_; mu=mu_; nu=nu_;
    }
};


bool VerifyPogsWork(void * work){
  if (!work) { return false; }
  PogsWork * p_work = static_cast<PogsWork *>(work);
  if (!(p_work->pogs_data) || !(p_work->x) || !(p_work->y) ){ return false; }
  else { return true; } 
}

//Dense 
template <typename T, ORD O>
void * PogsInit(size_t m, size_t n, T *A){

  bool densebit = true, rowmajorbit = O==ROW_MAJ;

  // data containers
  Dense<T, static_cast<POGS_ORD>(O)> A_(A);
  PogsData<T, Dense<T, static_cast<POGS_ORD>(O)> >  *pogs_data;
  std::vector<T> *x, *y, *mu, *nu;
  PogsWork * work;

  // create new data vectors
  x = new std::vector<T>;
  y = new std::vector<T>;
  mu = new std::vector<T>;
  nu = new std::vector<T>;

  y->resize(m);
  x->resize(n);
  nu->resize(m);
  mu->resize(n);


  // create new pogs data object
  pogs_data = new PogsData<T, Dense<T, static_cast<POGS_ORD>(O)> >(A_, m, n);
  pogs_data->x=x->data();
  pogs_data->y=y->data();
  pogs_data->mu=mu->data();
  pogs_data->nu=nu->data();

  // initialize function vectors
  pogs_data->f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data->f.emplace_back(static_cast<Function>(kZero));   

  pogs_data->g.reserve(n);
  for (unsigned int j = 0; j < n; ++j)
    pogs_data->g.emplace_back(static_cast<Function>(kZero));   

  // allocate factors (enables warm start)
  AllocDenseFactors(pogs_data);

  // create new PogsWork struct
  work = new PogsWork(m,n, densebit, rowmajorbit, static_cast<void *>(pogs_data), \
                    static_cast<void *>(x), static_cast<void *>(y), \
                    static_cast<void *>(mu), static_cast<void *>(nu));

  return static_cast<void *>(work);
}

template <typename T>
void PogsFunctionUpdate(size_t m, std::vector<FunctionObj<T> > &f, const T *f_a, const T *f_b, const T *f_c, 
                            const T *f_d, const T *f_e, const FUNCTION *f_h){


  for (unsigned int i = 0; i < m; ++i)
    f[i].a= f_a[i];
  for (unsigned int i = 0; i < m; ++i)
    f[i].b= f_b[i];
  for (unsigned int i = 0; i < m; ++i)
    f[i].c= f_c[i];
  for (unsigned int i = 0; i < m; ++i)
    f[i].d= f_d[i];
  for (unsigned int i = 0; i < m; ++i)
    f[i].e= f_e[i];
  for (unsigned int i = 0; i < m; ++i)
    f[i].h= static_cast<Function>(f_h[i]);
}


template <typename T, POGS_ORD O>
void PogsRun(PogsData<T, Dense<T, O> > &pogs_data, const PogsSettings<T> *settings, PogsInfo<T> *info, PogsSolution<T> *solution){


  // Set parameters.
  pogs_data.rho=settings->rho;
  pogs_data.abs_tol=settings->abs_tol;
  pogs_data.rel_tol=settings->rel_tol;
  pogs_data.max_iter=settings->max_iters;
  pogs_data.quiet=!static_cast<bool>(settings->verbose);
  pogs_data.adaptive_rho=static_cast<bool>(settings->adaptive_rho);
  pogs_data.gap_stop=static_cast<bool>(settings->gap_stop);

  // Get problem dims
  size_t m = pogs_data.f.size();
  size_t n = pogs_data.g.size();

  // Optionally, feed in warm start variables
  if (static_cast<bool>(settings->warm_start)){
    pogs_data.warm_start=true;
    std::memcpy(pogs_data.x, solution->x, n * sizeof(T));
    std::memcpy(pogs_data.nu, solution->nu, m * sizeof(T));
  }


  // Solve.
  info->status = Pogs(&pogs_data);

  // Retrieve solver output & state
  info->obj = pogs_data.optval;
  // info->iter = pogs_data.GetFinalIter();
  info->rho = pogs_data.rho;
  // info->solvetime = pogs_data.GetTime();

  std::memcpy(solution->x, pogs_data.x, n * sizeof(T));
  std::memcpy(solution->y, pogs_data.y, m * sizeof(T));
  std::memcpy(solution->mu, pogs_data.mu, n * sizeof(T));  
  std::memcpy(solution->nu, pogs_data.nu, m * sizeof(T));


  // always reset warm start flag
  pogs_data.warm_start=false;

}


template <typename T>
int PogsRun(void *work, const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e, const FUNCTION *f_h,
            const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e, const FUNCTION *g_h,
            void *settings_, void *info_, void *solution_){

  if (!VerifyPogsWork(work)) { return static_cast <int>(POGS_ERROR); }


  const PogsSettings<T> * settings = static_cast<PogsSettings<T> *>(settings_);
  PogsInfo<T> * info = static_cast<PogsInfo<T> *>(info_);
  PogsSolution<T> * solution = static_cast<PogsSolution<T> *>(solution_);  
  PogsWork * p_work = static_cast<PogsWork *>(work);

  size_t m = p_work->m;
  size_t n = p_work->n;


  // Run
  if (p_work->densebit){
    if (p_work->rowmajorbit)
    {
      PogsData<T, Dense<T, ROW> > *pogs_data = static_cast< PogsData<T, Dense<T, ROW> > *>(p_work->pogs_data);

      // Update f and g
      PogsFunctionUpdate(m, pogs_data->f, f_a, f_b, f_c, f_d, f_e, f_h);
      PogsFunctionUpdate(n, pogs_data->g, g_a, g_b, g_c, g_d, g_e, g_h);
      PogsRun(*pogs_data, settings, info, solution);    
    }else{
      PogsData<T, Dense<T, COL> > *pogs_data = static_cast< PogsData<T, Dense<T, COL> > *>(p_work->pogs_data);

      // Update f and g
      PogsFunctionUpdate(m, pogs_data->f, f_a, f_b, f_c, f_d, f_e, f_h);
      PogsFunctionUpdate(n, pogs_data->g, g_a, g_b, g_c, g_d, g_e, g_h);
      PogsRun(*pogs_data, settings, info, solution);    
    }

  }else{
    printf("\nWARNING: SPARSE POGS METHODS NOT IMPLEMENTED IN C INTERFACE.\n");
  }

  return info->status;
} 


template <typename T>
void PogsShutdown(void *work){
  PogsWork * p_work = static_cast<PogsWork *>(work);

  if (p_work->densebit){
    if (p_work->rowmajorbit)
    {
      PogsData<T, Dense<T, ROW> > *pogs_data = static_cast< PogsData<T, Dense<T, ROW> > *>(p_work->pogs_data);
      FreeDenseFactors(pogs_data);
      delete pogs_data;
    }else{
      PogsData<T, Dense<T, COL> > *pogs_data = static_cast< PogsData<T, Dense<T, COL> > *>(p_work->pogs_data);
      FreeDenseFactors(pogs_data);
      delete pogs_data;
    }

  }else{
    printf("\nWARNING: SPARSE POGS METHODS NOT IMPLEMENTED IN C INTERFACE.\n");
  }

  std::vector<T> *x = static_cast<std::vector<T> *>(p_work->x);
  std::vector<T> *y = static_cast<std::vector<T> *>(p_work->y);
  std::vector<T> *mu = static_cast<std::vector<T> *>(p_work->mu);
  std::vector<T> *nu = static_cast<std::vector<T> *>(p_work->nu);

  delete x;
  delete y;
  delete mu;
  delete nu;

  delete p_work;
}

extern "C" {

void * pogs_init_dense_single(enum ORD ord, size_t m, size_t n, float *A){
  return ord == ROW_MAJ ? PogsInit<float, ROW_MAJ>(m,n,A) : PogsInit<float, COL_MAJ>(m,n,A);   
}

void * pogs_init_dense_double(enum ORD ord, size_t m, size_t n, double *A){
  return ord == ROW_MAJ ? PogsInit<double, ROW_MAJ>(m,n,A) : PogsInit<double, COL_MAJ>(m,n,A);   
}

void * pogs_init_sparse_single(enum ORD ord, size_t m, size_t n, size_t nnz, const float *nzvals, const int *indices, const int *pointers){
  printf("\nWARNING: SPARSE POGS METHODS NOT IMPLEMENTED IN C INTERFACE. RETURNING NULL POINTER\n");
  return (void *)0;
}

void * pogs_init_sparse_double(enum ORD ord, size_t m, size_t n, size_t nnz, const double *nzvals, const int *indices, const int *pointers){
  printf("\nWARNING: SPARSE POGS METHODS NOT IMPLEMENTED IN C INTERFACE. RETURNING NULL POINTER\n");
  return (void *)0;
}

int pogs_solve_single(void *work, PogsSettingsS *settings, PogsSolutionS *solution, PogsInfoS *info,
                      const float *f_a, const float *f_b, const float *f_c,const float *f_d, const float *f_e, const enum FUNCTION *f_h,
                      const float *g_a, const float *g_b, const float *g_c,const float *g_d, const float *g_e, const enum FUNCTION *g_h){
  return PogsRun<float>(work, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h, settings, info, solution);
}
int pogs_solve_double(void *work, PogsSettingsD *settings, PogsSolutionD *solution, PogsInfoD *info,
                      const double *f_a, const double *f_b, const double *f_c,const double *f_d, const double *f_e, const enum FUNCTION *f_h,
                      const double *g_a, const double *g_b, const double *g_c,const double *g_d, const double *g_e, const enum FUNCTION *g_h){
  return PogsRun<double>(work, f_a, f_b, f_c, f_d, f_e, f_h, g_a, g_b, g_c, g_d, g_e, g_h, settings, info, solution);
}

void pogs_finish_single(void * work){ return PogsShutdown<float>(work); }
void pogs_finish_double(void * work){ return PogsShutdown<double>(work); }

}

