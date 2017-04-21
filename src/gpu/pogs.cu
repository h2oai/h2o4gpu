#include "pogs.h"
#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>

#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "interface_defs.h"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector.h"
#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "util.h"
#include "cuda_utils.h"

#include "timer.h"

typedef struct {
  double* sendBuff;
  double* recvBuff;
  int size;
  cudaStream_t stream;
} PerThreadData;


#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace pogs {

namespace {

template <typename T, typename Op>
struct ApplyOp: thrust::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  __host__ __device__ FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x);
    h.d = binary_op(h.d, x);
    h.e = binary_op(binary_op(h.e, x), x);
    return h;
  }
};

}  // namespace

template <typename T, typename M, typename P>
Pogs<T, M, P>::Pogs(int wDev, const M &A)
    : _A(wDev, A), _P(wDev, _A),
      _de(0), _z(0), _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0), _y(0), _mu(0), _lambda(0), _optval(static_cast<T>(0.)), _time(static_cast<T>(0.)),
      _trainPreds(0), _validPreds(0),
      _xp(0), _trainPredsp(0), _validPredsp(0),
      _trainrmse(0),_validrmse(0),
      _trainmean(0),_validmean(0),
      _trainstddev(0),_validstddev(0),
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _equil(kEquil),
      _gap_stop(kGapStop),
      _init_x(false), _init_lambda(false),
      _nDev(1), //FIXME - allow larger comm groups
      _wDev(wDev)
#ifdef USE_NCCL2
      ,_comms(0)
#endif
  
{

  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  printLegalNotice();

  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
  _trainPreds = new T[_A.Rows()]();
  _validPreds = new T[_A.ValidRows()]();
}

  
template <typename T, typename M, typename P>
Pogs<T, M, P>::Pogs(const M &A)
  :_A(A._wDev,A), _P(_A._wDev,_A),
      _de(0), _z(0), _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0), _y(0), _mu(0), _lambda(0), _optval(static_cast<T>(0.)), _time(static_cast<T>(0.)),
      _trainPreds(0), _validPreds(0),
      _xp(0), _trainPredsp(0), _validPredsp(0),
      _trainrmse(0),_validrmse(0),
      _trainmean(0),_validmean(0),
      _trainstddev(0),_validstddev(0),
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _equil(kEquil),
      _gap_stop(kGapStop),
      _init_x(false), _init_lambda(false),
      _nDev(1), //FIXME - allow larger comm groups
      _wDev(_A._wDev)
#ifdef USE_NCCL2
      ,comms(0)
#endif
{

  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  printLegalNotice();

  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
  _trainPreds = new T[_A.Rows()]();
  _validPreds = new T[_A.ValidRows()]();
}

  
template <typename T, typename M, typename P>
int Pogs<T, M, P>::_Init() {
  DEBUG_EXPECT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;
  CUDACHECK(cudaSetDevice(_wDev));


#ifdef _DEBUG
  // get device ID
  int devID;
  CUDACHECK(cudaGetDevice(&devID));
  cudaDeviceProp props;
  // get device properties
  CUDACHECK(cudaGetDeviceProperties(&props, devID));
#endif

#ifdef USE_NCCL2
  for (int i = 0; i < _nDev; i++){
    if(i==0 && i==_nDev-1) i=_wDev; // force to chosen device
    cudaDeviceProp props;
    CUDACHECK(cudaGetDeviceProperties(&props, i));
    CUDACHECK(cudaSetDevice(i));
    //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
    printf("Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,i); fflush(stdout);
  }


  // initialize nccl

  std::vector<int> dList(_nDev);
  for (int i = 0; i < _nDev; ++i)
    dList[i] = i % nVis;

  ncclComm_t* _comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*_nDev);
  NCCLCHECK(ncclCommInitAll(_comms, _nDev, dList.data())); // initialize communicator (One communicator per process)
  printf("# NCCL: Using devices\n");
  for (int g = 0; g < _nDev; ++g) {
    int cudaDev;
    int rank;
    cudaDeviceProp prop;
    NCCLCHECK(ncclCommCuDevice(_comms[g], &cudaDev));
    NCCLCHECK(ncclCommUserRank(_comms[g], &rank));
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev,
           prop.pciBusID, prop.name); fflush(stdout);
  }
#endif

  
  PUSH_RANGE("Malloc",Malloc,1);
  double t0 = timer<double>();

  size_t m = _A.Rows();
  size_t mvalid = _A.ValidRows();
  size_t n = _A.Cols();
  fprintf(stderr,"in pogs: m=%d n=%d\n",(int)m,(int)n); fflush(stderr);

  cudaMalloc(&_de, (m + n) * sizeof(T));
  cudaMalloc(&_z, (m + n) * sizeof(T));
  cudaMalloc(&_zt, (m + n) * sizeof(T));
  cudaMemset(_de, 0, (m + n) * sizeof(T));
  cudaMemset(_z, 0, (m + n) * sizeof(T));
  cudaMemset(_zt, 0, (m + n) * sizeof(T));

  // local (i.e. GPU) values for _x and training predictions (i.e. predicted y from Atrain*_x)
  cudaMalloc(&_xp, (n) * sizeof(T));
  cudaMalloc(&_trainPredsp, (m) * sizeof(T));
  cudaMalloc(&_validPredsp, (mvalid) * sizeof(T));
  cudaMemset(_xp, 0, (n) * sizeof(T));
  cudaMemset(_trainPredsp, 0, (m) * sizeof(T));
  cudaMemset(_validPredsp, 0, (mvalid) * sizeof(T));

  CUDA_CHECK_ERR();
 
  _A.Init();
  POP_RANGE("Malloc",Malloc,1);

  PUSH_RANGE("Eq",Eq,1);
  _A.Equil(_de, _de + m, _equil);
  POP_RANGE("Eq",Eq,1);
  

//  PUSH_RANGE("Init1",Init1,1);
  _P.Init();
  CUDA_CHECK_ERR();
//  POP_RANGE("Init1",Init1,1);

  printf("Time to allocate data structures: %f\n", timer<double>() - t0);

  return 0;
}

template <typename T, typename M, typename P>
PogsStatus Pogs<T, M, P>::Solve(const std::vector<FunctionObj<T> > &f,
                                const std::vector<FunctionObj<T> > &g) {
  double t0 = timer<double>();
  // TODO: Constants are set arbitrarily based upon limited experiments in academic papers
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin   = static_cast<T>(1.05); // for adaptive rho and rescaling
  const T kGamma      = static_cast<T>(1.01); // for adaptive rho and rescaling
  const T kTau        = static_cast<T>(0.8); // for adaptive rho and rescaling
  const T kAlpha      = static_cast<T>(1.7); // set to 1.0 to disable over-relaxation technique, normally 1.5-1.8 and was set to 1.7
  const T kRhoMin     = static_cast<T>(1e-4); // lower range for adaptive rho
  const T kRhoMax     = static_cast<T>(1e4); // upper range for adaptive rho
  const T kKappa      = static_cast<T>(0.9); // for adaptive rho and rescaling
  const T kOne        = static_cast<T>(1.0); // definition
  const T kZero       = static_cast<T>(0.0); // definition
  const T kProjTolMax = static_cast<T>(1e-6); // Projection tolerance
  const T kProjTolMin = static_cast<T>(1e-2); // Projection tolerance
  const T kProjTolPow = static_cast<T>(1.3); // Projection tolerance
  const T kProjTolIni = static_cast<T>(1e-5); // Projection tolerance
  const bool use_exact_stop = true; // false does worse in trainRMSE and maximum number of iterations with simple.R

  //  PUSH_RANGE("PogsSolve",PogsSolve,1);

  
  // Initialize Projector P and Matrix A.
  if (!_done_init){
//    PUSH_RANGE("Init2",Init2,1);
    _Init();
    //    POP_RANGE("Init2",Init2,1);
  }
  CUDACHECK(cudaSetDevice(_wDev));


  // Notes on variable names:
  //
  // Original Boyd ADMM paper solves:
  // http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
  // Minimize: f(x) + g(z)
  // Subject to: Ax + Bz = c
  // Primary variable: x
  // Dual variable: z
  // Step size: \rho
  // Where for Lasso: f(x) = (1/2)||x-b||_2^2 and g(z) = \lambda||z||_1 with constraint x=Az
  //
  // Pogs paper and code:
  // http://foges.github.io/pogs/ and http://stanford.edu/~boyd/papers/pogs.html
  // Minimize: f(y) + g(x) for a variety (but limited set) of f and g shown in src/include/prox_lib.h
  // Subject to: y = Ax (always)
  // Where for Lasso: f(y) = (1/2)||y-b||_2^2 and g(x) = \lambda||x||_1 and constraint is y=Ax
  // Primary variable: y
  // Dual variable: x
  // Step size or Proximal parameter: \rho
  // Intermediate variable: z
  // Internally pogs code uses \mu and \nu scaled variables, performs pre-conditioning using e and d.
  // \lambda_{max} = ||A^T b|| makes sense if have (1/2) in front of f(y) for Lasso
  //
  // Pogs overall steps:
  // 1) Precondition A using d and e and renormalize variables and all equations using d and e
  // 2) Compute Gramian: A^T A only once
  // 3) Cholesky of gram: Only compute cholesky once -- s and info->s in Project just kOne=1 and just ensure GPU has cholesky already.  Could have put into Init with Gramian)
  // 4) Project: Solve L L^T x = b for x by forward and backward solve (Ly=b for y and then y=L^T x for x)
  // 5) Repeat #4, until convergence from primary (min Ax-b) and dual (min f(y)+g(x)) residuals
  
  
  
  // Extract values from pogs_data
  PUSH_RANGE("PogsExtract",PogsExtract,3);
  size_t m = _A.Rows();
  size_t mvalid = _A.ValidRows();
  size_t n = _A.Cols();
  thrust::device_vector<FunctionObj<T> > f_gpu = f;
  thrust::device_vector<FunctionObj<T> > g_gpu = g;
  POP_RANGE("PogsExtract",PogsExtract,3);

  PUSH_RANGE("PogsAlloc",PogsAlloc,4);
  // Create cuBLAS handle.
  cublasHandle_t hdl;
  cublasCreate(&hdl);
  CUDA_CHECK_ERR();

  // Allocate data for ADMM variables.
  cml::vector<T> de    = cml::vector_view_array(_de, m + n);
  cml::vector<T> z     = cml::vector_view_array(_z, m + n);
  cml::vector<T> zt    = cml::vector_view_array(_zt, m + n);
  cml::vector<T> zprev = cml::vector_calloc<T>(m + n);
  cml::vector<T> ztemp = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12   = cml::vector_calloc<T>(m + n);
  CUDA_CHECK_ERR();

  // Create views for x and y components (same memory space used, not value copy)
  cml::vector<T> d     = cml::vector_subvector(&de, 0, m);
  cml::vector<T> e     = cml::vector_subvector(&de, m, n);
  cml::vector<T> x     = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y     = cml::vector_subvector(&z, n, m);
  cml::vector<T> x12   = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12   = cml::vector_subvector(&z12, n, m);
  cml::vector<T> xprev = cml::vector_subvector(&zprev, 0, n);
  cml::vector<T> yprev = cml::vector_subvector(&zprev, n, m);
  cml::vector<T> xtemp = cml::vector_subvector(&ztemp, 0, n);
  cml::vector<T> ytemp = cml::vector_subvector(&ztemp, n, m);
  CUDA_CHECK_ERR();
  POP_RANGE("PogsAlloc",PogsAlloc,4);

  
  PUSH_RANGE("PogsScale",PogsScale,5);
  // Scale f and g to account for diagonal scaling e and d.
  // f/d -> f
  thrust::transform(f_gpu.begin(), f_gpu.end(),
      thrust::device_pointer_cast(d.data), f_gpu.begin(),
      ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
  // g*e -> g
  thrust::transform(g_gpu.begin(), g_gpu.end(),
      thrust::device_pointer_cast(e.data), g_gpu.begin(),
      ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));
  CUDA_CHECK_ERR();
  POP_RANGE("PogsScale",PogsScale,5);


  PUSH_RANGE("Lambda",Lambda,6);
  // Initialize (x, lambda) from (x0, lambda0).
  if (_init_x) {
    cml::vector_memcpy(&xtemp, _x); // _x->xtemp
    cml::vector_div(&xtemp, &e); // xtemp/e -> xtemp
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data); // kOne*A*x + kZero*y -> y
    wrapcudaDeviceSynchronize(); // not needed, as vector_memory is cuda call and will follow sequentially on device
    cml::vector_memcpy(&z, &ztemp); // ztemp->z (xtemp and ytemp are views of ztemp)
    CUDA_CHECK_ERR();
  }
  if (_init_lambda) {
    cml::vector_memcpy(&ytemp, _lambda); // _lambda->ytemp
    cml::vector_div(&ytemp, &d); // ytemp/d -> ytemp
    _A.Mul('t', -kOne, ytemp.data, kZero, xtemp.data); // -kOne*y+kZero*x -> x
    wrapcudaDeviceSynchronize(); // not needed, as vector_memory is cuda call and will follow sequentially on device
    if(_rho!=0) cml::blas_scal(hdl, -kOne / _rho, &ztemp); // ztemp = ztemp * (-kOne/_rho)
    else cml::blas_scal(hdl, kZero, &ztemp); // ztemp = ztemp * (-kOne/_rho)
    cml::vector_memcpy(&zt, &ztemp); // ztemp->zt
    CUDA_CHECK_ERR();
  }
  POP_RANGE("Lambda",Lambda,6);

  PUSH_RANGE("Guess",Guess,7);
  // Make an initial guess for (x0 or lambda0).
  if (_init_x && !_init_lambda) {
    // Alternating projections to satisfy 
    //   1. \lambda \in \partial f(y), \mu \in \partial g(x)
    //   2. \mu = -A^T\lambda
    cml::vector_set_all(&zprev, kZero); // zprev = kZero
    for (unsigned int i = 0; i < kInitIter; ++i) {
#ifdef USE_NVTX
        char mystring[100];
    sprintf(mystring,"GStep%d",i);
    PUSH_RANGE(mystring,GStep,1);
#endif
      ProjSubgradEval(g_gpu, xprev.data, x.data, xtemp.data);
      ProjSubgradEval(f_gpu, yprev.data, y.data, ytemp.data);
      _P.Project(xtemp.data, ytemp.data, kOne, xprev.data, yprev.data,
          kProjTolIni);
      wrapcudaDeviceSynchronize(); // not needed, as blas's are cuda call and will follow sequentially on device
      CUDA_CHECK_ERR();
      cml::blas_axpy(hdl, -kOne, &ztemp, &zprev);// alpha*X + Y -> Y
      cml::blas_scal(hdl, -kOne, &zprev);
#ifdef USE_NVTX
        POP_RANGE(mystring,GStep,1);
#endif
    }
    // xt = -1 / \rho * \mu, yt = -1 / \rho * \lambda.
    cml::vector_memcpy(&zt, &zprev); // zprev->zt
    if(_rho!=0) cml::blas_scal(hdl, -kOne / _rho, &zt);
    else  cml::blas_scal(hdl, kZero, &zt);
  } else if (_init_lambda && !_init_x) {
    ASSERT(false);
  }
  _init_x = _init_lambda = false;
  POP_RANGE("Guess",Guess,7);

  // Save initialization time.
  double time_init = timer<double>() - t0;
  printf("Time to initialize: %f\n", time_init);

  // Signal start of execution.
  if (_verbose > 0) {
    printMe(std::cout, g[1].c, g[1].e); //debugging only: print the second since the first can be for intercept (which is then 0)
    //printData(std::cout); //only works for data in host memory!
    Printf(__HBAR__
      "           H2O AI GLM\n"
      "           Version: %s\n"
      "           (c) H2O.ai, Inc., 2017\n",
      POGS_VERSION.c_str());
  }
  if (_verbose > 1) {
    Printf(__HBAR__
        " Iter | pri res    | pri tol | dua res    | dua tol |   gap      | eps gap |"
        " pri obj\n" __HBAR__);
  }

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * _abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * _abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * _abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int k = 0u, kd = 0u, ku = 0u;
  bool converged = false;
  T nrm_r, nrm_s, gap, eps_gap, eps_pri, eps_dua;



  // LOOP until satisfy convergence criteria
  for (;; ++k) {
#ifdef USE_NVTX
    char mystring[100];
    sprintf(mystring,"Step%d",k);
    PUSH_RANGE(mystring,Step,1);
#endif
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators g and f based upon chosen problem setup
    PUSH_RANGE("Evaluate_fg",Evaluate_fg,9);
    cml::blas_axpy(hdl, -kOne, &zt, &z); // -kOne*zt+z -> z
    ProxEval(g_gpu, _rho, x.data, x12.data); // Evaluate g(rho,x)->x12 (x^{1/2} in paper)
    ProxEval(f_gpu, _rho, y.data, y12.data); // Evaluate f(rho,y)->y12 (y^{1/2} in paper)
    CUDA_CHECK_ERR();
    POP_RANGE("Evaluate_fg",Evaluate_fg,9);

    // Compute gap, optval, and tolerances.
    PUSH_RANGE("gapoptvaltol",gapoptvaltol,9);
    cml::blas_axpy(hdl, -kOne, &z12, &z); // -kOne*z12+z->z
    cml::blas_dot(hdl, &z, &z12, &gap); // z*z12 -> gap
    gap = std::abs(gap); // |gap| -> gap
    eps_gap = sqrtmn_atol + _rel_tol * cml::blas_nrm2(hdl, &z) *
        cml::blas_nrm2(hdl, &z12);
    eps_pri = sqrtm_atol + _rel_tol * cml::blas_nrm2(hdl, &y12);
    eps_dua = _rho * (sqrtn_atol + _rel_tol * cml::blas_nrm2(hdl, &x));
    CUDA_CHECK_ERR();
    POP_RANGE("gapoptvaltol",gapoptvaltol,9);

#ifdef DEBUG 
    fprintf(stderr,"DEBUG1: %g %g\n",sqrtm_atol,cml::blas_nrm2(hdl, &y12));
#endif

    // Apply over relaxation  (optional, can set kAlpha to 1, above, to disable)
    // http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf S3.4.3
    PUSH_RANGE("orelax",orelax,9);
    cml::vector_memcpy(&ztemp, &zt);
    cml::blas_axpy(hdl, kAlpha, &z12, &ztemp);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &ztemp);
    CUDA_CHECK_ERR();
    POP_RANGE("orelax",orelax,9);

    // Project onto y = Ax.
    PUSH_RANGE("project",project,9);
    T proj_tol = kProjTolMin / std::pow(static_cast<T>(k + 1), kProjTolPow);
    proj_tol = std::max(proj_tol, kProjTolMax);
    // (x^{k+1},y^{k+1}) := Project(x^{k+1/2}+\tilde{x}^k , y^{k+1/2}+\tilde{y}^k)
    // xtemp.data: \tilde{x}^k
    // ytemp.data: \tilde{y}^k
    // x.data: x^{k+1/2}
    // y.data: y^{k+1/2}
    _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);
    //cudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
    CUDA_CHECK_ERR();
    POP_RANGE("project",project,9);

    // Calculate residuals nrm_s (dual residual) and nrm_r (primary residual)
    PUSH_RANGE("resid",resid,9);
    cml::vector_memcpy(&ztemp, &zprev);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp); // -1*z + ztemp -> ztemp
    wrapcudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
    nrm_s = _rho * cml::blas_nrm2(hdl, &ztemp);

    cml::vector_memcpy(&ztemp, &z12); // z12 has both x^{k+1/2} and y^{k+1/2}
    cml::blas_axpy(hdl, -kOne, &z, &ztemp); // -1*z + ztemp -> ztemp (i.e. -x^k + x^{k+1/2} -> xtemp and -y^k + y^{k+1/2} -> ytemp)
    wrapcudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
    nrm_r = cml::blas_nrm2(hdl, &ztemp);

    // Calculate exact residuals only if necessary.
    bool exact = false;
    if ((nrm_r < eps_pri && nrm_s < eps_dua) || use_exact_stop) {
      cml::vector_memcpy(&ztemp, &z12);
      _A.Mul('n', kOne, x12.data, -kOne, ytemp.data);
      wrapcudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
      nrm_r = cml::blas_nrm2(hdl, &ytemp);
      if ((nrm_r < eps_pri) || use_exact_stop) {
        cml::vector_memcpy(&ztemp, &z12);
        cml::blas_axpy(hdl, kOne, &zt, &ztemp);
        cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
        _A.Mul('t', kOne, ytemp.data, kOne, xtemp.data);
        wrapcudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
        nrm_s = _rho * cml::blas_nrm2(hdl, &xtemp);
        exact = true;
      }
    }
    CUDA_CHECK_ERR();
    POP_RANGE("resid",resid,9);


    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!_gap_stop || gap < eps_gap);
    if (_verbose > 3 && k % 1  == 0 ||
        _verbose > 2 && k % 10  == 0 ||
        _verbose > 1 && k % 100 == 0 ||
        _verbose > 1 && converged) {
      T optval = FuncEval(f_gpu, y12.data) + FuncEval(g_gpu, x12.data);
      Printf("%5d : %.2e <? %.2e  %.2e  <? %.2e  %.2e  <? %.2e % .2e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);fflush(stdout);
    }



    // Break if converged or there are nans
    if (converged || k == _max_iter - 1){ // || cml::vector_any_isnan(&zt))
      _final_iter = k;
#ifdef USE_NVTX
      POP_RANGE(mystring,Step,1); // pop at end of loop iteration
#endif
      break;
    }

    
    // Update dual variable.
    PUSH_RANGE("update",update,9);
    cml::blas_axpy(hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(hdl, -kOne, &z, &zt);
    CUDA_CHECK_ERR();
    POP_RANGE("update",update,9);

    // Adaptive rho (optional)
    // http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf S3.4.1
    //    http://www.cs.umd.edu/sites/default/files/scholarly_papers/ZhengXu.pdf or https://arxiv.org/abs/1605.07246
    // choose: 1 = Pogs Boyd method
    // choose: 2 = Original Boyd method of balancing residuals
    // choose: 3 = Spectral method by Zheng et al. 2015
    int whichadap=1;

    if(_adaptive_rho && _rho!=0){
      PUSH_RANGE("adaprho",adaprho,9);
      if (whichadap==1){
        if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
            kTau * static_cast<T>(k) > static_cast<T>(kd)) {
          if (_rho < kRhoMax) {
            _rho *= delta;
            cml::blas_scal(hdl, 1 / delta, &zt);
            delta = kGamma * delta;
            ku = k;
            if (_verbose > 3)
              Printf("+ rho %e\n", _rho);
          }
        } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
                   kTau * static_cast<T>(k) > static_cast<T>(ku)) {
          if (_rho > kRhoMin) {
            _rho /= delta;
            cml::blas_scal(hdl, delta, &zt);
            delta = kGamma * delta;
            kd = k;
            if (_verbose > 3)
              Printf("- rho %e\n", _rho);
          }
        } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
          xi *= kKappa;
        } else {
          delta = kDeltaMin;
        }
        CUDA_CHECK_ERR();
      } // end adaptive_rho==1
      else if (whichadap==2) {
        if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri) {
          if (_rho < kRhoMax) {
            _rho *= delta;
            cml::blas_scal(hdl, 1 / delta, &zt);
            delta = kGamma * delta;
            if (_verbose > 3)
              Printf("+ rho %e\n", _rho);
          }
        } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri) {
          if (_rho > kRhoMin) {
            _rho /= delta;
            cml::blas_scal(hdl, delta, &zt);
            delta = kGamma * delta;
            if (_verbose > 3)
              Printf("- rho %e\n", _rho);
          }
        }
        else {
          delta = kDeltaMin;
        }      CUDA_CHECK_ERR();
      } // end adaptive_rho==2
      else if (whichadap==3){
        if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
            kTau * static_cast<T>(k) > static_cast<T>(kd)) {
          if (_rho < kRhoMax) {
            _rho *= delta;
            cml::blas_scal(hdl, 1 / delta, &zt);
            delta = kGamma * delta;
            ku = k;
            if (_verbose > 3)
              Printf("+ rho %e\n", _rho);
          }
        } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
                   kTau * static_cast<T>(k) > static_cast<T>(ku)) {
          if (_rho > kRhoMin) {
            _rho /= delta;
            cml::blas_scal(hdl, delta, &zt);
            delta = kGamma * delta;
            kd = k;
            if (_verbose > 3)
              Printf("- rho %e\n", _rho);
          }
        } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
          xi *= kKappa;
        } else {
          delta = kDeltaMin;
        }
        CUDA_CHECK_ERR();
      } // end adaptive_rho==3
      POP_RANGE("adaprho",adaprho,9);
    } // end adaptive_rho
#ifdef USE_NVTX
    POP_RANGE(mystring,Step,1); // pop at end of loop iteration
#endif
  }// end for loop in k


  
  // Get optimal value
  _optval = FuncEval(f_gpu, y12.data) + FuncEval(g_gpu, x12.data);


  // Check status
  PogsStatus status;
  if (!converged && k == _max_iter - 1)
    status = POGS_MAX_ITER;
  else if (!converged && k < _max_iter - 1)
    status = POGS_NAN_FOUND;
  else
    status = POGS_SUCCESS;

  // Get run time 
  _time = static_cast<T>(timer<double>() - t0);

  // Print summary
  if (_verbose > 0) {
    Printf(__HBAR__
        "Status: %s\n" 
        "Timing: Total = %3.2e s, Init = %3.2e s\n"
        "Iter  : %u\n",
        PogsStatusString(status).c_str(), _time, time_init, k);
    Printf(__HBAR__
        "Error Metrics:\n"
        "Pri: "
        "|Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = %.2e (goal: %0.2e)\n"
        "Dua: "
        "|A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = %.2e (goal: %0.2e)\n"
        "Gap: "
        "|x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = %.2e (goal: %0.2e, gap checked=%d)\n"
           __HBAR__, _rel_tol * nrm_r / eps_pri, _rel_tol, _rel_tol * nrm_s / eps_dua, _rel_tol,
           _rel_tol * gap / eps_gap,_rel_tol,_gap_stop); fflush(stdout);
  }

  
  // Scale x, y, lambda and mu for output.
  PUSH_RANGE("Scale",Scale,1);

  // xtemp and ytemp are views of ztemp, so these operations apply to xtemp and ytemp as well
  cml::vector_memcpy(&ztemp, &zt); // zt->ztemp
  cml::blas_axpy(hdl, -kOne, &zprev, &ztemp); // -kOne*zprev+ztemp->ztemp
  cml::blas_axpy(hdl, kOne, &z12, &ztemp); // kOne*z12+ztemp->ztemp
  cml::blas_scal(hdl, -_rho, &ztemp); // -_rho*ztemp -> ztemp

  // operatons on limited views of ztemp
  cml::vector_mul(&ytemp, &d); // ytemp*d -> ytemp
  cml::vector_div(&xtemp, &e); // xtemp/e -> xtemp

  cml::vector<T> x12copy = cml::vector_calloc<T>(n);
  cml::vector_memcpy(&x12copy,&x12); // copy de version first to GPU
  T * dcopy = new T[m]();
  cml::vector_memcpy(dcopy,&d); // copy d to CPU
  
  cml::vector_div(&y12, &d); // y12/d -> y12
  cml::vector_mul(&x12, &e); // x12*e -> x12
  POP_RANGE("Scale",Scale,1);

  // Copy results from GPU to CPU for output.
  PUSH_RANGE("Copy",Copy,1);
  cml::vector_memcpy(_x, &x12); // x12->_x (GPU->CPU with vector<T>* to T*)
  cml::vector_memcpy(_xp, &x12); // x12->_xp (GPU->GPU but vector<T>* to T*)
  cml::vector_memcpy(_y, &y12); // y12->_y
  cml::vector_memcpy(_mu, &xtemp); // xtemp->_mu
  cml::vector_memcpy(_lambda, &ytemp); // ytemp->_lambda

  // compute train predictions from trainPred = Atrain.xsolution
  _A.Mul('n', static_cast<T>(1.), x12copy.data, static_cast<T>(0.), _trainPredsp); // _xp and _trainPredsp are both simple pointers on GPU
  cml::vector_memcpy(m,1,_trainPreds, _trainPredsp); // pointer on GPU to pointer on CPU
  for(unsigned int i=0;i<m;i++){
    _trainPreds[i]/=dcopy[i];
    //    fprintf(stderr,"Tp[%d]=%g\n",i,_trainPreds[i]);
  }
  if(dcopy) delete [] dcopy;
  if(x12copy.data) cml::vector_free(&x12copy);
  
  if(mvalid>0){
    // compute valid from validPreds = Avalid.xsolution
    _A.Mulvalid('n', static_cast<T>(1.), _xp, static_cast<T>(0.), _validPredsp);
    cml::vector_memcpy(mvalid,1,_validPreds, _validPredsp);
  }
  // compute rmse (not yet)
    
  // compute mean (not yet)

  // compute stddev (not yet)
  

  // Store z.
  cml::vector_memcpy(&z, &zprev); // zprev->z

  // Free memory.
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  cml::vector_free(&ztemp);
  if(hdl) cublasDestroy(hdl);
  CUDA_CHECK_ERR();
  POP_RANGE("Copy",Copy,1);

  //  POP_RANGE("PogsSolve",PogsSolve,1);

  return status;
}

template <typename T, typename M, typename P>
void Pogs<T, M, P>::ResetX(void) {
  if (!_done_init)
    _Init();
  CUDACHECK(cudaSetDevice(_wDev));

  size_t m = _A.Rows();
  size_t mvalid = _A.ValidRows();
  size_t n = _A.Cols();
  fprintf(stderr,"in pogs ResetX: m=%d n=%d\n",(int)m,(int)n); fflush(stderr);

  cudaMemset(_z, 0, (m + n) * sizeof(T));
  cudaMemset(_zt, 0, (m + n) * sizeof(T));
}


  
template <typename T, typename M, typename P>
Pogs<T, M, P>::~Pogs() {
  CUDACHECK(cudaSetDevice(_wDev));

  if(_de) cudaFree(_de);
  if(_z) cudaFree(_z);
  if(_zt) cudaFree(_zt);
  if(_xp) cudaFree(_xp);
  if(_trainPredsp) cudaFree(_trainPredsp);
  if(_validPredsp) cudaFree(_validPredsp);
  _de = _z = _zt = _xp = _trainPredsp = _validPredsp = 0;
  CUDA_CHECK_ERR();

#ifdef USE_NCCL2
  for(int i=0; i<_nDev; ++i)
    ncclCommDestroy(_comms[i]);
  free(_comms);
#endif
  
  if(_x) delete [] _x;
  if(_y) delete [] _y;
  if(_mu) delete [] _mu;
  if(_lambda) delete [] _lambda;
  if(_trainPreds) delete [] _trainPreds;
  if(_validPreds) delete [] _validPreds;
  _x = _y = _mu = _lambda = _trainPreds = _validPreds = 0;
}



// Explicit template instantiation.
#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class Pogs<double, MatrixDense <double>, ProjectorDirect<double, MatrixDense <double> > >;
template class Pogs<double, MatrixDense <double>, ProjectorCgls<double,   MatrixDense <double> > >;
template class Pogs<double, MatrixSparse<double>, ProjectorCgls<double,   MatrixSparse<double> > >;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class Pogs<float, MatrixDense<float>,  ProjectorDirect<float, MatrixDense<float> > >;
template class Pogs<float, MatrixDense<float>,  ProjectorCgls<float, MatrixDense<float> > >;
template class Pogs<float, MatrixSparse<float>, ProjectorCgls<float, MatrixSparse<float> > >;
#endif

}  // namespace pogs

