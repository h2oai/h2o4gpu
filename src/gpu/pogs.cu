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
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _equil(kEquil),
      _gap_stop(kGapStop),
      _nDev(1), //FIXME - allow larger comm groups
      _wDev(wDev),
#ifdef USE_NCCL2
      _comms(0),
#endif
      _init_x(false), _init_lambda(false) {

  CUDACHECK(cudaSetDevice(_wDev));

  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
}

template <typename T, typename M, typename P>
int Pogs<T, M, P>::_Init() {
  DEBUG_EXPECT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;
  CUDACHECK(cudaSetDevice(_wDev));


#ifdef _DEBUG
  //  int _nDev=1; // number of cuda devices to use
  //  int _wDev=0; // which cuda device(s) to use
  // get number of devices visible/available
  int nVis = 0;
  CUDACHECK(cudaGetDeviceCount(&nVis));
  for (int i = 0; i < nVis; i++){
    cudaDeviceProp props;
    CUDACHECK(cudaGetDeviceProperties(&props, i));
    printf("Visible: Compute %d.%d CUDA device: [%s] : cudadeviceid: %2d of %2d devices [0x%02x] mpc=%d\n", props.major, props.minor, props.name, i, nVis, props.pciBusID, props.multiProcessorCount); fflush(stdout);
  }
  
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
  size_t n = _A.Cols();
  fprintf(stderr,"in pogs: m=%d n=%d\n",(int)m,(int)n); fflush(stderr);

  cudaMalloc(&_de, (m + n) * sizeof(T));
  cudaMalloc(&_z, (m + n) * sizeof(T));
  cudaMalloc(&_zt, (m + n) * sizeof(T));
  cudaMemset(_de, 0, (m + n) * sizeof(T));
  cudaMemset(_z, 0, (m + n) * sizeof(T));
  cudaMemset(_zt, 0, (m + n) * sizeof(T));
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
  const T kKappa      = static_cast<T>(0.4); // for adaptive rho and rescaling
  const T kOne        = static_cast<T>(1.0); // definition
  const T kZero       = static_cast<T>(0.0); // definition
  const T kProjTolMax = static_cast<T>(1e-8); // Projection tolerance
  const T kProjTolMin = static_cast<T>(1e-2); // Projection tolerance
  const T kProjTolPow = static_cast<T>(1.3); // Projection tolerance
  const T kProjTolIni = static_cast<T>(1e-5); // Projection tolerance
  bool use_exact_stop = false;

  //  PUSH_RANGE("PogsSolve",PogsSolve,1);

  
  // Initialize Projector P and Matrix A.
  if (!_done_init){
//    PUSH_RANGE("Init2",Init2,1);
    _Init();
    //    POP_RANGE("Init2",Init2,1);
  }
  CUDACHECK(cudaSetDevice(_wDev));


  
  // Extract values from pogs_data
  PUSH_RANGE("PogsExtract",PogsExtract,3);
  size_t m = _A.Rows();
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

  // Create views for x and y components.
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
  thrust::transform(f_gpu.begin(), f_gpu.end(),
      thrust::device_pointer_cast(d.data), f_gpu.begin(),
      ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
  thrust::transform(g_gpu.begin(), g_gpu.end(),
      thrust::device_pointer_cast(e.data), g_gpu.begin(),
      ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));
  CUDA_CHECK_ERR();
  POP_RANGE("PogsScale",PogsScale,5);

  PUSH_RANGE("Lambda",Lambda,6);
  // Initialize (x, lambda) from (x0, lambda0).
  if (_init_x) {
    cml::vector_memcpy(&xtemp, _x);
    cml::vector_div(&xtemp, &e);
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data); // y = kOne*A*x + kZero*y
    wrapcudaDeviceSynchronize(); // not needed, as vector_memory is cuda call and will follow sequentially on device
    cml::vector_memcpy(&z, &ztemp); // ztemp->z TODO: Bug or wrong?
    CUDA_CHECK_ERR();
  }
  if (_init_lambda) {
    cml::vector_memcpy(&ytemp, _lambda);
    cml::vector_div(&ytemp, &d);
    _A.Mul('t', -kOne, ytemp.data, kZero, xtemp.data);
    wrapcudaDeviceSynchronize(); // not needed, as vector_memory is cuda call and will follow sequentially on device
    if(_rho!=0) cml::blas_scal(hdl, -kOne / _rho, &ztemp); // ztemp = ztemp * (-kOne/_rho)
    else cml::blas_scal(hdl, kZero, &ztemp); // ztemp = ztemp * (-kOne/_rho)
    cml::vector_memcpy(&zt, &ztemp); // ztemp->z
    CUDA_CHECK_ERR();
  }
  POP_RANGE("Lambda",Lambda,6);

  PUSH_RANGE("Guess",Guess,7);
  // Make an initial guess for (x0 or lambda0).
  if (_init_x && !_init_lambda) {
    // Alternating projections to satisfy 
    //   1. \lambda \in \partial f(y), \mu \in \partial g(x)
    //   2. \mu = -A^T\lambda
    cml::vector_set_all(&zprev, kZero);
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
    printMe(std::cout, g[0].c, g[0].e);
    Printf(__HBAR__
      "           H2O.ai Proximal Graph Solver\n"
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
    cml::blas_axpy(hdl, -kOne, &zt, &z);
    ProxEval(g_gpu, _rho, x.data, x12.data);
    ProxEval(f_gpu, _rho, y.data, y12.data);
    CUDA_CHECK_ERR();
    POP_RANGE("Evaluate_fg",Evaluate_fg,9);

    // Compute gap, optval, and tolerances.
    PUSH_RANGE("gapoptvaltol",gapoptvaltol,9);
    cml::blas_axpy(hdl, -kOne, &z12, &z);
    cml::blas_dot(hdl, &z, &z12, &gap);
    gap = std::abs(gap);
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
    _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);
    //cudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
    CUDA_CHECK_ERR();
    POP_RANGE("project",project,9);

    // Calculate residuals nrm_s and nrm_r
    PUSH_RANGE("resid",resid,9);
    cml::vector_memcpy(&ztemp, &zprev);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    wrapcudaDeviceSynchronize(); // not needed, as next call is cuda call and will follow sequentially on device
    nrm_s = _rho * cml::blas_nrm2(hdl, &ztemp);

    cml::vector_memcpy(&ztemp, &z12);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp);
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
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);
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
      } // end adaptive_rho==1
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
           _rel_tol * gap / eps_gap,_rel_tol,_gap_stop);
  }

  
  // Scale x, y, lambda and mu for output.
  PUSH_RANGE("Scale",Scale,1);
  cml::vector_memcpy(&ztemp, &zt);
  cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
  cml::blas_axpy(hdl, kOne, &z12, &ztemp);
  cml::blas_scal(hdl, -_rho, &ztemp);
  cml::vector_mul(&ytemp, &d);
  cml::vector_div(&xtemp, &e);

  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);
  POP_RANGE("Scale",Scale,1);

  // Copy results to output.
  PUSH_RANGE("Copy",Copy,1);
  cml::vector_memcpy(_x, &x12);
  cml::vector_memcpy(_y, &y12);
  cml::vector_memcpy(_mu, &xtemp);
  cml::vector_memcpy(_lambda, &ytemp);

  // Store z.
  cml::vector_memcpy(&z, &zprev);

  // Free memory.
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  cml::vector_free(&ztemp);
  cublasDestroy(hdl);
  CUDA_CHECK_ERR();
  POP_RANGE("Copy",Copy,1);

  //  POP_RANGE("PogsSolve",PogsSolve,1);

  return status;
}

template <typename T, typename M, typename P>
Pogs<T, M, P>::~Pogs() {
  CUDACHECK(cudaSetDevice(_wDev));

  cudaFree(_de);
  cudaFree(_z);
  cudaFree(_zt);
  _de = _z = _zt = 0;
  CUDA_CHECK_ERR();

#ifdef USE_NCCL2
  for(int i=0; i<_nDev; ++i)
    ncclCommDestroy(_comms[i]);
  free(_comms);
#endif
  
  delete [] _x;
  delete [] _y;
  delete [] _mu;
  delete [] _lambda;
  _x = _y = _mu = _lambda = 0;
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

