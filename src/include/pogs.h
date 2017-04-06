#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
#endif

// Check CUDA calls
#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

#include <sstream>

#ifdef USE_NCCL
#include "nccl.h"

#include <curand.h>
#include <cerrno>
#include <string>

// Propagate errors up
#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
      printf("NCCL failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,ncclGetErrorString(r));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)


#endif // end if USE_NCCL defined



#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff , 0x00f0ffff , 0x000fffff  , 0x00f0f0ff , 0x000ff0f0 };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

// whether push/pop timer is enabled (1) or not (0)
//#define PUSHPOPTIMER 1

#define PUSH_RANGE(name,tid,cid) \
  { \
    fprintf(stderr,"START: name=%s cid=%d\n",name,cid); fflush(stderr); \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
  }\
    double timer##tid = timer<double>();

#define POP_RANGE(name,tid,cid) {                                       \
    fprintf(stderr,"STOP:  name=%s cid=%d duration=%g\n",name,cid,timer<double>() - timer##tid); fflush(stderr); \
    nvtxRangePop(); \
  }
#else
#define PUSH_RANGE(name,tid,cid)
#define POP_RANGE(name,tid,cid)
#endif


#ifndef POGS_H_
#define POGS_H_

#include <cstring>
#include <string>
#include <vector>

#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "prox_lib.h"


namespace pogs {

static const std::string POGS_VERSION = "0.2.0";

// TODO: Choose default constants better
// Defaults.
const double       kAbsTol      = 1e-4;
const double       kRelTol      = 1e-3;
const double       kRhoInit     = 1.;
const unsigned int kVerbose     = 2u;   // 0...4
const unsigned int kMaxIter     = 2500u;
const unsigned int kInitIter    = 10u;
const bool         kAdaptiveRho = true;
const bool         kEquil       = true;
const bool         kGapStop     = false;
const int          knDev        = 1;
const int          kwDev        = 0;

// Status messages
enum PogsStatus { POGS_SUCCESS,    // Converged successfully.
                  POGS_INFEASIBLE, // Problem likely infeasible.
                  POGS_UNBOUNDED,  // Problem likely unbounded
                  POGS_MAX_ITER,   // Reached max iter.
                  POGS_NAN_FOUND,  // Encountered nan.
                  POGS_ERROR };    // Generic error, check logs.


// Proximal Operator Graph Solver.
template <typename T, typename M, typename P>
class Pogs {
 private:
  // Data
  M _A;
  P _P;
  T *_de, *_z, *_zt, _rho;
  bool _done_init;

  // cuda number of devices and which device(s) to use
  int _nDev,_wDev;
  // NCCL communicator
#ifdef USE_NCCL
  ncclComm_t* _comms;
#endif

  // Setup matrix _A and solver _LS
  int _Init();

  // Output.
  T *_x, *_y, *_mu, *_lambda, _optval, _time;
  unsigned int _final_iter;

  // Parameters.
  T _abs_tol, _rel_tol;
  unsigned int _max_iter, _init_iter, _verbose;
  bool _adaptive_rho, _equil, _gap_stop, _init_x, _init_lambda;

 public:
  // Constructor and Destructor.
  Pogs(int wDev, const M &A);
  ~Pogs();
  
  // Solve for specific objective.
  PogsStatus Solve(const std::vector<FunctionObj<T> >& f,
                   const std::vector<FunctionObj<T> >& g);

  // Getters for solution variables and parameters.
  const T*     GetX()           const { return _x; }
  const T*     GetY()           const { return _y; }
  const T*     GetLambda()      const { return _lambda; }
  const T*     GetMu()          const { return _mu; }
  T            GetOptval()      const { return _optval; }
  unsigned int GetFinalIter()   const { return _final_iter; }
  T            GetRho()         const { return _rho; }
  T            GetRelTol()      const { return _rel_tol; }
  T            GetAbsTol()      const { return _abs_tol; }
  unsigned int GetMaxIter()     const { return _max_iter; }
  unsigned int GetInitIter()    const { return _init_iter; }
  unsigned int GetVerbose()     const { return _verbose; }
  bool         GetAdaptiveRho() const { return _adaptive_rho; }
  bool         GetEquil()       const { return _equil; }
  bool         GetGapStop()     const { return _gap_stop; }
  T            GetTime()        const { return _time; }
  int          GetnDev()        const { return _nDev; }
  int          GetwDev()        const { return _wDev; }

  void printMe(std::ostream &os) const {
    os << "Model parameters: ";
    std::string sep = ", ";
    os << "lambda: " << _lambda << sep;
    os << "rho: " << _rho << sep;
    os << "rel_tol: " << _rel_tol << sep;
    os << "abs_tol: " << _abs_tol << sep;
    os << "max_iter: " << _max_iter << sep;
    os << "init_iter: " << _init_iter << sep;
    os << "verbose: " << _verbose << sep;
    os << "adaptive_rho: " << _adaptive_rho << sep;
    os << "equil: " << _equil << sep;
    os << "gap_stop: " << _gap_stop << sep;
    os << "nDev: " << _nDev << sep;
    os << "wDev: " << _wDev;
    os << std::endl;
  }

  // Setters for parameters and initial values.
  void SetRho(T rho)                       { _rho = rho; }
  void SetAbsTol(T abs_tol)                { _abs_tol = abs_tol; }
  void SetRelTol(T rel_tol)                { _rel_tol = rel_tol; }
  void SetMaxIter(unsigned int max_iter)   { _max_iter = max_iter; }
  void SetInitIter(unsigned int init_iter) { _init_iter = init_iter; }
  void SetVerbose(unsigned int verbose)    { _verbose = verbose; }
  void SetAdaptiveRho(bool adaptive_rho)   { _adaptive_rho = adaptive_rho; }
  void SetEquil(bool equil)                { _equil = equil; }
  void SetGapStop(bool gap_stop)           { _gap_stop = gap_stop; }
  void SetnDev(int nDev)    { _nDev = nDev; }
  void SetwDev(int wDev)    { _wDev = wDev; }
  void SetInitX(const T *x) {
    memcpy(_x, x, _A.Cols() * sizeof(T));
    _init_x = true;
  }
  void SetInitLambda(const T *lambda) {
    memcpy(_lambda, lambda, _A.Rows() * sizeof(T));
    _init_lambda = true;
  }
};

// Templated typedefs
#ifndef __CUDACC__
template <typename T, typename M>
using PogsDirect = Pogs<T, M, ProjectorDirect<T, M> >;

template <typename T, typename M>
using PogsIndirect = Pogs<T, M, ProjectorCgls<T, M> >;
#endif

// String version of status message.
inline std::string PogsStatusString(PogsStatus status) {
  switch(status) {
    case POGS_SUCCESS:
      return "Solved";
    case POGS_UNBOUNDED:
      return "Unbounded";
    case POGS_INFEASIBLE:
      return "Infeasible";
    case POGS_MAX_ITER:
      return "Reached max iter";
    case POGS_NAN_FOUND:
      return "Encountered NaN";
    case POGS_ERROR:
    default:
      return "Error";  
  }
}

}  // namespace pogs

#endif  // POGS_H_

