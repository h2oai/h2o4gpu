#ifndef POGS_H_
#define POGS_H_

#include <vector>

#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "prox_lib.h"

namespace pogs {

// Defaults.
const double       kAbsTol      = 1e-9;
const double       kRelTol      = 1e-3;
const double       kRhoInit     = 1.;
const unsigned int kVerbose     = 2u;
const unsigned int kMaxIter     = 4000u;
const bool         kAdaptiveRho = true;
const bool         kGapStop     = false;
const bool         kInitX       = false;
const bool         kInitY       = false;

// Proximal Operator Graph Solver.
template <typename T, typename M, typename P>
class Pogs {
 private:
  // Data
  M _A;
  P _P;
  T *_de, *_z, *_zt, _rho;
  bool _done_init;

  // Setup matrix _A and solver _LS
  int _Init();

  // Output.
  T *_x, *_y, *_mu, *_lambda, _optval;

  // Parameters.
  T _abs_tol, _rel_tol;
  unsigned int _max_iter, _verbose;
  bool _adaptive_rho, _gap_stop, _init_x, _init_y;

 public:
  // Constructor and Destructor.
  Pogs(const M &A);
  ~Pogs();
  
  // Solve for specific objective.
  int Solve(const std::vector<FunctionObj<T> >& f,
            const std::vector<FunctionObj<T> >& g);

  // Getters for solution variables.
  const T* GetX()      const { return _x; }
  const T* GetY()      const { return _y; }
  const T* GetLambda() const { return _lambda; }
  const T* GetMu()     const { return _mu; }
  const T  GetOptval() const { return _optval; }

  // Setters for parameters and initial values.
  void SetRho(T rho)                     { _rho = rho; }
  void SetAbsTol(T abs_tol)              { _abs_tol = abs_tol; }
  void SetRelTol(T rel_tol)              { _rel_tol = rel_tol; }
  void SetMaxIter(unsigned int max_iter) { _max_iter = max_iter; }
  void SetVerbose(unsigned int verbose)  { _verbose = verbose; }
  void SetAdaptiveRho(bool adaptive_rho) { _adaptive_rho = adaptive_rho; }
  void SetGapStop(bool gap_stop)         { _gap_stop = gap_stop; }
  void SetX(const T *x) { memcpy(_x, x, _A.Cols() * sizeof(T)); }
  void SetY(const T *y) { memcpy(_y, y, _A.Rows() * sizeof(T)); }
};

#ifndef __CUDACC__
template <typename T, typename M>
using PogsDirect = Pogs<T, M, ProjectorDirect<T, M> >;

// Uncomment once CGLS has been implemented
//template <typename T, typename M>
//using PogsIndirect = Pogs<T, M, ProjectorCgls<T, M> >;
#endif

}  // namespace pogs

#endif  // POGS_H_

