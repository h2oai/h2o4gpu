#ifndef POGS_H_
#define POGS_H_

#include <cstring>
#include <string>
#include <vector>

#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector_cgls.h"
#include "projector/projector_direct.h"
#include "prox_lib.h"
#include "prox_lib_cone.h"

namespace pogs {

static const char POGS_VERSION[] = "0.2.0";

// Defaults.
const double       kAbsTol       = 1e-4;
const double       kRelTol       = 1e-3;
const double       kRhoInit      = 1.;
const unsigned int kVerbose      = 2u;   // 0...4
const unsigned int kMaxIter      = 2500u;
const unsigned int kInitIter     = 10u;
const bool         kAdaptiveRho  = true;
const bool         kGapStop      = false;

// Status messages
enum PogsStatus { POGS_SUCCESS,    // Converged succesfully.
                  POGS_INFEASIBLE, // Problem likely infeasible.
                  POGS_UNBOUNDED,  // Problem likely unbounded
                  POGS_MAX_ITER,   // Reached max iter.
                  POGS_NAN_FOUND,  // Encountered nan.
                  POGS_ERROR };    // Generic error, check logs.

// Generic POGS objective.
template <typename T>
class PogsObjective {
 public:
  virtual T evaluate(const T *x, const T *y) const = 0;
  virtual void prox(const T *x_in, const T *y_in, T *x_out, T *y_out,
                    T rho) const = 0;
  virtual void scale(const T *d, const T *e) = 0;
  virtual void constrain_d(T *d) const = 0;
  virtual void constrain_e(T *e) const = 0;
};

// Proximal Operator Graph Solver.
template <typename T, typename M, typename P>
class PogsImplementation {
 protected:
  // Data
  M _A;
  P _P;
  T *_de, *_z, *_zt, _rho;
  bool _done_init;

  // Setup matrix _A and projector _P.
  int _Init(const PogsObjective<T> *obj);

  // Output.
  T *_x, *_y, *_mu, *_lambda, _optval;
  unsigned int _final_iter;

  // Parameters.
  T _abs_tol, _rel_tol;
  unsigned int _max_iter, _init_iter, _verbose;
  bool _adaptive_rho, _gap_stop, _init_x, _init_lambda;

  // Solver
  PogsStatus Solve(PogsObjective<T> *obj);

 public:
  // Constructor and Destructor.
  PogsImplementation(const M &A);
  ~PogsImplementation();

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
  bool         GetGapStop()     const { return _gap_stop; }


  // Setters for parameters and initial values.
  void SetRho(T rho)                       { _rho = rho; }
  void SetAbsTol(T abs_tol)                { _abs_tol = abs_tol; }
  void SetRelTol(T rel_tol)                { _rel_tol = rel_tol; }
  void SetMaxIter(unsigned int max_iter)   { _max_iter = max_iter; }
  void SetInitIter(unsigned int init_iter) { _init_iter = init_iter; }
  void SetVerbose(unsigned int verbose)    { _verbose = verbose; }
  void SetAdaptiveRho(bool adaptive_rho)   { _adaptive_rho = adaptive_rho; }
  void SetGapStop(bool gap_stop)           { _gap_stop = gap_stop; }
  void SetInitX(const T *x) {
    memcpy(_x, x, _A.Cols() * sizeof(T));
    _init_x = true;
  }
  void SetInitLambda(const T *lambda) {
    memcpy(_lambda, lambda, _A.Rows() * sizeof(T));
    _init_lambda = true;
  }
};

template <typename T, typename M, typename P>
class PogsSeparable : public PogsImplementation<T, M, P> {
 public:
  PogsSeparable(const M &A);
  ~PogsSeparable();

  // Solve for specific objective.
  PogsStatus Solve(const std::vector<FunctionObj<T>>& f,
                   const std::vector<FunctionObj<T>>& g);
};

template <typename T, typename M, typename P>
class PogsCone : public PogsImplementation<T, M, P> {
 public:
  PogsCone(const M &A,
           const std::vector<ConeConstraint>& Kx,
           const std::vector<ConeConstraint>& Ky);
  ~PogsCone();

  // Solve for specific objective.
  PogsStatus Solve(const std::vector<T>& b, const std::vector<T>& c);

 private:
  std::vector<ConeConstraintRaw> Kx;
  std::vector<ConeConstraintRaw> Ky;
};

// Templated typedefs
#ifndef __CUDACC__
template <typename T, typename M>
using PogsDirect = PogsSeparable<T, M, ProjectorDirect<T, M> >;

template <typename T, typename M>
using PogsIndirect = PogsSeparable<T, M, ProjectorCgls<T, M> >;

template <typename T, typename M>
using PogsDirectCone = PogsCone<T, M, ProjectorDirect<T, M> >;

template <typename T, typename M>
using PogsIndirectCone = PogsCone<T, M, ProjectorDirect<T, M> >;
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

