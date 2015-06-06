#include "pogs.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "interface_defs.h"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector.h"
#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "util.h"

#include "timer.h"

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace pogs {

namespace {

template <typename T, typename Op>
struct ApplyOp: std::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x);
    h.d = binary_op(h.d, x);
    h.e = binary_op(binary_op(h.e, x), x);
    return h;
  }
};

template <typename T>
class PogsObjective {
 public:
  virtual T evaluate(const T *x, const T *y) const = 0;
  virtual void prox(const T *x_in, const T *y_in, T *x_out, T *y_out) const = 0;
  virtual void scale(const T *d, const T *e) = 0;
  virtual void average_equil(T *d, T *e) const = 0;
};

}  // namespace

template <typename T, typename M, typename P>
PogsImplementation<T, M, P>::PogsImplementation(const M &A)
    : _A(A), _P(_A),
      _de(0), _z(0), _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0), _y(0), _mu(0), _lambda(0), _optval(static_cast<T>(0.)),
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _gap_stop(kGapStop),
      _init_x(false), _init_lambda(false) {
  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
}

template <typename T, typename M, typename P>
int PogsImplementation<T, M, P>::_Init() {
  DEBUG_EXPECT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;

  size_t m = _A.Rows();
  size_t n = _A.Cols();

  _de = new T[m + n];
  ASSERT(_de != 0);
  _z = new T[m + n];
  ASSERT(_z != 0);
  _zt = new T[m + n];
  ASSERT(_zt != 0);
  memset(_de, 0, (m + n) * sizeof(T));
  memset(_z, 0, (m + n) * sizeof(T));
  memset(_zt, 0, (m + n) * sizeof(T));

  _A.Init();
  _A.Equil(_de, _de + m);
  _P.Init();

  return 0;
}

template <typename T, typename M, typename P>
PogsStatus PogsImplementation<T, M, P>::Solve(PogsObjective<T> *obj) {
  double t0 = timer<double>();
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin   = static_cast<T>(1.05);
  const T kGamma      = static_cast<T>(1.01);
  const T kTau        = static_cast<T>(0.8);
  const T kAlpha      = static_cast<T>(1.7);
  const T kRhoMin     = static_cast<T>(1e-4);
  const T kRhoMax     = static_cast<T>(1e4);
  const T kKappa      = static_cast<T>(0.9);
  const T kOne        = static_cast<T>(1.0);
  const T kZero       = static_cast<T>(0.0);
  const T kProjTolMax = static_cast<T>(1e-8);
  const T kProjTolMin = static_cast<T>(1e-2);
  const T kProjTolPow = static_cast<T>(1.3);
  const T kProjTolIni = static_cast<T>(1e-5);
  bool use_exact_stop = true;

  // Initialize Projector P and Matrix A.
  if (!_done_init)
    _Init();

  // Extract values from pogs_data
  size_t m = _A.Rows();
  size_t n = _A.Cols();

  // Allocate data for ADMM variables.
  gsl::vector<T> de    = gsl::vector_view_array(_de, m + n);
  gsl::vector<T> z     = gsl::vector_view_array(_z, m + n);
  gsl::vector<T> zt    = gsl::vector_view_array(_zt, m + n);
  gsl::vector<T> zprev = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> ztemp = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> z12   = gsl::vector_calloc<T>(m + n);

  // Create views for x and y components.
  gsl::vector<T> d     = gsl::vector_subvector(&de, 0, m);
  gsl::vector<T> e     = gsl::vector_subvector(&de, m, n);
  gsl::vector<T> x     = gsl::vector_subvector(&z, 0, n);
  gsl::vector<T> y     = gsl::vector_subvector(&z, n, m);
  gsl::vector<T> x12   = gsl::vector_subvector(&z12, 0, n);
  gsl::vector<T> y12   = gsl::vector_subvector(&z12, n, m);
  gsl::vector<T> xprev = gsl::vector_subvector(&zprev, 0, n);
  gsl::vector<T> yprev = gsl::vector_subvector(&zprev, n, m);
  gsl::vector<T> xtemp = gsl::vector_subvector(&ztemp, 0, n);
  gsl::vector<T> ytemp = gsl::vector_subvector(&ztemp, n, m);

  // Scale objective to account for diagonal scaling e and d.
  obj->scale(d, e);

  // Initialize (x, lambda) from (x0, lambda0).
  if (_init_x) {
    gsl::vector_memcpy(&xtemp, _x);
    gsl::vector_div(&xtemp, &e);
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data);
    gsl::vector_memcpy(&z, &ztemp);
  }
  if (_init_lambda) {
    gsl::vector_memcpy(&ytemp, _lambda);
    gsl::vector_div(&ytemp, &d);
    _A.Mul('t', -kOne, ytemp.data, kZero, xtemp.data);
    gsl::blas_scal(-kOne / _rho, &ztemp);
    gsl::vector_memcpy(&zt, &ztemp);
  }

  // Make an initial guess for (x0 or lambda0).
  if (_init_x && !_init_lambda) {
    // Alternating projections to satisfy
    //   1. \lambda \in \partial f(y), \mu \in \partial g(x)
    //   2. \mu = -A^T\lambda
    gsl::vector_set_all(&zprev, kZero);
    for (unsigned int i = 0; i < kInitIter; ++i) {
      ProjSubgradEval(g_cpu, xprev.data, x.data, xtemp.data);
      ProjSubgradEval(f_cpu, yprev.data, y.data, ytemp.data);
      _P.Project(xtemp.data, ytemp.data, kOne, xprev.data, yprev.data,
          kProjTolIni);
      gsl::blas_axpy(-kOne, &ztemp, &zprev);
      gsl::blas_scal(-kOne, &zprev);
    }
    // xt = -1 / \rho * \mu, yt = -1 / \rho * \lambda.
    gsl::vector_memcpy(&zt, &zprev);
    gsl::blas_scal(-kOne / _rho, &zt);
  } else if (_init_lambda && !_init_x) {
    ASSERT(false);
  }
  _init_x = _init_lambda = false;

  // Save initialization time.
  double time_init = timer<double>() - t0;

  // Signal start of execution.
  if (_verbose > 0) {
    Printf(__HBAR__
        "           POGS v%s - Proximal Graph Solver                      \n"
        "           (c) Christopher Fougner, Stanford University 2014-2015\n",
        POGS_VERSION);
  }
  if (_verbose > 1) {
    Printf(__HBAR__
        " Iter | pri res | pri tol | dua res | dua tol |   gap   | eps gap |"
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

  for (;; ++k) {
    gsl::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    gsl::blas_axpy(-kOne, &zt, &z);
    obj->prox(x.data, y.data, x12.data, y12.data, rho);

    // Compute gap, optval, and tolerances.
    gsl::blas_axpy(-kOne, &z12, &z);
    gsl::blas_dot(&z, &z12, &gap);
    gap = std::abs(gap);
    eps_gap = sqrtmn_atol + _rel_tol * gsl::blas_nrm2(&z) *
        gsl::blas_nrm2(&z12);
    eps_pri = sqrtm_atol + _rel_tol * gsl::blas_nrm2(&y12);
    eps_dua = sqrtn_atol + _rel_tol * _rho * gsl::blas_nrm2(&x);

    // Apply over relaxation.
    gsl::vector_memcpy(&ztemp, &zt);
    gsl::blas_axpy(kAlpha, &z12, &ztemp);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &ztemp);

    // Project onto y = Ax.
    T proj_tol = kProjTolMin / std::pow(static_cast<T>(k + 1), kProjTolPow);
    proj_tol = std::max(proj_tol, kProjTolMax);
    _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);

    // Calculate residuals.
    gsl::vector_memcpy(&ztemp, &zprev);
    gsl::blas_axpy(-kOne, &z, &ztemp);
    nrm_s = _rho * gsl::blas_nrm2(&ztemp);

    gsl::vector_memcpy(&ztemp, &z12);
    gsl::blas_axpy(-kOne, &z, &ztemp);
    nrm_r = gsl::blas_nrm2(&ztemp);

    // Calculate exact residuals only if necessary.
    bool exact = false;
    if ((nrm_r < eps_pri && nrm_s < eps_dua) || use_exact_stop) {
      gsl::vector_memcpy(&ztemp, &z12);
      _A.Mul('n', kOne, x12.data, -kOne, ytemp.data);
      nrm_r = gsl::blas_nrm2(&ytemp);
      if ((nrm_r < eps_pri) || use_exact_stop) {
        gsl::vector_memcpy(&ztemp, &z12);
        gsl::blas_axpy(kOne, &zt, &ztemp);
        gsl::blas_axpy(-kOne, &zprev, &ztemp);
        _A.Mul('t', kOne, ytemp.data, kOne, xtemp.data);
        nrm_s = _rho * gsl::blas_nrm2(&xtemp);
        exact = true;
      }
    }

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!_gap_stop || gap < eps_gap);
    if ((_verbose > 2 && k % 10  == 0) ||
        (_verbose > 1 && k % 100 == 0) ||
        (_verbose > 1 && converged)) {
      T optval = obj->eval(x12.data, y12.data);
      Printf("%5d : %.2e  %.2e  %.2e  %.2e  %.2e  %.2e % .2e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);
    }

    // Break if converged or there are nans
    if (converged || k == _max_iter - 1){
      _final_iter = k;
      break;
    }

    // Update dual variable.
    gsl::blas_axpy(kAlpha, &z12, &zt);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &zt);
    gsl::blas_axpy(-kOne, &z, &zt);

    // Rescale rho.
    if (_adaptive_rho) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (_rho < kRhoMax) {
          _rho *= delta;
          gsl::blas_scal(1 / delta, &zt);
          delta = kGamma * delta;
          ku = k;
          if (_verbose > 3)
            Printf("+ rho %e\n", _rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (_rho > kRhoMin) {
          _rho /= delta;
          gsl::blas_scal(delta, &zt);
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
    }
  }

  // Get optimal value
  _optval = obj->eval(x12.data, y12.data);

  // Check status
  PogsStatus status;
  if (!converged && k == _max_iter - 1)
    status = POGS_MAX_ITER;
  else if (!converged && k < _max_iter - 1)
    status = POGS_NAN_FOUND;
  else
    status = POGS_SUCCESS;

  // Print summary
  if (_verbose > 0) {
    Printf(__HBAR__
        "Status: %s\n"
        "Timing: Total = %3.2e s, Init = %3.2e s\n"
        "Iter  : %u\n",
        PogsStatusString(status).c_str(), timer<double>() - t0, time_init, k);
    Printf(__HBAR__
        "Error Metrics:\n"
        "Pri: "
        "|Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = %.2e\n"
        "Dua: "
        "|A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = %.2e\n"
        "Gap: "
        "|x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = %.2e\n"
        __HBAR__, _rel_tol * nrm_r / eps_pri, _rel_tol * nrm_s / eps_dua,
        _rel_tol * gap / eps_gap);
  }

  // Scale x, y, lambda and mu for output.
  gsl::vector_memcpy(&ztemp, &zt);
  gsl::blas_axpy(-kOne, &zprev, &ztemp);
  gsl::blas_axpy(kOne, &z12, &ztemp);
  gsl::blas_scal(-_rho, &ztemp);
  gsl::vector_mul(&ytemp, &d);
  gsl::vector_div(&xtemp, &e);

  gsl::vector_div(&y12, &d);
  gsl::vector_mul(&x12, &e);

  // Copy results to output.
  gsl::vector_memcpy(_x, &x12);
  gsl::vector_memcpy(_y, &y12);
  gsl::vector_memcpy(_mu, &xtemp);
  gsl::vector_memcpy(_lambda, &ytemp);

  // Store z.
  gsl::vector_memcpy(&z, &zprev);

  // Free memory.
  gsl::vector_free(&z12);
  gsl::vector_free(&zprev);
  gsl::vector_free(&ztemp);

  return status;
}

template <typename T, typename M, typename P>
Pogs<T, M, P>::~Pogs() {
  delete [] _de;
  delete [] _z;
  delete [] _zt;
  _de = _z = _zt = 0;

  delete [] _x;
  delete [] _y;
  delete [] _mu;
  delete [] _lambda;
  _x = _y = _mu = _lambda = 0;
}

// Pogs for separable problems
namespace {
template <typename T>
class PogsObjectiveSeparable : public PogsObjective<T> {
 private:
   std::vector<FunctionObj<T> > f, g;
 public:
  PogsObjectiveSeparable(const std::vector<FunctionObj<T> >& f,
                         const std::vector<FunctionObj<T> >& g)
      : f(f), g(g) { }

  T evaluate(const T *x, const T *y) const {
    return FuncEval(f, y) + FuncEval(g, x);
  }
  void prox(const T *x_in, const T *y_in, T *x_out, T *y_out, T rho) const {
    ProxEval(g, _rho, x.data, x12.data);
    ProxEval(f, _rho, y.data, y12.data);
  }

  void scale(const T *d, const T *e) {
    auto divide = [](FunctionObj<T> fi, T di) {
      fi.a *= di; fi.d *= di; fi.e *= di * di; return fi;
    };
    std::transform(f.begin(), f.end(), d.data, f.begin(), divide);
    auto multiply = [](FunctionObj<T> gi, T ei) {
      gi.a /= ei; gi.d /= ei; gi.e /= ei * ei; return gi;
    };
    std::transform(g.begin(), g.end(), e.data, g.begin(), multiply);
  }
};
}  // namespace

// Implementation of PogsSeparable
template <typename T, typename M, typename P>
PogsSeparable<T, M, P>::PogsSeparable(const M& A) : PogsImplementation(A) { }

template <typename T, typename M, typename P>
PogsSeparable<T, M, P>::~PogsSeparable() { }

template <typename T, typename M, typename P>
PogsStatus PogsSeparable<T, M, P>::Solve(const std::vector<FunctionObj>& f,
                                         const std::vector<FunctionObj>& g) {
  PogsObjectiveSeparable pogs_obj(f, g);
  return PogsImplementation(&pogs_obj);
}

// Pogs for cone problems
namespace {
template <typename T>
class PogsObjectiveCone : public PogsObjective<T> {
 private:
  std::vector<T> b, c;
  const std::vector<ConeConstraint<T> > &Kx, &Ky;
 public:
  PogsObjectiveSeparable(const std::vector<T>& b,
                         const std::vector<T>& c,
                         const std::vector<ConeConstraint<T> >& Kx,
                         const std::vector<ConeConstraint<T> >& Ky)
      : b(b), c(c), Kx(Kx), Ky(Ky) { }

  T evaluate(const T *x, const T*) const {
    return std::inner_product(c.begin(), c.end(), x, static_cast<T>(0));
  }

  void prox(const T *x_in, const T *y_in, T *x_out, T *y_out, T rho) const {
    memcpy(x_out, x_in, c.size());
    auto x_updater = [=rho](T ci, T xi) { return xi + c / rho; };
    std::transform(c.begin(), c.end(), x_out, x_out, x_updater);

    memcpy(y_out, y_in, b.size());
    std::transform(b.begin(), b.end(), y_out, y_out, std::minus<T>());

    ProxEvalConeCpu(Kx, c.size(), x_in, x_out);
    ProxEvalConeCpu(Ky, b.size(), y_in, y_out);

    std::tranform(b.begin(), b.end(), y_out, y_out, std::minus<T>());
  }

  void scale(const T *d, const T *e) {
    std::transform(c.begin(), c.end(), e, c.begin(), std::multiply<T>());
    std::transform(b.begin(), b.end(), d, b.begin(), std::multiply<T>());
  }

  // TODO
  void average_equil(T *d, T *e) {
    // Average the e_i in Kx
    for (auto& cone : Kx) {
      T sum = static_cast<T>(0.);
      for (auto i : cone.idx)
        sum += e[i];
      for (auto i : cone.idx)
        e[i] = sum / cone.idx.size();
    }

    // Average the d_i in Ky
    for (auto& cone : Ky) {
      T sum = static_cast<T>(0.);
      for (auto i : cone.idx)
        sum += d[i];
      for (auto i : cone.idx)
        d[i] = sum / cone.idx.size();
    }
  }
};
}  // namespace

// Implementation of PogsCone
template <typename T, typename M, typename P>
PogsCone<T, M, P>::PogsCone(const M& A)
    : PogsImplementation(A), Kx(Kx), Ky(Ky) { }

template <typename T, typename M, typename P>
PogsCone<T, M, P>::~PogsCone() { }

template <typename T, typename M, typename P>
PogsStatus PogsCone<T, M, P>::Solve(const std::vector<T>& b,
                                    const std::vector<T>& c) {
  PogsObjectiveCone pogs_obj(b, c, Kx, Ky);
  return PogsImplementation(&pogs_obj);
}

// Explicit template instantiation.
// Dense direct.
template class PogsSeparable<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class PogsSeparable<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;

// Dense indirect.
template class PogsSeparable<double, MatrixDense<double>,
    ProjectorCgls<double, MatrixDense<double> > >;
template class PogsSeparable<float, MatrixDense<float>,
    ProjectorCgls<float, MatrixDense<float> > >;

// Sparse indirect.
template class PogsSeparable<double, MatrixSparse<double>,
    ProjectorCgls<double, MatrixSparse<double> > >;
template class PogsSeparable<float, MatrixSparse<float>,
    ProjectorCgls<float, MatrixSparse<float> > >;

// Dense direct.
template class PogsCone<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class PogsCone<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;

// Dense indirect.
template class PogsCone<double, MatrixDense<double>,
    ProjectorCgls<double, MatrixDense<double> > >;
template class PogsCone<float, MatrixDense<float>,
    ProjectorCgls<float, MatrixDense<float> > >;

// Sparse indirect.
template class PogsCone<double, MatrixSparse<double>,
    ProjectorCgls<double, MatrixSparse<double> > >;
template class PogsCone<float, MatrixSparse<float>,
    ProjectorCgls<float, MatrixSparse<float> > >;


}  // namespace pogs

