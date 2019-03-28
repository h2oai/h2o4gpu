/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#include "solver/glm.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>

#include <cstring>
#include <functional>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "interface_defs.h"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector.h"
#include "projector/projector_cgls.h"
#include "projector/projector_direct.h"
#include "util.h"

#include "timer.h"

namespace h2o4gpu {

namespace {

template <typename T, typename Op>
struct ApplyOp : std::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) {}
  FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x);
    h.d = binary_op(h.d, x);
    h.e = binary_op(binary_op(h.e, x), x);
    return h;
  }
};

}  // namespace

template <typename T, typename M, typename P>
H2O4GPU<T, M, P>::H2O4GPU(int sharedA, int me, int wDev, const M &A)
    : _A(sharedA, me, wDev, A),
      _P(wDev, _A),
      _z(0),
      _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0),
      _y(0),
      _mu(0),
      _lambda(0),
      _optval(static_cast<T>(0.)),
      _time(static_cast<T>(0.)),
      _trainPreds(0),
      _validPreds(0),
      _xp(0),
      _trainPredsp(0),
      _validPredsp(0),
      _trainerror(0),
      _validerror(0),
      _trainmean(0),
      _validmean(0),
      _trainstddev(0),
      _validstddev(0),
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _stop_early(1),
      _stop_early_error_fraction(1.0),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _equil(kEquil),
      _gap_stop(kGapStop),
      _init_x(false),
      _init_lambda(false),
      _nDev(0),
      _wDev(wDev) {
  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
  _trainPreds = new T[_A.Rows()]();
  _validPreds = new T[_A.ValidRows()]();
}

template <typename T, typename M, typename P>
H2O4GPU<T, M, P>::H2O4GPU(const M &A)
    : _A(A._sharedA, A._me, A._wDev, A),
      _P(_A._wDev, _A),
      _z(0),
      _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0),
      _y(0),
      _mu(0),
      _lambda(0),
      _optval(static_cast<T>(0.)),
      _time(static_cast<T>(0.)),
      _trainPreds(0),
      _validPreds(0),
      _xp(0),
      _trainPredsp(0),
      _validPredsp(0),
      _trainerror(0),
      _validerror(0),
      _trainmean(0),
      _validmean(0),
      _trainstddev(0),
      _validstddev(0),
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _stop_early(1),
      _stop_early_error_fraction(1.0),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _equil(kEquil),
      _gap_stop(kGapStop),
      _init_x(false),
      _init_lambda(false),
      _nDev(0),
      _wDev(_A._wDev) {
  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
  _trainPreds = new T[_A.Rows()]();
  _validPreds = new T[_A.ValidRows()]();
}

template <typename T, typename M, typename P>
int H2O4GPU<T, M, P>::_Init() {
  DEBUG_EXPECT(!_done_init);
  if (_done_init) return 1;
  _done_init = true;

  size_t m = _A.Rows();
  size_t n = _A.Cols();

  _z = new T[m + n];
  ASSERT(_z != 0);
  memset(_z, 0, (m + n) * sizeof(T));
  _zt = new T[m + n];
  ASSERT(_zt != 0);
  memset(_zt, 0, (m + n) * sizeof(T));

  _A.Init();
  _A.Equil(_equil);
  _P.Init();

  return 0;
}

template <typename T, typename M, typename P>
H2O4GPUStatus H2O4GPU<T, M, P>::Solve(const std::vector<FunctionObj<T> > &f,
                                      const std::vector<FunctionObj<T> > &g) {
  double t0 = timer<double>();
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kGamma = static_cast<T>(1.01);
  const T kTau = static_cast<T>(0.8);
  const T kAlpha = static_cast<T>(1.7);
  const T kKappa = static_cast<T>(0.4);
  const T kOne = static_cast<T>(1.0);
  const T kZero = static_cast<T>(0.0);
  const T kProjTolMax = static_cast<T>(1e-8);
  const T kProjTolMin = static_cast<T>(1e-2);
  const T kProjTolPow = static_cast<T>(1.3);
  const T kProjTolIni = static_cast<T>(1e-5);
  bool use_exact_stop = true;

  // Initialize Projector P and Matrix A.
  if (!_done_init) _Init();

  // Extract values from h2o4gpu_data
  size_t m = _A.Rows();
  size_t mvalid = _A.ValidRows();
  size_t n = _A.Cols();
  std::vector<FunctionObj<T> > f_cpu = f;
  std::vector<FunctionObj<T> > g_cpu = g;

  // Allocate data for ADMM variables.
  gsl::vector<T> de = gsl::vector_view_array(_A._de, m + n);
  gsl::vector<T> z = gsl::vector_view_array(_z, m + n);
  gsl::vector<T> zt = gsl::vector_view_array(_zt, m + n);
  gsl::vector<T> zprev = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> ztemp = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> z12 = gsl::vector_calloc<T>(m + n);

  // Create views for x and y components.
  gsl::vector<T> d = gsl::vector_subvector(&de, 0, m);
  gsl::vector<T> e = gsl::vector_subvector(&de, m, n);
  gsl::vector<T> x = gsl::vector_subvector(&z, 0, n);
  gsl::vector<T> y = gsl::vector_subvector(&z, n, m);
  gsl::vector<T> x12 = gsl::vector_subvector(&z12, 0, n);
  gsl::vector<T> y12 = gsl::vector_subvector(&z12, n, m);
  gsl::vector<T> xprev = gsl::vector_subvector(&zprev, 0, n);
  gsl::vector<T> yprev = gsl::vector_subvector(&zprev, n, m);
  gsl::vector<T> xtemp = gsl::vector_subvector(&ztemp, 0, n);
  gsl::vector<T> ytemp = gsl::vector_subvector(&ztemp, n, m);

  // Scale f and g to account for diagonal scaling e and d.
  std::transform(f_cpu.begin(), f_cpu.end(), d.data, f_cpu.begin(),
                 ApplyOp<T, std::divides<T> >(std::divides<T>()));
  std::transform(g_cpu.begin(), g_cpu.end(), e.data, g_cpu.begin(),
                 ApplyOp<T, std::multiplies<T> >(std::multiplies<T>()));

  // TODO: Need to give scale to these
  //  const T kRhoMin     = static_cast<T>(1e-4); // lower range for adaptive
  //  rho const T kRhoMax     = static_cast<T>(1e4); // upper range for
  //  adaptive rho
  const T kRhoMin = static_cast<T>(
      std::numeric_limits<T>::epsilon());  // lower range for adaptive rho
  const T kRhoMax =
      static_cast<T>(1.0 / kRhoMin);  // upper range for adaptive rho

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
#pragma omp critical
    {
      printMe(std::cout, f[1].a, f[1].b, f[1].c, f[1].d, f[1].e, g[1].a, g[1].b,
              g[1].c, g[1].d, g[1].e);
      //    printData(std::cout); //debugging - print matrix A
    }
  }
  if (_verbose > 1) {
#pragma omp critical
    {
      Printf(__HBAR__
             " Iter | pri res | pri tol | dua res | dua tol |   gap   | eps "
             "gap |"
             " pri obj\n" __HBAR__);
    }
  }

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * _abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * _abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * _abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int k = 0u, kd = 0u, ku = 0u;
  bool converged = false;
  T nrm_r, nrm_s, gap, eps_gap, eps_pri, eps_dua;

  // Stop early setup
  unsigned int QUEUELENGTH = 10;
  std::deque<T> nrm_r_deque;
  std::deque<T> nrm_s_deque;
  std::deque<T> nrm_r_avg;
  std::deque<T> nrm_s_avg;
  std::deque<T> nrm_r_error;
  std::deque<T> nrm_s_error;

  for (;; ++k) {
    gsl::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    gsl::blas_axpy(-kOne, &zt, &z);
    ProxEval(g_cpu, _rho, x.data, x12.data);
    ProxEval(f_cpu, _rho, y.data, y12.data);

    // Compute gap, optval, and tolerances.
    gsl::blas_axpy(-kOne, &z12, &z);
    gsl::blas_dot(&z, &z12, &gap);
    gap = std::abs(gap);
    eps_gap =
        sqrtmn_atol + _rel_tol * gsl::blas_nrm2(&z) * gsl::blas_nrm2(&z12);
    eps_pri = sqrtm_atol + _rel_tol * gsl::blas_nrm2(&y12);
    eps_dua = _rho * (sqrtn_atol + _rel_tol * gsl::blas_nrm2(&x));

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

    // STOP EARLY CHECK
    bool stopearly = false;
    if (_stop_early) {
      nrm_r_deque.push_back(nrm_r);
      nrm_s_deque.push_back(nrm_s);

      nrm_r_avg.push_back(
          std::accumulate(nrm_r_deque.begin(), nrm_r_deque.end(), 0.0) /
          static_cast<T>(nrm_r_deque.size()));
      nrm_s_avg.push_back(
          std::accumulate(nrm_s_deque.begin(), nrm_s_deque.end(), 0.0) /
          static_cast<T>(nrm_s_deque.size()));
      if (nrm_r_deque.size() >= QUEUELENGTH &&
          nrm_r_avg.size() >= QUEUELENGTH) {
        T errorlocal_r = 0;
        T errorlocal_s = 0;
        for (unsigned int ii = 0; ii < QUEUELENGTH; ii++) {
          errorlocal_r += std::abs(nrm_r_avg[ii] - nrm_r_deque[ii]);
          errorlocal_s += std::abs(nrm_s_avg[ii] - nrm_s_deque[ii]);
        }
        nrm_r_error.push_back(errorlocal_r / static_cast<T>(QUEUELENGTH));
        nrm_s_error.push_back(errorlocal_s / static_cast<T>(QUEUELENGTH));
      }

      if (k > QUEUELENGTH && nrm_r_deque.size() >= QUEUELENGTH &&
          nrm_r_avg.size() >= QUEUELENGTH &&
          nrm_s_deque.size() >= QUEUELENGTH &&
          nrm_s_avg.size() >= QUEUELENGTH && nrm_r_error.size() >= 1 &&
          nrm_s_error.size() >= 1 &&
          std::abs(nrm_r_avg.back() - nrm_r_avg.front()) < nrm_r_error.back() &&
          std::abs(nrm_s_avg.back() - nrm_s_avg.front()) < nrm_s_error.back()) {
        Printf("Stopped Early at iteration=%d: %g %g %g : %g %g %g\n", k,
               nrm_r_avg.back(), nrm_r_avg.front(), nrm_r_error.back(),
               nrm_s_avg.back(), nrm_s_avg.front(), nrm_s_error.back());
        fflush(stdout);
        stopearly = true;
      }

      if (nrm_r_deque.size() >= QUEUELENGTH) {
        nrm_r_deque.pop_front();
      }
      if (nrm_s_deque.size() >= QUEUELENGTH) {
        nrm_s_deque.pop_front();
      }
      if (nrm_r_avg.size() >= QUEUELENGTH) {
        nrm_r_avg.pop_front();
      }
      if (nrm_s_avg.size() >= QUEUELENGTH) {
        nrm_s_avg.pop_front();
      }
      if (nrm_r_error.size() >= QUEUELENGTH) {
        nrm_r_error.pop_front();
      }
      if (nrm_s_error.size() >= QUEUELENGTH) {
        nrm_s_error.pop_front();
      }
    }

    // Evaluate stopping criteria.
    converged = stopearly || (exact && nrm_r < eps_pri && nrm_s < eps_dua &&
                              (!_gap_stop || gap < eps_gap));
    if ((_verbose > 2 && k % 10 == 0) || (_verbose > 1 && k % 100 == 0) ||
        (_verbose > 1 && converged)) {
      T optval = FuncEval(f_cpu, y12.data) + FuncEval(g_cpu, x12.data);
      Printf("%5d : %.2e  %.2e  %.2e  %.2e  %.2e  %.2e % .2e\n", k, nrm_r,
             eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);
    }

    // Break if converged or there are nans
    if (converged || k == _max_iter - 1) {
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
          if (_verbose > 3) Printf("+ rho %e\n", _rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
                 kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (_rho > kRhoMin) {
          _rho /= delta;
          gsl::blas_scal(delta, &zt);
          delta = kGamma * delta;
          kd = k;
          if (_verbose > 3) Printf("- rho %e\n", _rho);
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = kDeltaMin;
      }
    }
  }

  // Get optimal value
  _optval = FuncEval(f_cpu, y12.data) + FuncEval(g_cpu, x12.data);

  // Check status
  H2O4GPUStatus status;
  if (!converged && k == _max_iter - 1)
    status = H2O4GPU_MAX_ITER;
  else if (!converged && k < _max_iter - 1)
    status = H2O4GPU_NAN_FOUND;
  else
    status = H2O4GPU_SUCCESS;

  // Get run time
  _time = static_cast<T>(timer<double>() - t0);

  // Print summary
  if (_verbose > 0) {
    Printf(__HBAR__
           "Status: %s\n"
           "Timing: Total = %3.2e s, Init = %3.2e s\n"
           "Iter  : %u\n",
           H2O4GPUStatusString(status).c_str(), _time, time_init, k);
    Printf(__HBAR__
           "Error Metrics:\n"
           "Pri: "
           "|Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = "
           "%.2e\n"
           "Dua: "
           "|A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = "
           "%.2e\n"
           "Gap: "
           "|x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = "
           "%.2e\n" __HBAR__,
           _rel_tol * nrm_r / eps_pri, _rel_tol * nrm_s / eps_dua,
           _rel_tol * gap / eps_gap);
  }

  // Scale x, y, lambda and mu for output.
  gsl::vector_memcpy(&ztemp, &zt);
  gsl::blas_axpy(-kOne, &zprev, &ztemp);
  gsl::blas_axpy(kOne, &z12, &ztemp);
  gsl::blas_scal(-_rho, &ztemp);
  gsl::vector_mul(&ytemp, &d);
  gsl::vector_div(&xtemp, &e);

  gsl::vector<T> x12copy = gsl::vector_calloc<T>(n);
  gsl::vector_memcpy(&x12copy, &x12);  // copy de version first

  gsl::vector_div(&y12, &d);
  gsl::vector_mul(&x12, &e);

  // Copy results to output.
  gsl::vector_memcpy(_x, &x12);
  gsl::vector_memcpy(_y, &y12);
  gsl::vector_memcpy(_mu, &xtemp);
  gsl::vector_memcpy(_lambda, &ytemp);

  //  for(unsigned int i=0;i<m;i++){
  //    for(unsigned int j=0;j<n;j++){
  //      fprintf(stderr,"A[%d,%d]=%g\n",i,j,_A._data[j+i*n]);
  //    }
  //  }
  //  for(unsigned int j=0;j<n;j++){
  //    fprintf(stderr,"x[%d]=%g\n",j,_x[j]);
  //  }

  // compute train predictions from trainPred = Atrain.xsolution
  // _data is B = D*A*E

  _A.Mul('n', static_cast<T>(1.0), x12copy.data, static_cast<T>(0.),
         _trainPreds);  // _x and _trainPreds are both pointers on CPU
  for (unsigned int i = 0; i < m; i++) {
    _trainPreds[i] /= d.data[i];
    //    fprintf(stderr,"Tp[%d]=%g\n",i,_trainPreds[i]);
  }
  if (mvalid > 0) {
    // compute valid from validPreds = Avalid.xsolution
    _A.Mulvalid('n', static_cast<T>(1.), _x, static_cast<T>(0.), _validPreds);
  }
  // compute error (not yet)

  // compute mean (not yet)

  // compute stddev (not yet)

  // Store z.
  gsl::vector_memcpy(&z, &zprev);

  // Free memory.
  gsl::vector_free(&z12);
  gsl::vector_free(&zprev);
  gsl::vector_free(&ztemp);
  gsl::vector_free(&x12copy);

  return status;
}

template <typename T, typename M, typename P>
int H2O4GPU<T, M, P>::_Init_Predict() {
  DEBUG_EXPECT(!_done_init);
  if (_done_init) return 1;
  _done_init = true;

  size_t m = _A.Rows();
  size_t n = _A.Cols();

  _A.Init();

  return 0;
}

template <typename T, typename M, typename P>
int H2O4GPU<T, M, P>::Predict(void) {
  double t0 = timer<double>();

  // Initialize
  if (!_done_init) _Init_Predict();

  // compute valid from validPreds = Avalid.xsolution
  _A.Mulvalid('n', static_cast<T>(1.), _x, static_cast<T>(0.), _validPreds);

  // compute error (not yet)

  // compute mean (not yet)

  // compute stddev (not yet)

  return 0;
}

template <typename T, typename M, typename P>
void H2O4GPU<T, M, P>::ResetX(void) {
  if (!_done_init) _Init();

  size_t m = _A.Rows();
  size_t mvalid = _A.ValidRows();
  size_t n = _A.Cols();

  memset(_z, 0, (m + n) * sizeof(T));
  memset(_zt, 0, (m + n) * sizeof(T));
}

template <typename T, typename M, typename P>
H2O4GPU<T, M, P>::~H2O4GPU() {
  if (_z) delete[] _z;
  if (_zt) delete[] _zt;
  _z = _zt = 0;

  if (_x) delete[] _x;
  if (_y) delete[] _y;
  if (_mu) delete[] _mu;
  if (_lambda) delete[] _lambda;
  if (_trainPreds) delete[] _trainPreds;
  if (_validPreds) delete[] _validPreds;
  _x = _y = _mu = _lambda = _trainPreds = _validPreds = 0;
}

// Explicit template instantiation.
#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE == 1
template class H2O4GPU<double, MatrixDense<double>,
                       ProjectorDirect<double, MatrixDense<double> > >;
template class H2O4GPU<double, MatrixDense<double>,
                       ProjectorCgls<double, MatrixDense<double> > >;
template class H2O4GPU<double, MatrixSparse<double>,
                       ProjectorCgls<double, MatrixSparse<double> > >;
#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE == 1
template class H2O4GPU<float, MatrixDense<float>,
                       ProjectorDirect<float, MatrixDense<float> > >;
template class H2O4GPU<float, MatrixDense<float>,
                       ProjectorCgls<float, MatrixDense<float> > >;
template class H2O4GPU<float, MatrixSparse<float>,
                       ProjectorCgls<float, MatrixSparse<float> > >;
#endif

}  // namespace h2o4gpu
