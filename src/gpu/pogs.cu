#include "pogs.h"

#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "equil_helper.cuh"
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
int PogsImplementation<T, M, P>::_Init(const PogsObjective<T> *objective) {
  DEBUG_EXPECT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;

  size_t m = _A.Rows();
  size_t n = _A.Cols();

  cudaMalloc(&_de, (m + n) * sizeof(T));
  cudaMalloc(&_z, (m + n) * sizeof(T));
  cudaMalloc(&_zt, (m + n) * sizeof(T));
  cudaMemset(_de, 0, (m + n) * sizeof(T));
  cudaMemset(_z, 0, (m + n) * sizeof(T));
  cudaMemset(_zt, 0, (m + n) * sizeof(T));
  CUDA_CHECK_ERR();

  cublasHandle_t hdl;
  cublasCreate(&hdl);
  CUDA_CHECK_ERR();

  _A.Init();
  _A.Equil(_de, _de + m,
           std::function<void(T*)>([objective](T *v){
               objective->constrain_d(v);
           }),
           std::function<void(T*)>([objective](T *v){
               objective->constrain_e(v);
           }));
  _nrmA = Norm2Est(hdl, &_A);
  CUDA_CHECK_ERR();

  _P.Init();
  CUDA_CHECK_ERR();

  cublasDestroy(hdl);
  return 0;
}

template <typename T, typename M, typename P>
PogsStatus PogsImplementation<T, M, P>::Solve(PogsObjective<T> *objective) {
  double t0 = timer<double>();
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin       = static_cast<T>(1.05);
  const T kGamma          = static_cast<T>(1.01);
  const T kTau            = static_cast<T>(0.8);
  const T kAlpha          = static_cast<T>(1.7);
  const T kRhoMin         = static_cast<T>(1e-4);
  const T kRhoMax         = static_cast<T>(1e4);
  const T kKappa          = static_cast<T>(0.9);
  const T kOne            = static_cast<T>(1);
  const T kZero           = static_cast<T>(0);
  const T kProjTolMax     = static_cast<T>(1e-8);
  const T kProjTolMin     = static_cast<T>(1e-2);
  const T kProjTolPow     = static_cast<T>(2);
  const T kProjTolIni     = static_cast<T>(1e-5);
  const bool kUseExactTol = false;

  // Initialize Projector P and Matrix A.
  if (!_done_init)
    _Init(objective);

  // Extract values from pogs_data
  size_t m = _A.Rows();
  size_t n = _A.Cols();

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

  // Scale objective to account for diagonal scaling e and d.
  objective->scale(d.data, e.data);
  CUDA_CHECK_ERR();

  // Initialize (x, lambda) from (x0, lambda0).
  if (_init_x) {
    cml::vector_memcpy(&xtemp, _x);
    cml::vector_div(&xtemp, &e);
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data);
    cudaDeviceSynchronize();
    cml::vector_memcpy(&z, &ztemp);
    CUDA_CHECK_ERR();
  }
  if (_init_lambda) {
    cml::vector_memcpy(&ytemp, _lambda);
    cml::vector_div(&ytemp, &d);
    _A.Mul('t', -kOne, ytemp.data, kZero, xtemp.data);
    cudaDeviceSynchronize();
    cml::blas_scal(hdl, -kOne / _rho, &ztemp);
    cml::vector_memcpy(&zt, &ztemp);
    CUDA_CHECK_ERR();
  }

  // Make an initial guess for (x0 or lambda0).
  if (_init_x && !_init_lambda) {
    // Alternating projections to satisfy
    //   1. \lambda \in \partial f(y), \mu \in \partial g(x)
    //   2. \mu = -A^T\lambda
    cml::vector_set_all(&zprev, kZero);
    for (unsigned int i = 0; i < kInitIter; ++i) {
//      ProjSubgradEval(g_gpu, xprev.data, x.data, xtemp.data);
//      ProjSubgradEval(f_gpu, yprev.data, y.data, ytemp.data);
      _P.Project(xtemp.data, ytemp.data, kOne, xprev.data, yprev.data,
          kProjTolIni);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERR();
      cml::blas_axpy(hdl, -kOne, &ztemp, &zprev);
      cml::blas_scal(hdl, -kOne, &zprev);
    }
    // xt = -1 / \rho * \mu, yt = -1 / \rho * \lambda.
    cml::vector_memcpy(&zt, &zprev);
    cml::blas_scal(hdl, -kOne / _rho, &zt);
  } else if (_init_lambda && !_init_x) {
    ASSERT(false);
  }
  _init_x = _init_lambda = false;

  // Save initialization time.
  double time_init = timer<double>() - t0;

  // Signal start of execution.
  if (_verbose > 0) {
    Printf(__HBAR__
        "           POGS v%s - Proximal Graph Solver (GPU)                \n"
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
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    cml::blas_axpy(hdl, -kOne, &zt, &z);
    objective->prox(x.data, y.data, x12.data, y12.data, _rho);
    CUDA_CHECK_ERR();

    // Compute gap, optval, and tolerances.
    cml::blas_axpy(hdl, -kOne, &z12, &z);
    cml::blas_dot(hdl, &z, &z12, &gap);
    gap = std::abs(gap);
    eps_gap = sqrtmn_atol + _rel_tol * cml::blas_nrm2(hdl, &z) *
        cml::blas_nrm2(hdl, &z12);
    eps_pri = sqrtm_atol + _rel_tol * cml::blas_nrm2(hdl, &y12);
    eps_dua = _rho * (sqrtn_atol + _rel_tol * cml::blas_nrm2(hdl, &x));
    CUDA_CHECK_ERR();

    // Apply over relaxation.
    cml::vector_memcpy(&ztemp, &zt);
    cml::blas_axpy(hdl, kAlpha, &z12, &ztemp);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &ztemp);
    CUDA_CHECK_ERR();

    // Project onto y = Ax.
    T proj_tol = kProjTolMin / std::pow(static_cast<T>(k + 1), kProjTolPow);
    proj_tol = std::max(proj_tol, kProjTolMax);
    _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    // Calculate residuals.
    cml::vector_memcpy(&ztemp, &zprev);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    cudaDeviceSynchronize();
    nrm_s = _rho * (_nrmA * cml::blas_nrm2(hdl, &ytemp) +
        cml::blas_nrm2(hdl, &xtemp));

    cml::vector_memcpy(&ztemp, &z12);
    cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    cudaDeviceSynchronize();
    nrm_r = _nrmA * cml::blas_nrm2(hdl, &xtemp) + cml::blas_nrm2(hdl, &ytemp);

    // Calculate exact residuals only if necessary.
    bool exact = false;
    if ((nrm_r < 10 * eps_pri && nrm_s < 10 * eps_dua) || kUseExactTol) {
      cml::vector_memcpy(&ztemp, &z12);
      _A.Mul('n', kOne, x12.data, -kOne, ytemp.data);
      cudaDeviceSynchronize();
      nrm_r = cml::blas_nrm2(hdl, &ytemp);
      cml::vector_memcpy(&ztemp, &z12);
      cml::blas_axpy(hdl, kOne, &zt, &ztemp);
      cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
      _A.Mul('t', kOne, ytemp.data, kOne, xtemp.data);
      cudaDeviceSynchronize();
      nrm_s = _rho * cml::blas_nrm2(hdl, &xtemp);
      exact = true;
    }
    CUDA_CHECK_ERR();

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!_gap_stop || gap < eps_gap);
    if (_verbose > 2 && k % 10  == 0 ||
        _verbose > 1 && k % 100 == 0 ||
        _verbose > 1 && converged) {
      T optval = objective->evaluate(x12.data, y12.data);
      Printf("%5d : %.2e  %.2e  %.2e  %.2e  %.2e  %.2e % .2e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);
    }

    // Break if converged or there are nans
    if (converged || k == _max_iter - 1){ // || cml::vector_any_isnan(&zt))
      _final_iter = k;
      break;
    }

    // Update dual variable.
    cml::blas_axpy(hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(hdl, -kOne, &z, &zt);
    CUDA_CHECK_ERR();

    // Rescale rho.
    if (_adaptive_rho) {
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
    }
  }

  // Get optimal value
  _optval = objective->evaluate(x12.data, y12.data);

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
  cml::vector_memcpy(&ztemp, &zt);
  cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
  cml::blas_axpy(hdl, kOne, &z12, &ztemp);
  cml::blas_scal(hdl, -_rho, &ztemp);
  cml::vector_mul(&ytemp, &d);
  cml::vector_div(&xtemp, &e);

  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);

  // Copy results to output.
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

  return status;
}

template <typename T, typename M, typename P>
PogsImplementation<T, M, P>::~PogsImplementation() {
  cudaFree(_de);
  cudaFree(_z);
  cudaFree(_zt);
  _de = _z = _zt = 0;
  CUDA_CHECK_ERR();

  delete [] _x;
  delete [] _y;
  delete [] _mu;
  delete [] _lambda;
  _x = _y = _mu = _lambda = 0;
}

// Pogs for separable problems
namespace {
template <typename T>
struct Divide {
  __host__ __device__ FunctionObj<T> operator()(FunctionObj<T> &fi, T di) {
    fi.a /= di; fi.d /= di; fi.e /= di * di; return fi;
  }
};

template <typename T>
struct Multiply {
  __host__ __device__ FunctionObj<T> operator()(FunctionObj<T> &gi, T ei) {
    gi.a *= ei; gi.d *= ei; gi.e *= ei * ei; return gi;
  }
};

template <typename T>
class PogsObjectiveSeparable : public PogsObjective<T> {
 private:
   thrust::device_vector<FunctionObj<T> > f, g;
 public:
  PogsObjectiveSeparable(const std::vector<FunctionObj<T> >& f,
                         const std::vector<FunctionObj<T> >& g)
      : f(f), g(g) { }

  T evaluate(const T *x, const T *y) const {
    return FuncEval(f, y) + FuncEval(g, x);
  }
  void prox(const T *x_in, const T *y_in, T *x_out, T *y_out, T rho) const {
    ProxEval(g, rho, x_in, x_out);
    ProxEval(f, rho, y_in, y_out);
  }

  void scale(const T *d, const T *e) {
    thrust::transform(f.begin(), f.end(), thrust::device_pointer_cast(d),
        f.begin(), Divide<T>());
    thrust::transform(g.begin(), g.end(), thrust::device_pointer_cast(e),
        g.begin(), Multiply<T>());
  }

  void constrain_d(T *d) const { }
  void constrain_e(T *e) const { }
};
}  // namespace

// Implementation of PogsSeparable
template <typename T, typename M, typename P>
PogsSeparable<T, M, P>::PogsSeparable(const M& A)
    : PogsImplementation<T, M, P>(A) { }

template <typename T, typename M, typename P>
PogsSeparable<T, M, P>::~PogsSeparable() { }

template <typename T, typename M, typename P>
PogsStatus PogsSeparable<T, M, P>::Solve(const std::vector<FunctionObj<T>>& f,
                                         const std::vector<FunctionObj<T>>& g) {
  PogsObjectiveSeparable<T> pogs_obj(f, g);
  return this->PogsImplementation<T, M, P>::Solve(&pogs_obj);
}

// Pogs for cone problems
namespace {

template <typename T>
struct Square : thrust::unary_function<T, T> {
  inline __host__ __device__ T operator()(const T &x) const {
    return x * x;
  }
};

template <typename T>
struct Updater {
  T rho;
  Updater(T rho) : rho(rho) { }
  __host__ __device__ T operator()(T ci, T xi) { return xi - ci / rho; };
};

template <typename T>
class PogsObjectiveCone : public PogsObjective<T> {
 private:
  T c_scale;
  thrust::device_vector<T> b, c;
  const std::vector<ConeConstraintRaw> &Kx, &Ky;
  std::vector<cudaStream_t> streams_x, streams_y;

 public:
  PogsObjectiveCone(const std::vector<T>& b,
                    const std::vector<T>& c,
                    const std::vector<ConeConstraintRaw>& Kx,
                    const std::vector<ConeConstraintRaw>& Ky)
      : b(b), c(c), Kx(Kx), Ky(Ky) {
    streams_x.resize(Kx.size());
    streams_y.resize(Ky.size());
    for (auto &stream : streams_x) {
      cudaStreamCreate(&stream);
    }
    for (auto &stream : streams_y) {
      cudaStreamCreate(&stream);
    }
  }

  ~PogsObjectiveCone() {
    for (auto &stream : streams_x) {
      cudaStreamDestroy(stream);
    }
    for (auto &stream : streams_y) {
      cudaStreamDestroy(stream);
    }
  }

  T evaluate(const T *x, const T*) const {
    return thrust::inner_product(c.begin(), c.end(),
        thrust::device_pointer_cast(x), static_cast<T>(0)) / c_scale;
  }

  void prox(const T *x_in, const T *y_in, T *x_out, T *y_out, T rho) const {
    cudaMemcpy(x_out, x_in, c.size() * sizeof(T), cudaMemcpyDeviceToDevice);
    thrust::transform(c.begin(), c.end(), thrust::device_pointer_cast(x_out),
        thrust::device_pointer_cast(x_out), Updater<T>(rho));

    cudaMemcpy(y_out, y_in, b.size() * sizeof(T), cudaMemcpyDeviceToDevice);
    thrust::transform(b.begin(), b.end(), thrust::device_pointer_cast(y_out),
        thrust::device_pointer_cast(y_out), thrust::minus<T>());
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    ProxEvalConeGpu(Kx, streams_x, c.size(), x_out, x_out);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    ProxEvalConeGpu(Ky, streams_y, b.size(), y_out, y_out);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    thrust::transform(b.begin(), b.end(), thrust::device_pointer_cast(y_out),
        thrust::device_pointer_cast(y_out), thrust::minus<T>());
  }

  void scale(const T *d, const T *e) {
    thrust::transform(c.begin(), c.end(), thrust::device_pointer_cast(e),
        c.begin(), thrust::multiplies<T>());
    thrust::transform(b.begin(), b.end(), thrust::device_pointer_cast(d),
        b.begin(), thrust::multiplies<T>());

    c_scale = 1 / std::sqrt(thrust::transform_reduce(c.begin(), c.end(),
        Square<T>(), static_cast<T>(0), thrust::plus<T>()));

    thrust::transform(c.begin(), c.end(),
        thrust::constant_iterator<T>(c_scale), c.begin(),
        thrust::multiplies<T>());
  }

  // Average the e_i in Kx
  void constrain_e(T *e) const {
    for (auto& cone : Kx) {
      if (IsSeparable(cone.cone))
        continue;

      using ValIt = thrust::device_ptr<T>;
      using IdxIt = thrust::device_ptr<CONE_IDX>;

      thrust::permutation_iterator<ValIt, IdxIt> iter(
          thrust::device_pointer_cast(e),
          thrust::device_pointer_cast(cone.idx));
      T sum = thrust::reduce(iter, iter + cone.size);
      thrust::fill(iter, iter + cone.size, sum / cone.size);
      CUDA_CHECK_ERR();
    }
  }

  // Average the d_i in Ky
  void constrain_d(T *d) const {
    for (auto& cone : Ky) {
      if (IsSeparable(cone.cone))
        continue;

      using ValIt = thrust::device_ptr<T>;
      using IdxIt = thrust::device_ptr<CONE_IDX>;

      thrust::permutation_iterator<ValIt, IdxIt> iter(
          thrust::device_pointer_cast(d),
          thrust::device_pointer_cast(cone.idx));
      T sum = thrust::reduce(iter, iter + cone.size);
      thrust::fill(iter, iter + cone.size, sum / cone.size);
      CUDA_CHECK_ERR();
    }
  }
};

void MakeRawCone(const std::vector<ConeConstraint> &K,
                 std::vector<ConeConstraintRaw> *K_raw) {
  for (const auto& cone_constraint : K) {
    ConeConstraintRaw raw;
    raw.size = cone_constraint.idx.size();
    cudaMalloc(&(raw.idx), raw.size * sizeof(CONE_IDX));
    cudaMemcpy(raw.idx, cone_constraint.idx.data(), raw.size * sizeof(CONE_IDX),
        cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR();
    raw.cone = cone_constraint.cone;
    K_raw->push_back(raw);
  }
}

}  // namespace

// Implementation of PogsCone
template <typename T, typename M, typename P>
PogsCone<T, M, P>::PogsCone(const M& A,
                            const std::vector<ConeConstraint>& Kx,
                            const std::vector<ConeConstraint>& Ky)
    : PogsImplementation<T, M, P>(A) , valid_cones(true) {
  valid_cones = ValidCone(Kx, A.Cols()) && ValidCone(Ky, A.Rows());
  MakeRawCone(Kx, &this->Kx);
  MakeRawCone(Ky, &this->Ky);
}

template <typename T, typename M, typename P>
PogsCone<T, M, P>::~PogsCone() {
  for (const auto& cone_constraint : this->Kx)
    cudaFree(cone_constraint.idx);
  for (const auto& cone_constraint : this->Ky)
    cudaFree(cone_constraint.idx);
}

template <typename T, typename M, typename P>
PogsStatus PogsCone<T, M, P>::Solve(const std::vector<T>& b,
                                    const std::vector<T>& c) {
  if (!valid_cones)
    return POGS_INVALID_CONE;
  PogsObjectiveCone<T> pogs_obj(b, c, Kx, Ky);
  return this->PogsImplementation<T, M, P>::Solve(&pogs_obj);
}

// Explicit template instantiation.
#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
// Dense direct.
template class PogsSeparable<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class PogsSeparable<double, MatrixDense<double>,
    ProjectorCgls<double, MatrixDense<double> > >;
template class PogsSeparable<double, MatrixSparse<double>,
    ProjectorCgls<double, MatrixSparse<double> > >;

template class PogsCone<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class PogsCone<double, MatrixDense<double>,
    ProjectorCgls<double, MatrixDense<double> > >;
template class PogsCone<double, MatrixSparse<double>,
    ProjectorCgls<double, MatrixSparse<double> > >;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class PogsSeparable<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;
template class PogsSeparable<float, MatrixDense<float>,
    ProjectorCgls<float, MatrixDense<float> > >;
template class PogsSeparable<float, MatrixSparse<float>,
    ProjectorCgls<float, MatrixSparse<float> > >;

template class PogsCone<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;
template class PogsCone<float, MatrixDense<float>,
    ProjectorCgls<float, MatrixDense<float> > >;
template class PogsCone<float, MatrixSparse<float>,
    ProjectorCgls<float, MatrixSparse<float> > >;
#endif

}  // namespace pogs

