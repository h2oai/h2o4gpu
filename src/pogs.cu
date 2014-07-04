#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <vector>

#include "cml/cml_blas.cuh"
#include "cml/cml_linalg.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_vector.cuh"

#include "matrix_util.hpp"
#include "pogs.hpp"
#include "sinkhorn_knopp.cuh"

#ifdef __MEX__
#define printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);
#endif  // __MEX__

// Apply operator to h.a and h.d.
template <typename T, typename Op>
struct ApplyOp: thrust::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  __device__ FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x); h.d = binary_op(h.d, x);
    return h;
  }
};

// Proximal Operator Graph Solver.
template <typename T, typename M>
void Pogs(PogsData<T, M> *pogs_data) {
  // Extract values from pogs_data
  size_t m = pogs_data->m, n = pogs_data->n, min_dim = std::min(m, n);
  const T kOne = static_cast<T>(1), kZero = static_cast<T>(0);

  // Copy f and g to device
  thrust::device_vector<FunctionObj<T> > f = pogs_data->f;
  thrust::device_vector<FunctionObj<T> > g = pogs_data->g;

  // Create cuBLAS handle.
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate data for ADMM variables.
  cml::vector<T> d = cml::vector_calloc<T>(m);
  cml::vector<T> e = cml::vector_calloc<T>(n);
  cml::vector<T> z = cml::vector_calloc<T>(m + n);
  cml::vector<T> zt = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
  cml::vector<T> z_prev = cml::vector_calloc<T>(m + n);
  cml::matrix<T> L = cml::matrix_alloc<T>(min_dim, min_dim);
  cml::matrix<T> AA = cml::matrix_alloc<T>(min_dim, min_dim);
  cml::matrix<T> A = cml::matrix_alloc<T>(m, n);

  // Create views for x and y components.
  cml::vector<T> x = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y = cml::vector_subvector(&z, n, m);
  cml::vector<T> xt = cml::vector_subvector(&zt, 0, n);
  cml::vector<T> yt = cml::vector_subvector(&zt, n, m);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n, m);

  // Copy A to device (assume input row-major).
  T *Acm = new T[m * n];
  RowToColMajor(pogs_data->A, m, n, Acm);
  SinkhornKnopp(handle, Acm, &A, &d, &e);
  delete [] Acm;

  // Compuate A^TA or AA^T.
  cublasOperation_t op_type = m >= n ? CUBLAS_OP_T : CUBLAS_OP_N;
  cml::blas_syrk(handle, CUBLAS_FILL_MODE_LOWER, op_type, kOne, &A, kZero, &AA);

  // Scale A.
  cml::vector<T> diag_AA = cml::matrix_diagonal(&AA);
  T mean_diag = cml::blas_asum(handle, &diag_AA) / static_cast<T>(min_dim);
  T sqrt_mean_diag = sqrt(mean_diag);
  cml::matrix_scale(&AA, kOne / mean_diag);
  cml::matrix_scale(&A, kOne / sqrt_mean_diag);
  cml::vector_scale(&e, kOne / sqrt(sqrt_mean_diag));
  cml::vector_scale(&d, kOne / sqrt(sqrt_mean_diag));

  // Compute cholesky decomposition of (I + A^TA) or (I + AA^T)
  cml::matrix_memcpy(&L, &AA);
  cml::vector<T> diag_L = cml::matrix_diagonal(&L);
  cml::vector_add_constant(&diag_L, kOne);
  cml::linalg_cholesky_decomp(handle, &L);

  // Scale f and g to account for diagonal scaling e and d.
  thrust::transform(f.begin(), f.end(), thrust::device_pointer_cast(d.data),
      f.begin(), ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
  thrust::transform(g.begin(), g.end(), thrust::device_pointer_cast(e.data),
      g.begin(), ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));

  // Signal start of execution.
  if (!pogs_data->quiet)
    printf("%4s %12s %10s %10s %10s %10s %10s\n",
           "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective", "gap");

  T rho = pogs_data->rho;
  T sqrtn_atol = sqrt(static_cast<T>(n)) * pogs_data->abs_tol;

  for (unsigned int k = 0; k < pogs_data->max_iter; ++k) {
    // Evaluate Proximal Operators
    cml::blas_axpy(handle, -kOne, &zt, &z);
    ProxEval(g, rho, x.data, x12.data);
    ProxEval(f, rho, y.data, y12.data);

    // Compute Gap.
    cml::blas_axpy(handle, -kOne, &z12, &z);
    T gap;
    cml::blas_dot(handle, &z, &z12, &gap);

    // Project and Update Dual Variables
    cml::blas_axpy(handle, kOne, &z12, &zt);
    if (m >= n) {
      cml::vector_memcpy(&x, &xt);
      cml::blas_gemv(handle, CUBLAS_OP_T, kOne, &A, &yt, kOne, &x);
      cml::linalg_cholesky_svx(handle, &L, &x);
      cml::blas_gemv(handle, CUBLAS_OP_N, kOne, &A, &x, kZero, &y);
      cml::blas_axpy(handle, -kOne, &y, &yt);
    } else {
      cml::blas_gemv(handle, CUBLAS_OP_N, kOne, &A, &xt, kZero, &y);
      cml::blas_symv(handle, CUBLAS_FILL_MODE_LOWER, kOne, &AA, &yt, kOne, &y);
      cml::linalg_cholesky_svx(handle, &L, &y);
      cml::blas_axpy(handle, -kOne, &y, &yt);
      cml::vector_memcpy(&x, &xt);
      cml::blas_gemv(handle, CUBLAS_OP_T, kOne, &A, &yt, kOne, &x);
    }
    cml::blas_axpy(handle, -kOne, &x, &xt);

    // Compute primal and dual tolerances.
    T nrm_z = cml::blas_nrm2(handle, &z);
    T nrm_zt = cml::blas_nrm2(handle, &zt);
    T nrm_z12 = cml::blas_nrm2(handle, &z12);
    T eps_pri = sqrtn_atol + pogs_data->rel_tol * std::max(nrm_z12, nrm_z);
    T eps_dual = sqrtn_atol + pogs_data->rel_tol * rho * nrm_zt;

    // Compute ||r^k||_2 and ||s^k||_2 (use z_prev for temp storage).
    cml::blas_axpy(handle, -kOne, &z, &z_prev);
    T nrm_s = rho * cml::blas_nrm2(handle, &z_prev);
    cml::vector_memcpy(&z_prev, &z12);
    cml::blas_axpy(handle, -kOne, &z, &z_prev);
    T nrm_r = cml::blas_nrm2(handle, &z_prev);

    // Evaluate stopping criteria.
    bool converged = nrm_r <= eps_pri && nrm_s <= eps_dual;
    if (!pogs_data->quiet && (k % 10 == 0 || converged)) {
      T optval = FuncEval(f, y.data) + FuncEval(g, x.data);
      printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e %.3e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dual, optval, std::abs(gap));
    }

    if (converged)
      break;

    // Rescale rho.
    if (nrm_r > static_cast<T>(10.0) * nrm_s / rho) {
      rho = rho * static_cast<T>(5);
      cml::vector_scale(&zt, static_cast<T>(0.2));
    } else if (nrm_r < static_cast<T>(0.1) * nrm_s / rho) {
      rho = rho * static_cast<T>(0.2);
      cml::vector_scale(&zt, static_cast<T>(5));
    }

    // Make copy of z.
    cml::vector_memcpy(&z_prev, &z);
  }

  // Compute optimal value.
  pogs_data->optval = FuncEval(f, y.data) + FuncEval(g, x.data);

  // Scale x and y for output.
  cml::vector_div(&y, &d);
  cml::vector_mul(&x, &e);

  // Copy results to output.
  if (pogs_data->y != 0)
    cml::vector_memcpy(pogs_data->y, &y);
  if (pogs_data->x != 0)
    cml::vector_memcpy(pogs_data->x, &x);

  // Free up memory.
  cml::matrix_free(&L);
  cml::matrix_free(&AA);
  cml::matrix_free(&A);
  cml::vector_free(&d);
  cml::vector_free(&e);
  cml::vector_free(&z);
  cml::vector_free(&zt);
  cml::vector_free(&z12);
  cml::vector_free(&z_prev);
}

template void Pogs<double>(PogsData<double, double*> *);
template void Pogs<float>(PogsData<float, float*> *);

