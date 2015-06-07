#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "matrix/matrix_dense.h"
#include "pogs.h"

SEXP getListElement(SEXP list, const char *str) {
  SEXP elmt = R_NilValue, names = getAttrib(list, R_NamesSymbol);
  for (int i = 0; i < length(list); i++) {
    if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
      elmt = VECTOR_ELT(list, i);
      break;
    }
  }
  return elmt;
}

void PopulateFunctionObj(SEXP f, unsigned int n,
                         std::vector<FunctionObj<double> > *f_pogs) {
  const unsigned int kNumParam = 6u;
  char alpha[] = "h\0a\0b\0c\0d\0e\0";

  SEXP param_data[kNumParam] = {R_NilValue};
  #pragma unroll
  for (unsigned int i = 0; i < kNumParam; ++i)
    param_data[i] = getListElement(f, &alpha[i * 2]);

  Function func_param;
  double real_params[] = {1.0, 0.0, 1.0, 0.0, 0.0};

  // Find index and pointer to data of (h, a, b, c, d, e) in struct if present.
  #pragma unroll
  for (unsigned int i = 0; i < kNumParam; ++i) {
    if (param_data[i] != R_NilValue) {
      // If parameter is scalar, then repeat it.
      if (length(param_data[i]) == 1) {
        if (i == 0) {
          func_param = static_cast<Function>(REAL(param_data[i])[0]);
        } else {
          real_params[i - 1] = REAL(param_data[i])[0];
        }
        param_data[i] = R_NilValue;
      }
    }
  }

  // Populate f_pogs.
  #pragma omp parellel for
  for (unsigned int i = 0; i < n; ++i) {
    #pragma unroll
    for (unsigned int j = 0; j < kNumParam; ++j) {
      if (param_data[j] != R_NilValue) {
        if (j == 0) {
          func_param = static_cast<Function>(REAL(param_data[j])[i]);
        } else {
          real_params[j - 1] = REAL(param_data[j])[i];
        }
      }
    }

    f_pogs->push_back(FunctionObj<double>(func_param, real_params[0],
         real_params[1], real_params[2], real_params[3], real_params[4]));
  }
}

template <typename T, typename M, typename P>
void PopulateParams(SEXP params, pogs::Pogs<T, M, P> *pogs_data) {
  // Check if parameter exists in params, and set the corresponding
  // value in pogs_data.

  SEXP rel_tol = getListElement(params, "rel_tol");
  if (rel_tol != R_NilValue)
    pogs_data->SetRelTol(REAL(rel_tol)[0]);

  SEXP abs_tol = getListElement(params, "abs_tol");
  if (abs_tol != R_NilValue)
    pogs_data->SetAbsTol(REAL(abs_tol)[0]);

  SEXP rho = getListElement(params, "rho");
  if (rho != R_NilValue)
    pogs_data->SetRho(REAL(rho)[0]);

  SEXP max_iter = getListElement(params, "max_iter");
  if (max_iter != R_NilValue)
    pogs_data->SetMaxIter(REAL(max_iter)[0]);

  SEXP verbose = getListElement(params, "verbose");
  if (verbose != R_NilValue)
    pogs_data->SetVerbose(INTEGER(verbose)[0]);

  SEXP adaptive_rho = getListElement(params, "adaptive_rho");
  if (adaptive_rho != R_NilValue)
    pogs_data->SetAdaptiveRho(LOGICAL(adaptive_rho)[0]);

  SEXP gap_stop = getListElement(params, "gap_stop");
  if (gap_stop != R_NilValue)
    pogs_data->SetGapStop(LOGICAL(gap_stop)[0]);
}

template <typename T>
void SolverWrap(SEXP A, SEXP fin, SEXP gin, SEXP params, SEXP x, SEXP y,
                SEXP mu, SEXP nu, SEXP opt, SEXP status) {
  SEXP Adim = GET_DIM(A);
  size_t m = INTEGER(Adim)[0];
  size_t n = INTEGER(Adim)[1];
  unsigned int num_obj = length(fin);

  pogs::MatrixDense<T> A_dense('c', m, n, REAL(A));

  // Initialize Pogs data structure
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_dense);
  std::vector<FunctionObj<T> > f, g;

  f.reserve(m);
  g.reserve(n);

  // Populate parameters.
  PopulateParams(params, &pogs_data);

  // Allocate space for factors if more than one objective.
  int err = 0;

  for (unsigned int i = 0; i < num_obj && !err; ++i) {
    // Populate function objects.
    f.clear();
    g.clear();
    PopulateFunctionObj(VECTOR_ELT(fin, i), m, &f);
    PopulateFunctionObj(VECTOR_ELT(gin, i), n, &g);

    // Run solver.
    INTEGER(status)[i] = pogs_data.Solve(f, g);

    // Get Solution
    memcpy(REAL(x) + i * n, pogs_data.GetX(), n * sizeof(T));
    memcpy(REAL(y) + i * m, pogs_data.GetY(), m * sizeof(T));
    memcpy(REAL(mu) + i * n, pogs_data.GetMu(), n * sizeof(T));
    memcpy(REAL(nu) + i * m, pogs_data.GetNu(), m * sizeof(T));

    REAL(opt)[i] = pogs_data.GetOptval();
  }
}

extern "C" {
SEXP PogsWrapper(SEXP A, SEXP f, SEXP g, SEXP params) {
  // Setup output.
  SEXP x, y, mu, nu, opt, status, ans, retnames;
  SEXP Adim = GET_DIM(A);
  size_t m = INTEGER(Adim)[0];
  size_t n = INTEGER(Adim)[1];
  unsigned int num_obj = length(f);

  // Create output list.
  PROTECT(ans = NEW_LIST(6));
  PROTECT(retnames = NEW_CHARACTER(6));
  SET_NAMES(ans, retnames);

  // Allocate x.
  PROTECT(x = allocMatrix(REALSXP, n, num_obj));
  SET_STRING_ELT(retnames, 0, mkChar("x"));
  SET_VECTOR_ELT(ans, 0, x);

  // Allocate y.
  PROTECT(y = allocMatrix(REALSXP, m, num_obj));
  SET_STRING_ELT(retnames, 1, mkChar("y"));
  SET_VECTOR_ELT(ans, 1, y);

  // Allocate nu.
  PROTECT(nu = allocMatrix(REALSXP, m, num_obj));
  SET_STRING_ELT(retnames, 2, mkChar("nu"));
  SET_VECTOR_ELT(ans, 2, nu);

  // Allocate mu.
  PROTECT(mu = allocMatrix(REALSXP, n, num_obj));
  SET_STRING_ELT(retnames, 3, mkChar("mu"));
  SET_VECTOR_ELT(ans, 3, mu);

  // Allocate opt.
  PROTECT(opt = NEW_NUMERIC(num_obj));
  SET_STRING_ELT(retnames, 4, mkChar("optval"));
  SET_VECTOR_ELT(ans, 4, opt);

  // Allocate status.
  PROTECT(status = NEW_INTEGER(num_obj));
  SET_STRING_ELT(retnames, 5, mkChar("status"));
  SET_VECTOR_ELT(ans, 5, status);

  SolverWrap<double>(A, f, g, params, x, y, mu, nu, opt, status);

  UNPROTECT(8);
  return ans;
}
}

