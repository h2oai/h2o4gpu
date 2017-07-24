#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#include <algorithm>
#include <cstring>
#include <vector>
#include <iterator>
#include <iostream>

#include "matrix/matrix_dense.h"
#include "h2ogpumlglm.h"
#include "../../../include/prox_lib.h"

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
                         std::vector<FunctionObj<double> > *f_h2ogpuml) {
  const unsigned int kNumParam = 6u;
  char alpha[] = "h\0a\0b\0c\0d\0e\0";

  SEXP param_data[kNumParam] = {R_NilValue};
  for (unsigned int i = 0; i < kNumParam; ++i)
    param_data[i] = getListElement(f, &alpha[i * 2]);

  Function func_param = kZero;
  double real_params[] = {1.0, 0.0, 1.0, 0.0, 0.0};

  // Find index and pointer to data of (h, a, b, c, d, e) in struct if present.
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

  // Populate f_h2ogpuml.
  f_h2ogpuml->resize(n);
  //#pragma omp parallel for
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < kNumParam; ++j) {
      if (param_data[j] != R_NilValue) {
        if (j == 0) {
          func_param = static_cast<Function>(REAL(param_data[j])[i]);
        } else {
          real_params[j - 1] = REAL(param_data[j])[i];
        }
      }
    }
    (*f_h2ogpuml)[i]=FunctionObj<double>(func_param, real_params[0],
         real_params[1], real_params[2], real_params[3], real_params[4]);
  }
}

template <typename T, typename M, typename P>
void PopulateParams(SEXP params, h2ogpuml::H2OGPUML<T, M, P> *h2ogpuml_data) {
  // Check if parameter exists in params, and set the corresponding
  // value in h2ogpuml_data.

  SEXP rel_tol = getListElement(params, "rel_tol");
  if (rel_tol != R_NilValue)
    h2ogpuml_data->SetRelTol(REAL(rel_tol)[0]);

  SEXP abs_tol = getListElement(params, "abs_tol");
  if (abs_tol != R_NilValue)
    h2ogpuml_data->SetAbsTol(REAL(abs_tol)[0]);

  SEXP rho = getListElement(params, "rho");
  if (rho != R_NilValue)
    h2ogpuml_data->SetRho(REAL(rho)[0]);

  SEXP max_iter = getListElement(params, "max_iter");
  if (max_iter != R_NilValue)
    h2ogpuml_data->SetMaxIter(REAL(max_iter)[0]);

  SEXP verbose = getListElement(params, "verbose");
  if (verbose != R_NilValue)
    h2ogpuml_data->SetVerbose(INTEGER(verbose)[0]);

  SEXP adaptive_rho = getListElement(params, "adaptive_rho");
  if (adaptive_rho != R_NilValue)
    h2ogpuml_data->SetAdaptiveRho(LOGICAL(adaptive_rho)[0]);

  SEXP equil = getListElement(params, "equil");
  if (equil != R_NilValue)
    h2ogpuml_data->SetEquil(LOGICAL(equil)[0]);

  SEXP gap_stop = getListElement(params, "gap_stop");
  if (gap_stop != R_NilValue)
    h2ogpuml_data->SetGapStop(LOGICAL(gap_stop)[0]);

  SEXP nDev = getListElement(params, "nDev");
  if (nDev != R_NilValue)
    h2ogpuml_data->SetnDev(INTEGER(nDev)[0]);

  SEXP wDev = getListElement(params, "wDev");
  if (wDev != R_NilValue)
    h2ogpuml_data->SetwDev(INTEGER(wDev)[0]);
}

template <typename T>
void SolverWrap(SEXP A, SEXP fin, SEXP gin, SEXP params, SEXP x, SEXP y,
                SEXP u, SEXP v, SEXP opt, SEXP status) {
  SEXP Adim = GET_DIM(A);
  size_t m = INTEGER(Adim)[0];
  size_t n = INTEGER(Adim)[1];
  unsigned int num_obj = length(fin);

  int sharedA=0;
  int me=0;
  int wDev = 0;
  SEXP pw = getListElement(params, "wDev");
  if (pw != R_NilValue)
    wDev = INTEGER(pw)[0];

  //column major data (R data frame)
//  std::cout << "A cols = " << INTEGER(GET_DIM(A))[0] << std::endl;
//  std::cout << "A rows = " << INTEGER(GET_DIM(A))[1] << std::endl;
//  std::copy(REAL(A), REAL(A) + m*n, std::ostream_iterator<T>(std::cout, "\n"));
  h2ogpuml::MatrixDense<T> A_dense(sharedA, me, wDev, 'c', m, n, REAL(A));

  // Initialize H2OGPUML data structure
  h2ogpuml::H2OGPUMLDirect<T, h2ogpuml::MatrixDense<T> > h2ogpuml_data(A_dense);
  std::vector<FunctionObj<T> > f, g;

  f.reserve(m);
  g.reserve(n);

  // Populate parameters.
  PopulateParams(params, &h2ogpuml_data); //also sets wDev again

  // Allocate space for factors if more than one objective.
  int err = 0;

  for (unsigned int i = 0; i < num_obj && !err; ++i) {
    // Populate function objects.
    f.clear();
    g.clear();
    PopulateFunctionObj(VECTOR_ELT(fin, i), m, &f);
    PopulateFunctionObj(VECTOR_ELT(gin, i), n, &g);

    // Run solver.
    INTEGER(status)[i] = h2ogpuml_data.Solve(f, g);

    // Get Solution
    memcpy(REAL(x) + i * n, h2ogpuml_data.GetX(), n * sizeof(T));
    memcpy(REAL(y) + i * m, h2ogpuml_data.GetY(), m * sizeof(T));
    memcpy(REAL(u) + i * n, h2ogpuml_data.GetMu(), n * sizeof(T));
    memcpy(REAL(v) + i * m, h2ogpuml_data.GetLambda(), m * sizeof(T));

    REAL(opt)[i] = h2ogpuml_data.GetOptval();

    // TODO FIXME: add early stopping.  Add RMSE and other errors and control tolerance and stopping.
    
  }
}

extern "C" {
SEXP H2OGPUMLWrapper(SEXP A, SEXP f, SEXP g, SEXP params) {
  // Setup output.
  SEXP x, y, u, v, opt, status, ans, retnames;
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
  PROTECT(v = allocMatrix(REALSXP, m, num_obj));
  SET_STRING_ELT(retnames, 2, mkChar("v"));
  SET_VECTOR_ELT(ans, 2, v);

  // Allocate mu.
  PROTECT(u = allocMatrix(REALSXP, n, num_obj));
  SET_STRING_ELT(retnames, 3, mkChar("u"));
  SET_VECTOR_ELT(ans, 3, u);

  // Allocate opt.
  PROTECT(opt = NEW_NUMERIC(num_obj));
  SET_STRING_ELT(retnames, 4, mkChar("optval"));
  SET_VECTOR_ELT(ans, 4, opt);

  // Allocate status.
  PROTECT(status = NEW_INTEGER(num_obj));
  SET_STRING_ELT(retnames, 5, mkChar("status"));
  SET_VECTOR_ELT(ans, 5, status);

  SolverWrap<double>(A, f, g, params, x, y, u, v, opt, status);

  UNPROTECT(8);
  return ans;
}
}

