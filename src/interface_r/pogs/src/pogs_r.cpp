#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "matrix_util.h"
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

template <typename M>
void PopulateParams(SEXP params, PogsData<double, M> *pogs_data) {
  // Check if parameter exists in params, and set the corresponding
  // value in pogs_data.
  
  SEXP rel_tol = getListElement(params, "rel_tol");
  if (rel_tol != R_NilValue)
    pogs_data->rel_tol = REAL(rel_tol)[0];

  SEXP abs_tol = getListElement(params, "abs_tol");
  if (abs_tol != R_NilValue)
    pogs_data->abs_tol = REAL(abs_tol)[0];

  SEXP rho = getListElement(params, "rho");
  if (rho != R_NilValue)
    pogs_data->rho = REAL(rho)[0];

  SEXP max_iter = getListElement(params, "max_iter");
  if (max_iter != R_NilValue)
    pogs_data->max_iter = REAL(max_iter)[0];

  SEXP quiet = getListElement(params, "quiet");
  if (quiet != R_NilValue)
    pogs_data->quiet = LOGICAL(quiet)[0];

  SEXP adaptive_rho = getListElement(params, "adaptive_rho");
  if (adaptive_rho != R_NilValue)
    pogs_data->adaptive_rho = LOGICAL(adaptive_rho)[0];
}

void SolverWrap(SEXP A, SEXP f, SEXP g, SEXP params, SEXP x, SEXP y, SEXP l,
                SEXP opt) {
  SEXP Adim = GET_DIM(A);
  size_t m = INTEGER(Adim)[0];
  size_t n = INTEGER(Adim)[1];
  unsigned int num_obj = length(f);

  Dense<double, CblasColMajor> A_dense(REAL(A));

  // Initialize Pogs data structure
  PogsData<double, Dense<double, CblasColMajor> > pogs_data(A_dense, m, n);
  pogs_data.f.reserve(m);
  pogs_data.g.reserve(n);

  // Populate parameters.
  PopulateParams(params, &pogs_data);

  // Allocate space for factors if more than one objective.
  int err = 0;
  if (num_obj > 1)
    err = AllocDenseFactors(&pogs_data);

  for (unsigned int i = 0; i < num_obj && !err; ++i) {
    pogs_data.x = REAL(x) + i * n;
    pogs_data.y = REAL(y) + i * m;
    pogs_data.l = REAL(l) + i * m;

    // Populate function objects.
    pogs_data.f.clear();
    pogs_data.g.clear();
    PopulateFunctionObj(VECTOR_ELT(f, i), m, &pogs_data.f);
    PopulateFunctionObj(VECTOR_ELT(g, i), n, &pogs_data.g);

    // Run solver.
    Pogs(&pogs_data);

    REAL(opt)[i] = pogs_data.optval;
  }

  if (num_obj > 1)
    FreeDenseFactors(&pogs_data);
}

extern "C" {
SEXP PogsWrapper(SEXP A, SEXP f, SEXP g, SEXP params) {
  // Setup output.
  SEXP x, y, l, opt, ans, retnames;
  SEXP Adim = GET_DIM(A);
  size_t m = INTEGER(Adim)[0];
  size_t n = INTEGER(Adim)[1];
  unsigned int num_obj = length(f);

  // Create output list.
  PROTECT(ans = NEW_LIST(4));
  PROTECT(retnames = NEW_CHARACTER(4));
  SET_NAMES(ans, retnames);

  // Allocate x.
  PROTECT(x = allocMatrix(REALSXP, n, num_obj));
  SET_STRING_ELT(retnames, 0, mkChar("x"));
  SET_VECTOR_ELT(ans, 0, x);

  // Allocate y.
  PROTECT(y = allocMatrix(REALSXP, m, num_obj));
  SET_STRING_ELT(retnames, 1, mkChar("y"));
  SET_VECTOR_ELT(ans, 1, y);

  // Allocate l.
  PROTECT(l = allocMatrix(REALSXP, m, num_obj));
  SET_STRING_ELT(retnames, 2, mkChar("l"));
  SET_VECTOR_ELT(ans, 2, l);

  // Allocate opt.
  PROTECT(opt = NEW_NUMERIC(num_obj));
  SET_STRING_ELT(retnames, 3, mkChar("optval"));
  SET_VECTOR_ELT(ans, 3, opt);

  SolverWrap(A, f, g, params, x, y, l, opt);

  UNPROTECT(6);
  return ans;
}
}

