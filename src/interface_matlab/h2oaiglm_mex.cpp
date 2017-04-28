#include <matrix.h>
#include <mex.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "h2oaiglm.h"

using h2oaiglm::H2OAIGLM_INT;

// Returns value of pr[idx], with appropriate casting from id to T.
// If id is not a numeric type, then it returns nan.
template <typename T>
inline T GetVal(const void *pr, size_t idx, mxClassID id) {
  switch(id) {
    case mxDOUBLE_CLASS:
      return static_cast<T>(reinterpret_cast<const double*>(pr)[idx]);
    case mxSINGLE_CLASS:
      return static_cast<T>(reinterpret_cast<const float*>(pr)[idx]);
    case mxINT8_CLASS:
      return static_cast<T>(reinterpret_cast<const char*>(pr)[idx]);
    case mxUINT8_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned char*>(pr)[idx]);
    case mxINT16_CLASS:
      return static_cast<T>(reinterpret_cast<const short*>(pr)[idx]);
    case mxUINT16_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned short*>(pr)[idx]);
    case mxINT32_CLASS:
      return static_cast<T>(reinterpret_cast<const int*>(pr)[idx]);
    case mxUINT32_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned int*>(pr)[idx]);
    case mxINT64_CLASS:
      return static_cast<T>(reinterpret_cast<const long*>(pr)[idx]);
    case mxUINT64_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned long*>(pr)[idx]);
    case mxLOGICAL_CLASS:
      return static_cast<T>(reinterpret_cast<const bool*>(pr)[idx]);
    case mxCELL_CLASS:
    case mxCHAR_CLASS:
    case mxFUNCTION_CLASS:
    case mxSTRUCT_CLASS:
    case mxUNKNOWN_CLASS:
    case mxVOID_CLASS:
    default:
      return std::numeric_limits<T>::quiet_NaN();
  }
}

// Populates a vector of function objects from a matlab struct
// containing the fields (f, a, b, c, d). The latter 4 are optional,
// while f is required. Each field (if present) is a vector of length n.
template <typename T>
int PopulateFunctionObj(const char fn_name[], const mxArray *f_mex,
                        unsigned int field_idx, unsigned int n,
                        std::vector<FunctionObj<T> > *f_h2oaiglm) {
  const unsigned int kNumParam = 6u;
  char alpha[] = "h\0a\0b\0c\0d\0e\0";

  int param_idx[kNumParam];
#pragma unroll
  for (unsigned int i = 0; i < kNumParam; ++i)
    param_idx[i] = mxGetFieldNumber(f_mex, &alpha[2 * i]);

  if (param_idx[0] == -1) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:missingParam",
        "Field %s.h is required.", fn_name);
    return 1;
  }

  void *param_data[kNumParam] = {0};
  mxClassID param_id[kNumParam];
  mxArray *param_arr[kNumParam];
  Function func_param;
  T real_params[] = { static_cast<T>(1), static_cast<T>(0), static_cast<T>(1),
                      static_cast<T>(0), static_cast<T>(0) };

  // Find index and pointer to data of (h, a, b, c, d) in struct if present.
#pragma unroll
  for (unsigned int i = 0; i < kNumParam; ++i) {
    if (param_idx[i] != -1) {
      mxArray *arr = mxGetFieldByNumber(f_mex, field_idx, param_idx[i]);
      param_data[i] = static_cast<T*>(mxGetData(arr));
      param_id[i] = mxGetClassID(arr);

      // If parameter is scalar, then repeat it.
      if (mxGetM(arr) == 1 && mxGetN(arr) == 1) {
        if (i == 0) {
          func_param = GetVal<Function>(param_data[i], 0, param_id[i]);
        } else {
          real_params[i - 1] = GetVal<T>(param_data[i], 0, param_id[i]);
        }
        param_data[i] = 0;
      } else if (i > 0 && mxIsEmpty(arr)) {
        param_data[i] = 0;
      } else if (i == 0 && mxIsEmpty(arr)) {
        mexErrMsgIdAndTxt("MATLAB:h2oaiglm:missingParam",
            "Field %s.h is required.", fn_name);
        return 1;
      } else if (!(mxGetM(arr) == n && mxGetN(arr) == 1) &&
                 !(mxGetN(arr) == n && mxGetM(arr) == 1)) {
        mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
            "Dimensions of %s.%s and A must match.", fn_name, &alpha[2 * i]);
        return 1;
      }
    }
  }

  // Populate f_h2oaiglm.
#pragma omp parellel for
  for (unsigned int i = 0; i < n; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < kNumParam; ++j) {
      if (param_data[j] != 0) {
        if (j == 0) {
          func_param = GetVal<Function>(param_data[j], i, param_id[j]);
        } else {
          real_params[j - 1] = GetVal<T>(param_data[j], i, param_id[j]);
        }
      }
    }

    f_h2oaiglm->push_back(FunctionObj<T>(func_param, real_params[0], real_params[1],
        real_params[2], real_params[3], real_params[4]));
  }
  return 0;
}

// Populate parameters (rel_tol, abs_tol, max_iter, rho and quiet) in PogsData.
template <typename T, typename M, typename P>
int PopulateParams(const mxArray *params, h2oaiglm::Pogs<T, M, P> *h2oaiglm_data) {
  // Check if parameter exists in params, then make sure that it has
  // dimension 1x1 and finally set the corresponding value in h2oaiglm_data.
  int rel_tol_idx = mxGetFieldNumber(params, "rel_tol");
  if (rel_tol_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, rel_tol_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter rel_tol must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetRelTol(GetVal<T>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int abs_tol_idx = mxGetFieldNumber(params, "abs_tol");
  if (abs_tol_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, abs_tol_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter abs_tol must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetAbsTol(GetVal<T>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int rho_idx = mxGetFieldNumber(params, "rho");
  if (rho_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, rho_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter rho must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetRho(GetVal<T>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int max_iter_idx = mxGetFieldNumber(params, "max_iter");
  if (max_iter_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, max_iter_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter max_iter must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetMaxIter(
        GetVal<unsigned int>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int quiet_idx = mxGetFieldNumber(params, "verbose");
  if (quiet_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, quiet_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter quiet must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetVerbose(GetVal<int>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int adaptive_rho_idx = mxGetFieldNumber(params, "adaptive_rho");
  if (adaptive_rho_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, adaptive_rho_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter adaptive_rho must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetAdaptiveRho(
        GetVal<bool>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int equil_idx = mxGetFieldNumber(params, "equil");
  if (equil_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, equil_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter equil must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetEquil(
        GetVal<bool>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int gap_stop_idx = mxGetFieldNumber(params, "gap_stop");
  if (gap_stop_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, gap_stop_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter gap_stop must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetGapStop(GetVal<bool>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int nDev_idx = mxGetFieldNumber(params, "nDev");
  if (nDev_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, nDev_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter nDev must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetnDev(
        GetVal<int>(mxGetData(arr), 0, mxGetClassID(arr)));
  }
  int wDev_idx = mxGetFieldNumber(params, "wDev");
  if (wDev_idx != -1) {
    mxArray *arr = mxGetFieldByNumber(params, 0, wDev_idx);
    if (mxGetM(arr) != 1 || mxGetN(arr) != 1) {
      mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
          "Parameter wDev must have dimension (1,1)");
      return 1;
    }
    h2oaiglm_data->SetwDev(
        GetVal<int>(mxGetData(arr), 0, mxGetClassID(arr)));
  }

  return 0;
}

template <typename T1, typename T2>
void IntToInt(size_t n, const T1 *in, T2 *out) {
#ifdef _OPENMP
#pragma omp paralell for
#endif
  for (size_t i = 0; i < n; ++i)
    out[i] = static_cast<T2>(in[i]);
}

// Wrapper for graph h2oaiglm. Populates h2oaiglm_data structure and calls h2oaiglm.
template <typename T>
void SolverWrapDn(int wDev, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  size_t m = mxGetM(prhs[0]);
  size_t n = mxGetN(prhs[0]);

  // Initialize Pogs data structure
  h2oaiglm::MatrixDense<T> A_(wDev,'c', m, n, reinterpret_cast<T*>(mxGetData(prhs[0])));
  h2oaiglm::PogsDirect<T, h2oaiglm::MatrixDense<T> > h2oaiglm_data(wDev,A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  g.reserve(n);

  int err = 0;

  unsigned int num_obj = std::max(mxGetN(prhs[1]), mxGetM(prhs[1]));

  // Populate parameters.
  if (!err && nrhs == 4)
    err = PopulateParams(prhs[3], &h2oaiglm_data);

  for (unsigned int i = 0; i < num_obj && !err; ++i) {

    // Populate function objects.
    f.clear();
    g.clear();
    err = PopulateFunctionObj("f", prhs[1], i, m, &f);
    if (err)
      break;
    err = PopulateFunctionObj("g", prhs[2], i, n, &g);
    if (err)
      break;
    
    // Run solver.
    h2oaiglm::PogsStatus status = h2oaiglm_data.Solve(f, g);

    // Get solution.
    memcpy(reinterpret_cast<T*>(mxGetData(plhs[0])) + i * n, h2oaiglm_data.GetX(),
        n * sizeof(T));
    if (nlhs >= 2) {
      memcpy(reinterpret_cast<T*>(mxGetData(plhs[1])) + i * m,
          h2oaiglm_data.GetY(), m * sizeof(T));
    }
    if (nlhs >= 3) {
      memcpy(reinterpret_cast<T*>(mxGetData(plhs[2])) + i * m,
          h2oaiglm_data.GetLambda(), m * sizeof(T));
    }
    if (nlhs >= 4) {
      memcpy(reinterpret_cast<T*>(mxGetData(plhs[3])) + i * n,
          h2oaiglm_data.GetMu(), n * sizeof(T));
    }

    if (nlhs >= 5)
      reinterpret_cast<T*>(mxGetData(plhs[4]))[i] = h2oaiglm_data.GetOptval();
    if (nlhs >= 6)
      reinterpret_cast<T*>(mxGetData(plhs[5]))[i] = status;
  }
}

template <typename T>
void SolverWrapSp(int wDev, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  mwIndex *mw_row_ind = mxGetIr(prhs[0]);
  mwIndex *mw_col_ptr = mxGetJc(prhs[0]);
  T *val = reinterpret_cast<T*>(mxGetData(prhs[0]));
  size_t m = mxGetM(prhs[0]);
  size_t n = mxGetN(prhs[0]);
  size_t nnz = mw_col_ptr[n];

  H2OAIGLM_INT *row_ind = new H2OAIGLM_INT[nnz];
  H2OAIGLM_INT *col_ptr = new H2OAIGLM_INT[n + 1];
  IntToInt(nnz, mw_row_ind, row_ind);
  IntToInt(n + 1, mw_col_ptr, col_ptr);

  // Initialize Pogs data structure
  h2oaiglm::MatrixSparse<T> A(wDev, 'c', m, n, nnz, val, col_ptr, row_ind);
  h2oaiglm::PogsIndirect<T, h2oaiglm::MatrixSparse<T> > h2oaiglm_data(wDev, A);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  g.reserve(n);

  int err = 0;

  unsigned int num_obj = std::max(mxGetN(prhs[1]), mxGetM(prhs[1]));

  // Populate parameters.
  if (!err && nrhs == 4)
    err = PopulateParams(prhs[3], &h2oaiglm_data);

  for (unsigned int i = 0; i < num_obj && !err; ++i) {
    // Populate function objects.
    f.clear();
    g.clear();
    err = PopulateFunctionObj("f", prhs[1], i, m, &f);
    if (err)
      break;
    err = PopulateFunctionObj("g", prhs[2], i, n, &g);
    if (err)
      break;
    
    // Run solver.
    h2oaiglm::PogsStatus status = h2oaiglm_data.Solve(f, g);

    // Get solution.
    memcpy(reinterpret_cast<T*>(mxGetData(plhs[0])) + i * n, h2oaiglm_data.GetX(),
        n * sizeof(T));
    if (nlhs >= 2) {
      memcpy(reinterpret_cast<T*>(mxGetData(plhs[1])) + i * m,
          h2oaiglm_data.GetY(), m * sizeof(T));
    }
    if (nlhs >= 3) {
      memcpy(reinterpret_cast<T*>(mxGetData(plhs[2])) + i * m,
          h2oaiglm_data.GetLambda(), m * sizeof(T));
    }
    if (nlhs >= 4) {
      memcpy(reinterpret_cast<T*>(mxGetData(plhs[3])) + i * n,
          h2oaiglm_data.GetMu(), n * sizeof(T));
    }

    if (nlhs >= 5)
      reinterpret_cast<T*>(mxGetData(plhs[4]))[i] = h2oaiglm_data.GetOptval();
    if (nlhs >= 6)
      reinterpret_cast<T*>(mxGetData(plhs[5]))[i] = status;
  }

  delete [] row_ind;
  delete [] col_ptr;
}

void mexFunction(int wDev, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Check number of arguments.
  if (nrhs < 3 || nrhs > 4) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:insufficientInputArgs",
        "Usage: [x, y, l, u, optval, status] = h2oaiglm(A, f, g, [params])");
    return;
  }
  if (nlhs > 6) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:extraneousOutputArgs",
        "Usage: [x, y, l, u, optval, status] = h2oaiglm(A, f, g, [params])");
    return;
  }

  // Check that the argument class is correct.
  mxClassID class_id_A = mxGetClassID(prhs[0]);
  if (class_id_A != mxSINGLE_CLASS && class_id_A != mxDOUBLE_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:inputNotNumeric",
        "Matrix A must either be single or double precision.");
    return;
  }
  if (mxGetClassID(prhs[1]) != mxSTRUCT_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:inputNotStruct",
        "Function f must be a struct.");
    return;
  }
  if (mxGetClassID(prhs[2]) != mxSTRUCT_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:inputNotStruct",
        "Function g must be a struct.");
    return;
  }
  if (nrhs == 4 && mxIsEmpty(prhs[3])) {
    nrhs = 3;
  } else if (nrhs == 4 && mxGetClassID(prhs[3]) != mxSTRUCT_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:inputNotStruct",
        "Parameters must be a struct.");
    return;
  }
  if (std::max(mxGetN(prhs[1]), mxGetM(prhs[1])) !=
      std::max(mxGetN(prhs[2]), mxGetM(prhs[2])) || 
      std::min(mxGetN(prhs[1]), mxGetM(prhs[1])) != 1 ||
      std::min(mxGetN(prhs[2]), mxGetM(prhs[2])) != 1) {
    mexErrMsgIdAndTxt("MATLAB:h2oaiglm:dimensionMismatch",
        "Dimension of f and g must match");
    return;
  }
  unsigned int num_obj = std::max(mxGetN(prhs[1]), mxGetM(prhs[1]));
  unsigned int m = mxGetM(prhs[0]);
  unsigned int n = mxGetN(prhs[0]);

  // Allocate memory for output.
  plhs[0] = mxCreateNumericMatrix(n, num_obj, class_id_A, mxREAL);
  if (nlhs >= 2)
    plhs[1] = mxCreateNumericMatrix(m, num_obj, class_id_A, mxREAL);
  if (nlhs >= 3)
    plhs[2] = mxCreateNumericMatrix(m, num_obj, class_id_A, mxREAL);
  if (nlhs >= 4)
    plhs[3] = mxCreateNumericMatrix(n, num_obj, class_id_A, mxREAL);
  if (nlhs >= 5)
    plhs[4] = mxCreateNumericMatrix(num_obj, 1, class_id_A, mxREAL);
  if (nlhs >= 6)
    plhs[5] = mxCreateNumericMatrix(num_obj, 1, class_id_A, mxREAL);

  if (mxIsSparse(prhs[0])) {
    if (class_id_A == mxDOUBLE_CLASS)
      SolverWrapSp<double>(wDev, nlhs, plhs, nrhs, prhs);
    else if (class_id_A == mxSINGLE_CLASS)
      SolverWrapSp<float>(wDev, nlhs, plhs, nrhs, prhs);
  } else {
    if (class_id_A == mxDOUBLE_CLASS)
      SolverWrapDn<double>(wDev, nlhs, plhs, nrhs, prhs);
    else if (class_id_A == mxSINGLE_CLASS)
      SolverWrapDn<float>(wDev, nlhs, plhs, nrhs, prhs);
  }
}

