#ifndef CML_SPBLAS_CUH_
#define CML_SPBLAS_CUH_

#include "cgls.cuh"

namespace cml {

template <typename T, typename I, CGLS_FMT F>
solve(cusparseHandle_t handle_s, cublasHandle_t handle_b, spmat<T, I, O> A,
      const T shift, const T tol, const I maxit, bool quiet) {

  solve(handle_s, handle_b, cusparseMatDescr_t descr, const T *val_a, const INT *ptr_a,
          const INT *ind_a, const T *val_at, const INT *ptr_at,
          const INT *ind_at, const INT m, const INT n, const INT nnz,
          const T *b, T *x, const T shift, const T tol, const INT maxit,
          bool quiet) {

}



cusparseStatus_t cusparseDcsr2csc(cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal, const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscRowInd, int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)


 
}

#endif  // CML_SPBLAS_CUH_
