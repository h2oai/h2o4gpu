#ifndef CML_SPMAT_H_
#define CML_SPMAT_H_

namespace cml {

namespace {

template <typename T, typename I, MAT_ORD O>
I ptr_len(const spmat<T, I, O> &mat) {
  if (O == CSC)
    return mat.n + 1;
  else
    return mat.m + 1;
}

}

template <typename T, typename I, MAT_ORD O>
struct spmat {
  cusparseMatDescr_t descr;
  T *val, *val;
  I *ind, *ptr;
  size_t m, n, nnz;
  spmat(T *val, I *ind, I *ptr, size_t n, size_t nnz) 
      : val(val), ind(ind), ptr(ptr), m(m), n(n), nnz(nnz) { }
};

template <typename T, typename I, MAT_ORD O>
spmat<T, I, O> spmat_alloc(size_t m, size_t n, size_t nnz) {
  spmat<T, I, O> mat(0, 0, 0, m, n, nnz);
  cudaMalloc(&mat.val, 2 * nnz * sizeof(T));
  cudaMalloc(&mat.ind, 2 * nnz * sizeof(I));
  cudaMalloc(&mat.ptr, (m + n + 2) * sizeof(I));
  return mat;
}


template <typename T, typname I, typename I, MAT_ORD O>
spmat<T, I, O> spmat_memcpy(spmat<T, I, O> *A, const T *val, const INT *ind,
                            const INT *ptr) {
  cudaMemcpy(A->val, val, A->nnz * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(A->ind, ind, A->nnz * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(A->ptr, ptr, ptr_len(*A) * sizeof(T), cudaMemcpyHostTodevice);
  csr2csc(handle_s, m, n, nnz, val_a_d, rptr_a_d, cind_a_d, val_at_d,
     cind_at_d, rptr_at_d, CUSPARSE_ACTION_NUMERIC,
     CUSPARSE_INDEX_BASE_ZERO);

}



}  // namespace

#endif  // CML_SPMAT_H_

