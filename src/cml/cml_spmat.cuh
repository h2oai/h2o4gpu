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

// Maybe store transpose sequentially after (makes factors easier to store).
template <typename T, typename I, MAT_ORD O>
struct spmat {
  T *val_n, *val_t;
  I *ind_n, *ptr_n, *ind_t, *ptr_t;
  size_t m, n, nnz;
  spmat(T *val_n, I *ind_n, I *ptr_n, T *val_t, I *ind_t, I *ptr_t, size_t m,
        size_t n, size_t nnz) 
    : val_n(val_n), ind_n(ind_n), ptr_n(ptr_n), val_t(val_t), ind_n(ind_t),
      ptr_t(ptr_t), m(m), n(n), nnz(nnz) { }
};

template <typename T, typname I, MAT_ORD O>
spmat<T, I, O> spmat_alloc(size_t m, size_t n, size_t nnz) {
  spmat<T, I, O> mat(0, 0, 0, 0, 0, 0, m, n, nnz);
  cudaMalloc(&mat.val_n, nnz * sizeof(T));
  cudaMalloc(&mat.ind_n, nnz * sizeof(I));
  cudaMalloc(&mat.ptr_n, ptr_len(spmat) * sizeof(I));
  cudaMalloc(&mat.val_t, nnz * sizeof(T));
  cudaMalloc(&mat.ind_t, nnz * sizeof(I));
  cudaMalloc(&mat.ptr_t, ptr_len(spmat) * sizeof(I));
  return mat;
}


template <typename T, typname I, MAT_ORD O>
spmat<T, I, O> spmat_memcpy(spmat<T, I, O> *A, const T *val, const I *ind,
                            const I *ptr) {
  cudaMemcpy(A->val, val, A->nnz * sizeof(T));
  cudaMemcpy(A->ind, ind, A->nnz * sizeof(T));
  cudaMemcpy(A->ptr, ptr, ptr_len(*A) sizeof(T));
}



}  // namespace

#endif  // CML_SPMAT_H_

