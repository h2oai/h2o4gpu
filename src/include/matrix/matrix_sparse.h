#ifndef MATRIX_MATRIX_DENSE_H_
#define MATRIX_MATRIX_DENSE_H_

#include "matrix.h"

namespace pogs {

typedef int POGS_INT;

template <typename T>
class MatrixSparse : public Matrix<T> {
 public:
  enum Ord {ROW, COL};

 private:
  T *_data;
  
  POGS_INT *_ptr, *_ind, _nnz;

  Ord _ord;

 public:
  MatrixSparse(char ord, POGS_INT m, POGS_INT n, POGS_INT nnz, const T *data,
      const POGS_INT *ptr, const POGS_INT *ind);
  ~MatrixSparse();

  // Call this before any other method.
  int Init();

  // Free up data.
  int Free();

  // Method to equilibrate.
  int Equil(T *d, T *e);

  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y);

  // Getters
  T* Data() { return _data; }
  POGS_INT* Ptr() { return _ptr; }
  POGS_INT* Ind() { return _ind; }
  Ord Order() { return _ord; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DENSE_H_

