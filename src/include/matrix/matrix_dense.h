#ifndef MATRIX_MATRIX_DENSE_H_
#define MATRIX_MATRIX_DENSE_H_

#include "matrix.h"

namespace pogs {

template <typename T>
class MatrixDense : public Matrix<T> {
 public:
  enum Ord {ROW, COL};

 private:
  T *_data;

  Ord _ord;

 public:
  // Constructor (only sets variables)
  MatrixDense(char ord, size_t m, size_t n, const T *data);
  ~MatrixDense();

  // Initialize matrix, call this before any other methods.
  int Init();

  // Free up data, factors_direct and factors_indirect.
  int Free();

  // Method to equilibrate.
  int Equil(T *d, T *e);

  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;

  // Getters
  const T* Data() const { return _data; }
  Ord Order() const { return _ord; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DENSE_H_

