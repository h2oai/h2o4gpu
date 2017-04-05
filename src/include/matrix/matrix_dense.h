#ifndef MATRIX_MATRIX_DENSE_H_
#define MATRIX_MATRIX_DENSE_H_

#include "matrix.h"

namespace pogs {

template <typename T>
class MatrixDense : public Matrix<T> {
 public:
  enum Ord {ROW, COL};

 private:
  // TODO: This should be shared cpu/gpu pointer?
  T *_data;

  int _wDev;
  Ord _ord;

  // Get rid of assignment operator.
  MatrixDense<T>& operator=(const MatrixDense<T>& A);

 public:
  // Constructor (only sets variables)
  MatrixDense(int wDev, char ord, size_t m, size_t n, const T *data);
  MatrixDense(int wDev, const MatrixDense<T>& A);
  ~MatrixDense();

  // Initialize matrix, call this before any other methods.
  int Init();

  // Method to equilibrate.
  int Equil(T *d, T *e, bool equillocal);

  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;

  // Getters
  const T* Data() const { return _data; }
  Ord Order() const { return _ord; }
  int WDEV() const { return _wDev; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DENSE_H_

