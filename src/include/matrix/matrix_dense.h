#ifndef MATRIX_MATRIX_DENSE_H_
#define MATRIX_MATRIX_DENSE_H_

#include "matrix.h"

namespace pogs {

template <typename T>
class MatrixDense : public Matrix<T> {
 public:
  int _wDev;
  int _datatype;

 public:
  // TODO: This should be shared cpu/gpu pointer?
  T *_data; // trainX
  T *_datay; // trainY
  T *_vdata;  // validX
  T *_vdatay; // validY
  enum Ord {ROW, COL};


 private:
  // Get rid of assignment operator.
  MatrixDense<T>& operator=(const MatrixDense<T>& A);
  Ord _ord;

 public:
  // Constructor (only sets variables)
  MatrixDense(int wDev, char ord, size_t m, size_t n, const T *data);
  MatrixDense(char ord, size_t m, size_t n, const T *data);
  
  MatrixDense(int wDev, char ord, size_t m, size_t n, size_t mvalid, const T *data, const T *datay, const T *vdata, const T *vdatay);

  MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, T *data);
  
  MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, size_t mvalid, T *data, T *datay, T *vdata, T *vdatay);

  MatrixDense(int wDev, const MatrixDense<T>& A);
  MatrixDense(const MatrixDense<T>& A);

  ~MatrixDense();

  // Initialize matrix, call this before any other methods.
  int Init();

  // Method to equilibrate.
  int Equil(T *d, T *e, bool equillocal);

  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;
  int Mulvalid(char trans, T alpha, const T *x, T beta, T *y) const;

  void GetTrainX(int datatype, size_t size, T**data) const;
  void GetTrainY(int datatype, size_t size, T**data) const;
  void GetValidX(int datatype, size_t size, T**data) const;
  void GetValidY(int datatype, size_t size, T**data) const;

  // Getters
  const T* Data() const { return _data; }
  const T* Datay() const { return _datay; }
  const T* vData() const { return _vdata; }
  const T* vDatay() const { return _vdatay; }
  Ord Order() const { return _ord; }
  int wDev() const { return _wDev; }
  int Datatype() const { return _datatype; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DENSE_H_

