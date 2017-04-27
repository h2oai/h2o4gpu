#ifndef MATRIX_MATRIX_DENSE_H_
#define MATRIX_MATRIX_DENSE_H_

#include "matrix.h"

namespace pogs {

template <typename T>
class MatrixDense : public Matrix<T> {
 public:
  int _sharedA;
  int _me;
  int _wDev;
  int _datatype;

 public:
  // TODO: This should be shared cpu/gpu pointer?
  T *_data; // trainX
  T *_datay; // trainY
  T *_vdata;  // validX
  T *_vdatay; // validY
  T *_de;
  enum Ord {ROW, COL};


 private:
  // Get rid of assignment operator.
  MatrixDense<T>& operator=(const MatrixDense<T>& A);
  Ord _ord;

 public:
  // Constructor (only sets variables)
  MatrixDense(int sharedA, int wDev, char ord, size_t m, size_t n, const T *data); // Asource_ outside parallel for examples/cpp/elastic_net.cpp
  MatrixDense(char ord, size_t m, size_t n, const T *data); // orig pgs

  MatrixDense(int sharedA, int wDev, int datatype, char ord, size_t m, size_t n, T *data); // can be used by src/common/elastic_net_ptr.cpp
  
  MatrixDense(int sharedA, int me, int wDev, char ord, size_t m, size_t n, size_t mvalid, const T *data, const T *datay, const T *vdata, const T *vdatay); // initial ptr copy for examples/cpp/elastic_net_ptr_driver.cpp
  MatrixDense(int wDev, char ord, size_t m, size_t n, size_t mvalid, const T *data, const T *datay, const T *vdata, const T *vdatay); // not used now

  MatrixDense(int sharedA, int me, int wDev, int datatype, char ord, size_t m, size_t n, size_t mvalid, T *data, T *datay, T *vdata, T *vdatay); // Asource_ inside parallel for src/common/elastic_net_ptr.cpp
  MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, size_t mvalid, T *data, T *datay, T *vdata, T *vdatay); // not used now

  MatrixDense(int sharedA, int me, int wDev, const MatrixDense<T>& A); // used by examples/cpp/elasticnet*.cpp inside parallel region
  MatrixDense(int me, int wDev, const MatrixDense<T>& A); // not used
  MatrixDense(int wDev, const MatrixDense<T>& A); // not used
  MatrixDense(const MatrixDense<T>& A); // orig pogs

  ~MatrixDense();

  // Initialize matrix, call this before any other methods.
  int Init();

  // Method to equilibrate.
  int Equil(bool equillocal);

  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;
  int Mulvalid(char trans, T alpha, const T *x, T beta, T *y) const;

  void GetTrainX(int datatype, size_t size, T**data) const;
  void GetTrainY(int datatype, size_t size, T**data) const;
  void GetValidX(int datatype, size_t size, T**data) const;
  void GetValidY(int datatype, size_t size, T**data) const;

  int Stats(T *min, T *max, T *mean, T *var, T *sd, T *skew, T *kurt);

  // Getters
  const T* Data() const { return _data; }
  const T* Datay() const { return _datay; }
  const T* vData() const { return _vdata; }
  const T* vDatay() const { return _vdatay; }
  Ord Order() const { return _ord; }
  int GetsharedA() const { return _sharedA; }
  int wDev() const { return _wDev; }
  int Getme() const { return _me; }
  int Datatype() const { return _datatype; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DENSE_H_

