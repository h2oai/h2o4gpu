/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#ifndef MATRIX_MATRIX_DENSE_H_
#define MATRIX_MATRIX_DENSE_H_

#include "matrix.h"

namespace h2o4gpu {

template <typename T>
class MatrixDense : public Matrix<T> {
 public:
  int _sharedA;
  int _me;
  int _wDev;
  int _datatype;
  int _dopredict;

 public:
  // TODO: This should be shared cpu/gpu pointer?
  T *_data; // trainX
  T *_datay; // trainY
  T *_vdata;  // validX
  T *_vdatay; // validY
  T *_weight; // weight
  T *_de;
  enum Ord {COL, ROW};


 private:
  // Get rid of assignment operator.
  MatrixDense<T>& operator=(const MatrixDense<T>& A);
  Ord _ord;

 public:
  // Constructor (only sets variables)
  MatrixDense(int sharedA, int wDev, char ord, size_t m, size_t n, const T *data); // Asource_ outside parallel for examples/cpp/elastic_net.cpp
  MatrixDense(char ord, size_t m, size_t n, const T *data); // orig pgs

  MatrixDense(int sharedA, int wDev, int datatype, char ord, size_t m, size_t n, T *data); // can be used by src/common/elastic_net_ptr.cpp
  
  MatrixDense(int sharedA, int me, int wDev, char ord, size_t m, size_t n, size_t mvalid, const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight); // initial ptr copy for examples/cpp/elastic_net_ptr_driver.cpp
  MatrixDense(int wDev, char ord, size_t m, size_t n, size_t mvalid, const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight); // not used now

  MatrixDense(int sharedA, int me, int wDev, int datatype, char ord, size_t m, size_t n, size_t mvalid, T *data, T *datay, T *vdata, T *vdatay, T *weight); // Asource_ inside parallel for src/common/elastic_net_ptr.cpp
  MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, size_t mvalid, T *data, T *datay, T *vdata, T *vdatay, T *weight); // not used now

  MatrixDense(int sharedA, int me, int wDev, const MatrixDense<T>& A); // used by examples/cpp/elasticnet*.cpp inside parallel region
  MatrixDense(int me, int wDev, const MatrixDense<T>& A); // not used
  MatrixDense(int wDev, const MatrixDense<T>& A); // not used
  MatrixDense(const MatrixDense<T>& A); // orig h2o4gpu

  ~MatrixDense();

  // Initialize matrix, call this before any other methods.
  int Init();

  // Method to equilibrate.
  int Equil(bool equillocal);

  // Method for SVD #1
  int svd1(void);
  
  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;
  int Mulvalid(char trans, T alpha, const T *x, T beta, T *y) const;

  int GetTrainX(int datatype, size_t size, T**data) const;
  int GetTrainY(int datatype, size_t size, T**data) const;
  int GetValidX(int datatype, size_t size, T**data) const;
  int GetValidY(int datatype, size_t size, T**data) const;
  int GetWeight(int datatype, size_t size, T**data) const;

  int Stats(int intercept, T *min, T *max, T *mean, T *var, T *sd, T *skew, T *kurt, T&lambda_max0);

  // Getters
  const T* Data() const { return _data; }
  const T* Datay() const { return _datay; }
  const T* vData() const { return _vdata; }
  const T* vDatay() const { return _vdatay; }
  const T* Weight() const { return _weight; }

  Ord Order() const { return _ord; }
  int GetsharedA() const { return _sharedA; }
  int wDev() const { return _wDev; }
  int Getme() const { return _me; }
  int Datatype() const { return _datatype; }
  int DoPredict() const { return _dopredict; }
};




}  // namespace h2o4gpu

#endif  // MATRIX_MATRIX_DENSE_H_

