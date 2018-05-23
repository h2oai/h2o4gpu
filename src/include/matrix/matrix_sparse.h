/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef MATRIX_MATRIX_SPARSE_H_
#define MATRIX_MATRIX_SPARSE_H_

#include "matrix.h"

namespace h2o4gpu {

typedef int H2O4GPU_INT;

template <typename T>
class MatrixSparse : public Matrix<T> {
 public:
  int _sharedA;
  int _me;
  int _wDev;
  int _datatype;
  int _dopredict;
  
 public:
  T *_data;
  T *_datay;
  T *_vdata;
  T *_vdatay;
  T *_weight;
  T * _de;
  enum Ord {COL, ROW};

 private:
  
  H2O4GPU_INT *_ptr, *_ind, _nnz;

  Ord _ord;

  // Get rid of assignment operator.
  MatrixSparse<T>& operator=(const MatrixSparse<T>& A);

 public:
  MatrixSparse(int sharedA, int me, int wDev, char ord, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz, const T *data,
      const H2O4GPU_INT *ptr, const H2O4GPU_INT *ind);
  MatrixSparse(int wDev, char ord, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz, const T *data,
      const H2O4GPU_INT *ptr, const H2O4GPU_INT *ind);
  MatrixSparse(char ord, H2O4GPU_INT m, H2O4GPU_INT n, H2O4GPU_INT nnz, const T *data,
      const H2O4GPU_INT *ptr, const H2O4GPU_INT *ind);
  MatrixSparse(int sharedA, int me, int wDev, const MatrixSparse<T>& A);
  MatrixSparse(int wDev, const MatrixSparse<T>& A);
  MatrixSparse(const MatrixSparse<T>& A);
  ~MatrixSparse();

  // Call this before any other method.
  int Init();

  // Method to equilibrate.
  int Equil(bool equillocal);

  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;
  int Mulvalid(char trans, T alpha, const T *x, T beta, T *y) const;

  // Getters
  const T* Data() const { return _data; }
  const T* Datay() const { return _datay; }
  const T* vData() const { return _vdata; }
  const T* vDatay() const { return _vdatay; }
  const T* Weight() const { return _weight; }
  
  const H2O4GPU_INT* Ptr() const { return _ptr; }
  const H2O4GPU_INT* Ind() const { return _ind; }
  H2O4GPU_INT Nnz() const { return _nnz; }
  Ord Order() const { return _ord; }
  int GetsharedA() const { return _sharedA; }
  int wDev() const { return _wDev; }
  int Getme() const { return _me; }
  int Datatype() const { return _datatype; }
  int DoPredict() const { return _dopredict; }
};

}  // namespace h2o4gpu

#endif  // MATRIX_MATRIX_SPARSE_H_

