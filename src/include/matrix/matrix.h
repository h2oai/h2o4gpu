#ifndef MATRIX_MATRIX_H_
#define MATRIX_MATRIX_H_

#include <memory>

namespace h2oaiglm {

template <typename T>
class Matrix {
 protected:
  const size_t _m, _n, _mvalid;

  void *_info, *_infoy, *_vinfo, *_vinfoy;

  bool _done_init, _done_alloc, _done_equil;

 public:
 Matrix(size_t m, size_t n) : _m(m), _n(n), _mvalid(0), _info(0), _infoy(0), _vinfo(0), _vinfoy(0), _done_init(false), _done_alloc(false), _done_equil(false) { };
 Matrix(size_t m, size_t n, size_t mValid) : _m(m), _n(n), _mvalid(mValid), _info(0), _infoy(0), _vinfo(0), _vinfoy(0), _done_init(false), _done_alloc(false), _done_equil(false) { };

  virtual ~Matrix() { };

  // Call this methods to initialize the matrix.
  virtual int Init() = 0;

  // Method to equilibrate and return equilibration vectors.
  virtual int Equil(bool equillocal) = 0;

  // Method to multiply by A and A^T.
  virtual int Mul(char trans, T alpha, const T *x, T beta, T *y) const = 0;

  // Get dimensions and check if initialized
  size_t Rows() const { return _m; } // trainX rows
  size_t Cols() const { return _n; } // trainX cols
  size_t ValidRows() const { return _mvalid; } // validX rows (validX has same columns as trainX)
  bool IsInit() const { return _done_init; }
};

}  // namespace h2oaiglm

#endif  // MATRIX_MATRIX_H_

