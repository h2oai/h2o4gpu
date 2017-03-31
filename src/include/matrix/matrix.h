#ifndef MATRIX_MATRIX_H_
#define MATRIX_MATRIX_H_

#include <memory>

namespace pogs {

template <typename T>
class Matrix {
 protected:
  const size_t _m, _n;

  void *_info;

  bool _done_init;

 public:
  Matrix(size_t m, size_t n) : _m(m), _n(n), _info(0), _done_init(false) { };

  virtual ~Matrix() { };

  // Call this methods to initialize the matrix.
  virtual int Init() = 0;

  // Method to equilibrate and return equilibration vectors.
  virtual int Equil(T *d, T *e, bool equillocal) = 0;

  // Method to multiply by A and A^T.
  virtual int Mul(char trans, T alpha, const T *x, T beta, T *y) const = 0;

  // Get dimensions and check if initialized
  size_t Rows() const { return _m; }
  size_t Cols() const { return _n; }
  bool IsInit() const { return _done_init; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_H_

