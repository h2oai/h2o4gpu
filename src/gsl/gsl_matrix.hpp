#ifndef GSL_MATRIX_HPP_
#define GSL_MATRIX_HPP_

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "gsl_vector.hpp"

// Gnu Scientific Library
namespace gsl {

// Matrix Class
template <typename T>
struct matrix {
  size_t size1, size2, tda;
  T* data;
};

template <typename T>
matrix<T> matrix_alloc(size_t m, size_t n) {
  matrix<T> mat;
  mat.size1 = m;
  mat.size2 = n;
  mat.tda = n;
  mat.data = new T[m * n];
  return mat;
}

template <typename T>
matrix<T> matrix_calloc(size_t m, size_t n) {
  matrix<T> mat = matrix_alloc<T>(m, n);
  memset(mat.data, 0, m * n * sizeof(T));
  return mat;
}

template<typename T>
void matrix_free(matrix<T> *A) {
  delete [] A->data;
}

template <typename T>
matrix<T> matrix_submatrix(matrix<T> *A, size_t i, size_t j, size_t n1,
                           size_t n2) {
  matrix<T> submat;
  submat.size1 = n1;
  submat.size2 = n2;
  submat.data = A->data + i * A->tda + j;
  submat.tda = A->tda;
  return submat;
}

template <typename T>
vector<T> matrix_row(matrix<T> *A, size_t i) {
  vector<T> v;
  v.size = A->size2;
  v.data = A->data + i * A->tda;
  v.stride = static_cast<size_t>(1);
  return v;
}

template <typename T>
vector<T> matrix_column(matrix<T> *A, size_t j) {
  vector<T> v;
  v.size = A->size1;
  v.data = A->data + j;
  v.stride = A->tda;
  return v;
}

template <typename T>
const matrix<T> matrix_const_view_array(const T *base, size_t n1, size_t n2) {
  matrix<T> mat;
  mat.size1 = n1;
  mat.size2 = n2;
  mat.tda = n2;
  mat.data = const_cast<T*>(base);
  return mat;
}

template <typename T>
T matrix_get(matrix<T> *A, size_t i, size_t j) {
  return A->data[i * A->tda + j];
}

template <typename T>
void matrix_set(matrix<T> *A, size_t i, size_t j, T x) {
  A->data[i * A->tda + j] = x;
}

// TODO: Take tda into account properly
template <typename T>
void matrix_memcpy(matrix<T> *A, const matrix<T> *B) {
  memcpy(A->data, B->data, A->tda * A->size1 * sizeof(T));
}

template <typename T>
void matrix_memcpy(matrix<T> *A, const T *B) {
  memcpy(A->data, B, A->tda * A->size1 * sizeof(T));
}

template <typename T>
void matrix_memcpy(T *A, const matrix<T> *B) {
  memcpy(A, B->data, B->tda * B->size1 * sizeof(T));
}

template <typename T>
void matrix_print(const matrix<T> &A) {
  for (unsigned int i = 0; i < A.size1; ++i) {
    for (unsigned int j = 0; j < A.size2; ++j)
      printf("%e ", A.data[i * A.tda + j]);
    printf("\n");
  }
  printf("\n");
}

template <typename T>
vector<T> matrix_diagonal(matrix<T> *A) {
  vector<T> v;
  v.data = A->data;
  v.stride = A->tda + 1;
  v.size = std::min(A->size1, A->size2);
  return v;
}

template <typename T>
void matrix_scale(matrix<T> *A, T x) {
  for (unsigned int i = 0; i < A->size1; ++i)
    for (unsigned int j = 0; j < A->size2; ++j)
      A->data[i * A->tda + j] *= x;
}

}  // namespace gsl

#endif  // GSL_MATRIX_HPP_

