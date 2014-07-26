#ifndef GSL_MATRIX_H_
#define GSL_MATRIX_H_

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "gsl_vector.h"

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
matrix<T> matrix_view_array(T *base, size_t n1, size_t n2) {
  matrix<T> mat;
  mat.size1 = n1;
  mat.size2 = n2;
  mat.tda = n2;
  mat.data = base;
  return mat;
}

template <typename T>
inline T matrix_get(const matrix<T> *A, size_t i, size_t j) {
  return A->data[i * A->tda + j];
}

template <typename T>
inline void matrix_set(const matrix<T> *A, size_t i, size_t j, T x) {
  A->data[i * A->tda + j] = x;
}

template<typename T>
void matrix_set_all(matrix<T> *A, T x) {
  for (unsigned int i = 0; i < A->size1; ++i)
    for (unsigned int j = 0; j < A->size2; ++j)
      matrix_set(A, i, j, x);
}

template <typename T>
void matrix_memcpy(matrix<T> *A, const matrix<T> *B) {
  if (A->tda == A->size2 && B->tda == B->size2) {
    memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(T));
  } else {
    for (unsigned int i = 0; i < A->size1; ++i) 
      for (unsigned int j = 0; j < A->size2; ++j) 
        matrix_set(A, i, j, matrix_get(B, i, j));
  }
}

template <typename T>
void matrix_memcpy(matrix<T> *A, const T *B) {
  if (A->tda == A->size2) {
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(T));
  } else {
    for (unsigned int i = 0; i < A->size1; ++i) 
      for (unsigned int j = 0; j < A->size2; ++j) 
        matrix_set(A, i, j, B[i * A->size2 + j]);
  }
}

template <typename T>
void matrix_memcpy(T *A, const matrix<T> *B) {
  if (B->tda == B->size2) {
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(T));
  } else {
    for (unsigned int i = 0; i < B->size1; ++i) 
      for (unsigned int j = 0; j < B->size2; ++j) 
        A[i * B->size2 + j] = matrix_get(B, i, j);
  }
}

template <typename T>
void matrix_print(const matrix<T> *A) {
  for (unsigned int i = 0; i < A->size1; ++i) {
    for (unsigned int j = 0; j < A->size2; ++j)
      printf("%e ", matrix_get(A, i, j));
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

#endif  // GSL_MATRIX_H_

