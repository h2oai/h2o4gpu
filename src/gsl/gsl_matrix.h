#ifndef GSL_MATRIX_H_
#define GSL_MATRIX_H_

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "cblas_def.h"
#include "gsl_vector.h"

// Gnu Scientific Library
namespace gsl {

// Matrix Class
template <typename T, CBLAS_ORDER O>
struct matrix {
  size_t size1, size2, tda;
  T* data;
};

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_alloc(size_t m, size_t n) {
  matrix<T, O> mat;
  mat.size1 = m;
  mat.size2 = n;
  if (O == CblasRowMajor)
    mat.tda = n;
  else
    mat.tda = m;
  mat.data = new T[m * n];
  return mat;
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_calloc(size_t m, size_t n) {
  matrix<T, O> mat = matrix_alloc<T, O>(m, n);
  memset(mat.data, 0, m * n * sizeof(T));
  return mat;
}

template<typename T, CBLAS_ORDER O>
void matrix_free(matrix<T, O> *A) {
  delete [] A->data;
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_submatrix(matrix<T, O> *A, size_t i, size_t j, size_t n1,
                              size_t n2) {
  matrix<T, O> submat;
  submat.size1 = n1;
  submat.size2 = n2;
  if (O == CblasRowMajor)
    submat.data = A->data + i * A->tda + j;
  else
    submat.data = A->data + i + j * A->tda;
  submat.tda = A->tda;
  return submat;
}

template <typename T, CBLAS_ORDER O>
vector<T> matrix_row(matrix<T, O> *A, size_t i) {
  vector<T> v;
  v.size = A->size2;
  if (O == CblasRowMajor) {
    v.data = A->data + i * A->tda;
    v.stride = static_cast<size_t>(1);
  } else {
    v.data = A->data + i;
    v.stride = A->tda;
  }
  return v;
}

template <typename T, CBLAS_ORDER O>
vector<T> matrix_column(matrix<T, O> *A, size_t j) {
  vector<T> v;
  v.size = A->size1;
  if (O == CblasRowMajor) {
    v.data = A->data + j;
    v.stride = A->tda;
  } else {
    v.data = A->data + j * A->tda;
    v.stride = static_cast<size_t>(1);
  }
  return v;
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_view_array(const T *base, size_t n1, size_t n2) {
  matrix<T, O> mat;
  mat.size1 = n1;
  mat.size2 = n2;
  if (O == CblasRowMajor)
    mat.tda = n2;
  else
    mat.tda = n1;
  mat.data = const_cast<T*>(base);
  return mat;
}

template <typename T, CBLAS_ORDER O>
matrix<T, O> matrix_view_array(T *base, size_t n1, size_t n2) {
  matrix<T, O> mat;
  mat.size1 = n1;
  mat.size2 = n2;
  if (O == CblasRowMajor)
    mat.tda = n2;
  else
    mat.tda = n1;
  mat.data = base;
  return mat;
}

template <typename T, CBLAS_ORDER O>
inline T matrix_get(const matrix<T, O> *A, size_t i, size_t j) {
  if (O == CblasRowMajor)
    return A->data[i * A->tda + j];
  else
    return A->data[i + j * A->tda];
}

template <typename T, CBLAS_ORDER O>
inline void matrix_set(const matrix<T, O> *A, size_t i, size_t j, T x) {
  if (O == CblasRowMajor)
    A->data[i * A->tda + j] = x;
  else
    A->data[i + j * A->tda] = x;
}

template<typename T, CBLAS_ORDER O>
void matrix_set_all(matrix<T, O> *A, T x) {
  if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      for (unsigned int j = 0; j < A->size2; ++j)
        matrix_set(A, i, j, x);
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      for (unsigned int i = 0; i < A->size1; ++i)
        matrix_set(A, i, j, x);
}

template <typename T, CBLAS_ORDER O>
void matrix_memcpy(matrix<T, O> *A, const matrix<T, O> *B) {
  if ((O == CblasRowMajor && A->tda == A->size2 && B->tda == B->size2) ||
      (O == CblasColMajor && A->tda == A->size1 && B->tda == B->size1))
    memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(T));
  else if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i) 
      for (unsigned int j = 0; j < A->size2; ++j) 
        matrix_set(A, i, j, matrix_get(B, i, j));
  else 
    for (unsigned int j = 0; j < A->size2; ++j) 
      for (unsigned int i = 0; i < A->size1; ++i) 
        matrix_set(A, i, j, matrix_get(B, i, j));
}

template <typename T, CBLAS_ORDER O>
void matrix_memcpy(matrix<T, O> *A, const T *B) {
  if ((O == CblasRowMajor && A->tda == A->size2) ||
      (O == CblasColMajor && A->tda == A->size1))
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(T));
  else if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i) 
      for (unsigned int j = 0; j < A->size2; ++j) 
        matrix_set(A, i, j, B[i * A->size2 + j]);
  else
    for (unsigned int j = 0; j < A->size2; ++j) 
      for (unsigned int i = 0; i < A->size1; ++i) 
        matrix_set(A, i, j, B[i + j * A->size1]);
}

template <typename T, CBLAS_ORDER O>
void matrix_memcpy(T *A, const matrix<T, O> *B) {
  if ((O == CblasRowMajor && B->tda == B->size2) ||
      (O == CblasColMajor && B->tda == B->size1)) 
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(T));
  else if (O == CblasRowMajor)
    for (unsigned int i = 0; i < B->size1; ++i) 
      for (unsigned int j = 0; j < B->size2; ++j) 
        A[i * B->size2 + j] = matrix_get(B, i, j);
  else 
    for (unsigned int j = 0; j < B->size2; ++j) 
      for (unsigned int i = 0; i < B->size1; ++i) 
        A[i + j * B->size1] = matrix_get(B, i, j);
}

// template <typename T, CBLAS_ORDER O>
// void matrix_print(const matrix<T, O> *A) {
//   for (unsigned int i = 0; i < A->size1; ++i) {
//     for (unsigned int j = 0; j < A->size2; ++j)
//       Printf("%e ", matrix_get(A, i, j));
//     Printf("\n");
//   }
//   Printf("\n");
// }

template <typename T, CBLAS_ORDER O>
vector<T> matrix_diagonal(matrix<T, O> *A) {
  vector<T> v;
  v.data = A->data;
  v.stride = A->tda + 1;
  v.size = std::min(A->size1, A->size2);
  return v;
}

template <typename T, CBLAS_ORDER O>
void matrix_scale(matrix<T, O> *A, T x) {
  if (O == CblasRowMajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      for (unsigned int j = 0; j < A->size2; ++j)
        A->data[i * A->tda + j] *= x;
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      for (unsigned int i = 0; i < A->size1; ++i)
        A->data[i + j * A->tda] *= x;
}

}  // namespace gsl

#endif  // GSL_MATRIX_H_

