#ifndef GSL_VECTOR_HPP_
#define GSL_VECTOR_HPP_

#include <cstdio>
#include <cstring>

// Gnu Scientific Library
namespace gsl {

// Vector Class
template <typename T>
struct vector {
  size_t size, stride;
  T* data;
};

template <typename T>
vector<T> vector_alloc(size_t n) {
  vector<T> vec;
  vec.size = n;
  vec.stride = 1;
  vec.data = new T[n];
  return vec;
}

template <typename T>
vector<T> vector_calloc(size_t n) {
  vector<T> vec = vector_alloc<T>(n);
  memset(vec.data, 0, n * sizeof(T));
  return vec;
}

template<typename T>
void vector_free(vector<T> *x) {
  delete [] x->data;
}

template<typename T>
void vector_set_all(vector<T> *v, T x) {
  for (unsigned int i = 0; i < v->size; i++)
    v->data[i * v->stride] = x;
}

template<typename T>
void vector_set(vector<T> *v, size_t i, T x) {
  v->data[i * v->stride] = x;
}

template<typename T>
T vector_get(vector<T> *v, size_t i) {
  return v->data[i * v->stride];
}

template <typename T>
vector<T> vector_subvector(vector<T> *vec, size_t offset, size_t n) {
  vector<T> subvec;
  subvec.size = n;
  subvec.data = vec->data + offset * vec->stride;
  subvec.stride = vec->stride;
  return subvec;
}

// TODO: Take stride into account.
template <typename T>
void vector_memcpy(vector<T> *x, const vector<T> *y) {
  memcpy(x->data, y->data, x->size * sizeof(T));
}

template <typename T>
void vector_memcpy(vector<T> *x, const T *y) {
  memcpy(x->data, y, x->size * sizeof(T));
}

template <typename T>
void vector_memcpy(T *x, const vector<T> *y) {
  memcpy(x, y->data, y->size * sizeof(T));
}

template <typename T>
void print_vector(const vector<T> &x) {
  for (unsigned int i = 0; i < x.size; ++i)
    printf("%e ", x[i * x.stride]);
  printf("\n");
}

template <typename T>
void vector_scale(vector<T> *a, T x) {
  for (unsigned int i = 0; i < a->size; ++i)
    a->data[i * a->stride] *= x;
}

template <typename T>
void vector_add(vector<T> *a, const vector<T> *b) {
  for (unsigned int i = 0; i < a->size; i++)
    a->data[i * a->stride] += b->data[i * b->stride];
}

template <typename T>
void vector_sub(vector<T> *a, const vector<T> *b) {
  for (unsigned int i = 0; i < a->size; i++)
    a->data[i * a->stride] -= b->data[i * b->stride];
}

template <typename T>
void vector_mul(vector<T> *a, const vector<T> *b) {
  for (unsigned int i = 0; i < a->size; i++)
    a->data[i * a->stride] *= b->data[i * b->stride];
}

template <typename T>
void vector_div(vector<T> *a, const vector<T> *b) {
  for (unsigned int i = 0; i < a->size; i++)
    a->data[i * a->stride] /= b->data[i * b->stride];
}

template <typename T>
void vector_add_constant(vector<T> *a, const T x) {
  for (unsigned int i = 0; i < a->size; i++)
    a->data[i * a->stride] += x;
}

}  // namespace gsl

#endif  // GSL_VECTOR_HPP_

