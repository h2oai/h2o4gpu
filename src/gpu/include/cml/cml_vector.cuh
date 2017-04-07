#ifndef CML_VECTOR_CUH_
#define CML_VECTOR_CUH_

#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include <cstdio>

#include "cml/cml_defs.cuh"
#include "cml/cml_utils.cuh"
// TODO Clean this up
#include "../../include/interface_defs.h"

// Cuda Matrix Library
namespace cml {

// Vector Class
template <typename T>
struct vector {
  size_t size, stride;
  T* data;
  vector(T *data, size_t size, size_t stride)
      : size(size), stride(stride), data(data) { }
  vector() : size(0), stride(0), data(0) { }
};

// Helper methods
namespace {

template <typename T>
__global__ void __set_vector(T *data, T val, size_t stride, size_t size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < size; i += gridDim.x * blockDim.x)
    data[i * stride] = val;
}

template <typename T>
__global__ void __strided_memcpy(T *x, size_t stride_x, const T *y,
                                 size_t stride_y, size_t size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < size; i += gridDim.x * blockDim.x)
    x[i * stride_x] = y[i * stride_y];
}

template <typename T>
__global__ void __any_isnan(T *x, size_t stride, size_t size, int *result) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < size; i += gridDim.x * blockDim.x) {
    if (isnan(x[i * stride])) {
      *result = 1;
    }
  }
}

}  // namespace

template <typename T>
vector<T> vector_alloc(size_t n) {
  vector<T> vec;
  vec.size = n;
  vec.stride = 1;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&vec.data),
      n * sizeof(T));
  CudaCheckError(err);
  if (err != cudaSuccess)
    vec.data = 0;
  return vec;
}

template <typename T>
void vector_set_all(vector<T> *v, T x) {
  uint grid_dim = calc_grid_dim(v->size, kBlockSize);
  __set_vector<<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
}

template <typename T>
bool vector_any_isnan(vector<T> *v) {
  int *res_ptr, res;
  cudaMalloc(&res_ptr, sizeof(int));
  cudaMemset(res_ptr, 0, sizeof(int));
  uint grid_dim = calc_grid_dim(v->size, kBlockSize);
  __any_isnan<<<grid_dim, kBlockSize>>>(v->data, v->stride, v->size, res_ptr);
  cudaMemcpy(&res, res_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(res_ptr);
  return res > 0;
}


template <typename T>
vector<T> vector_calloc(size_t n) {
  vector<T> vec = vector_alloc<T>(n);
  if (vec.data != 0)
    vector_set_all(&vec, static_cast<T>(0));
  return vec;
}

template<typename T>
void vector_free(vector<T> *x) {
  cudaError_t err = cudaFree(x->data);
  CudaCheckError(err);
}

template <typename T>
vector<T> vector_subvector(vector<T> *vec, size_t offset, size_t n) {
  vector<T> subvec;
  subvec.size = n;
  subvec.data = vec->data + offset * vec->stride;
  subvec.stride = vec->stride;
  return subvec;
}

template <typename T>
vector<T> vector_view_array(T *base, size_t n) {
  return vector<T>(base, n, 1);
}

template <typename T>
const vector<T> vector_view_array(const T *base, size_t n) {
  return vector<T>(const_cast<T*>(base), n, 1);
}


template <typename T>
void vector_memcpy(vector<T> *x, const vector<T> *y) {
  if (x->stride == 1 && y->stride == 1) {
    cudaError_t err;
    err = cudaMemcpy(reinterpret_cast<void*>(x->data),
        reinterpret_cast<const void*>(y->data), x->size * sizeof(T),
        cudaMemcpyDefault);
    CudaCheckError(err);
  } else {
    uint grid_dim = calc_grid_dim(x->size, kBlockSize);
    __strided_memcpy<<<grid_dim, kBlockSize>>>(x->data, x->stride, y->data,
        y->stride, x->size);
  }
}

template <typename T>
void vector_memcpy(vector<T> *x, const T *y) {
  cudaError_t err;
  if (x->stride == 1)
     err = cudaMemcpy(reinterpret_cast<void*>(x->data),
         reinterpret_cast<const void*>(y), x->size * sizeof(T),
         cudaMemcpyDefault);
  else
    for (unsigned int i = 0; i < x->size; ++i)
      err = cudaMemcpy(reinterpret_cast<void*>(x->data + i),
          reinterpret_cast<const void*>(y + i), sizeof(T),
          cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void vector_memcpy(T *x, const vector<T> *y) {
  cudaError_t err;
  if (y->stride == 1)
     err = cudaMemcpy(reinterpret_cast<void*>(x),
         reinterpret_cast<const void*>(y->data), y->size * sizeof(T),
         cudaMemcpyDefault);
  else
    for (unsigned int i = 0; i < y->size; ++i)
      err = cudaMemcpy(reinterpret_cast<void*>(x + i),
          reinterpret_cast<const void*>(y->data + i), sizeof(T),
          cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void vector_print(const vector<T> *x) {
  T* x_ = new T[x->size * x->stride];
  vector_memcpy(x_, x);
  for (unsigned int i = 0; i < x->size; ++i)
    Printf("(%u, %e) ", i, x_[i * x->stride]);
  Printf("\n");
  delete [] x_;
}

template <typename T>
void vector_scale(vector<T> *a, T x) {
  strided_range<thrust::device_ptr<T> > idx(
      thrust::device_pointer_cast(a->data),
      thrust::device_pointer_cast(a->data + a->stride * a->size), a->stride);
  thrust::transform(idx.begin(), idx.end(), thrust::constant_iterator<T>(x),
      idx.begin(), thrust::multiplies<T>());
}

template <typename T>
void vector_mul(vector<T> *a, const vector<T> *b) {
  strided_range<thrust::device_ptr<T> > idx_a(
      thrust::device_pointer_cast(a->data),
      thrust::device_pointer_cast(a->data + a->stride * a->size), a->stride);
  strided_range<thrust::device_ptr<T> > idx_b(
      thrust::device_pointer_cast(b->data),
      thrust::device_pointer_cast(b->data + b->stride * b->size), b->stride);
  thrust::transform(idx_a.begin(), idx_a.end(), idx_b.begin(), idx_a.begin(),
      thrust::multiplies<T>());
}

template <typename T>
void vector_div(vector<T> *a, const vector<T> *b) {
  strided_range<thrust::device_ptr<T> > idx_a(
      thrust::device_pointer_cast(a->data),
      thrust::device_pointer_cast(a->data + a->stride * a->size), a->stride);
  strided_range<thrust::device_ptr<T> > idx_b(
      thrust::device_pointer_cast(b->data),
      thrust::device_pointer_cast(b->data + b->stride * b->size), b->stride);
  thrust::transform(idx_a.begin(), idx_a.end(), idx_b.begin(), idx_a.begin(),
      thrust::divides<T>());
}

template <typename T>
void vector_add_constant(vector<T> *a, const T x) {
  strided_range<thrust::device_ptr<T> > idx(
      thrust::device_pointer_cast(a->data),
      thrust::device_pointer_cast(a->data + a->stride * a->size), a->stride);
  thrust::transform(idx.begin(), idx.end(), thrust::constant_iterator<T>(x),
      idx.begin(), thrust::plus<T>());
}

}  // namespace cml

#endif  // CML_VECTOR_CUH_

