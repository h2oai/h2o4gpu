/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include "Eigen/Dense"
#include "KmMatrix/KmMatrix.hpp"
// #include <vector>

namespace H2O4GPU{
namespace KMeans {

// Wrappers for Eigen matrix and vector
template <typename T>
struct EiMatrix;

template <>
struct EiMatrix <float> {
  using type = Eigen::MatrixXf;
};
template <>
struct EiMatrix <double> {
  using type = Eigen::MatrixXd;
};
template <>
struct EiMatrix<int> {
  using type = Eigen::MatrixXi;
};

template <typename T>
struct EiVector;
template <>
struct EiVector<float> {
  using type = Eigen::VectorXf;
};
template <>
struct EiVector<double> {
  using type = Eigen::VectorXd;
};
template <>
struct EiVector<int> {
  using type = Eigen::VectorXi;
};

// Work around for shared memory
// https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name
template <typename T>
struct KmShardMem;

template <>
struct KmShardMem<float> {
  __device__ float * ptr() {
    extern __shared__ __align__(sizeof(float)) float s_float[];
    return s_float;
  }
};

template <>
struct KmShardMem<double> {
  __device__ double * ptr() {
    extern __shared__ __align__(sizeof(double)) double s_double[];
    return s_double;
  }
};

template <>
struct KmShardMem<int> {
  __device__ int * ptr() {
    extern __shared__ __align__(sizeof(int)) int s_int[];
    return s_int;
  }
};

#define MA_T(T)                                 \
  typename EiMatrix<T>::type
#define VE_T(T)                                 \
  typename EiVector<T>::type

template <typename T>
struct kMParam {
  T* ptr;
  size_t rows;
  size_t cols;

  kMParam(T* _ptr, size_t _rows, size_t _cols) :
      ptr (_ptr), rows (_rows), cols (_cols) {}
  kMParam(size_t _rows, size_t _cols):
      rows (_rows), cols (_cols) {}
  kMParam(size_t _cols) : cols (_cols) {}
};

template <typename T>
struct kVParam {
  T* ptr;
  size_t size;
  kVParam(T* _ptr, size_t _size) : ptr(_ptr), size(_size) {}
};

// template <typename T>
// struct HostDeviceVector {
//  private:
//   kMParam<T> param;
//   // thrust::device_vector<T> _d_vector;
//   std::vector<T>* _h_vector;

//  public:
//   HostDeviceVector (const std::vector<T>& _h_vec, size_t _cols) :
//       param(_cols) {
//     _h_vector = new std::vector<T>(_h_vec);
//   }
//   HostDeviceVector (const std::vector<T>& _h_vec,
//                     size_t _rows, size_t _cols) :
//       param(_rows, _cols) {
//     _h_vector = new std::vector<T>(_h_vec);
//   }
//   ~HostDeviceVector() { delete _h_vector; }
//   // HostDeviceVector (size_t _cols) :
//   // param.rows {1}, param.cols (_cols) {
//   //   _d_vector.resize(_cols);
//   // }
//   size_t rows() { return param.rows; }
//   size_t cols() { return param.cols; }
//   size_t size() { return param.rows * param.cols; }

//   // kMParam<T> kParam() {
//   //   param.ptr = _d_vector.data().get();
//   //   return param;
//   // }
// };


template <typename T>
class KmeansInitBase {
 public:
  virtual ~KmeansInitBase() {}
  virtual KmMatrix<T> operator()(KmMatrix<T>& data) = 0;
};

template <typename T>
struct KmeansLlInit : public KmeansInitBase<T> {
 private:
  double over_sample;
  int seed;

  T potential(MA_T(T)& data, MA_T(T)& centroids);
  T probability(MA_T(T)& data, MA_T(T)& controids);

 public:
  KmeansLlInit () : over_sample (2.0), seed (0) {}
  virtual ~KmeansLlInit () override {}

  MA_T(T) sample_centroids(MA_T(T)& data, MA_T(T)& centroids);

  // MA_T(T) operator()(MA_T(T)&) override;
  KmMatrix<T> operator()(KmMatrix<T>& data) override;
};

template <typename T1, typename T2>
T1 div_roundup(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}

}  // namespace Kmeans
}  // namespace H2O4GPU
