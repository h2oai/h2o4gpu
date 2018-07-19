#ifndef KM_MATRIX_CUDA_CUH_
#define KM_MATRIX_CUDA_CUH_

#include "KmMatrix.hpp"
#include "thrust/device_vector.h";
#include <memory>

namespace H2O4GPU {
namespace KMeans {

template <typename T>
class KmMatrix;

template <typename T>
class KmMatrixImpl;

template <typename T>
class KmMatrixProxy;

struct CudaInfo {
  int n_devices;
  int * _devices;

  CudaInfo (int _n_devices)
  : n_devices(_n_devices) {
    _devices = new int[_n_devices];
  }
  ~CudaInfo () {
    delete [] _devices;
  }
};

template <typename T>
class CudaKmMatrixImpl : public KmMatrixImpl<T> {
 private:
  thrust::device_vector<T> d_vector_;
  thrust::host_vector<T> h_vector_;

  bool on_device_;
  KmMatrix<T>* matrix_;

  void host_to_device();
  void device_to_host();

 public:
  CudaKmMatrixImpl(KmMatrix<T> * _par);
  CudaKmMatrixImpl(const thrust::host_vector<T>& _h_vec, KmMatrix<T>* _par);
  CudaKmMatrixImpl(size_t _size, KmMatrix<T> * _par);
  CudaKmMatrixImpl(KmMatrix<T>& _other,
                   size_t _start, size_t _size, size_t _stride,
                   KmMatrix<T> * _par);

  CudaKmMatrixImpl(const CudaKmMatrixImpl<T>&) = delete;
  CudaKmMatrixImpl(CudaKmMatrixImpl<T>&&) = delete;

  virtual ~CudaKmMatrixImpl();

  virtual void set_interface(KmMatrix<T>* _par) override;

  void operator=(const CudaKmMatrixImpl<T>&) = delete;
  void operator=(CudaKmMatrixImpl<T>&&) = delete;

  KmMatrix<T> stack(KmMatrix<T>& _second, KmMatrixDim _dim) override;

  virtual T* host_ptr() override;
  virtual T* dev_ptr() override;

  virtual size_t size() const override;

  // virtual bool equal(std::shared_ptr<CudaKmMatrixImpl<T>>& _rhs);
  virtual bool equal(KmMatrix<T>& _rhs);

  virtual bool on_device() const override;

  friend KmMatrix<T>;
};

}  // MkMatrix
}  // H204GPU

#endif
