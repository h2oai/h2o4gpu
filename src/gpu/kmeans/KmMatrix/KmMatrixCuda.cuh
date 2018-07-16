#ifndef KM_MATRIX_CUDA_CUH_
#define KM_MATRIX_CUDA_CUH_

#include "KmMatrix.hpp"

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
  thrust::device_vector<T> _d_vector;
  thrust::host_vector<T> _h_vector;

  bool _on_device;
  KmMatrix<T>* _matrix;

  void host_to_device();
  void device_to_host();

 public:
  CudaKmMatrixImpl(KmMatrix<T> * _par);
  CudaKmMatrixImpl(const thrust::host_vector<T>& _h_vec,
                   KmMatrix<T>* _par);
  CudaKmMatrixImpl(const KmMatrixProxy<T>& _other,
                   KmMatrix<T>* _par);

  virtual ~CudaKmMatrixImpl();

  bool on_device() const override;

  virtual T* host_ptr() override;
  virtual T* dev_ptr() override;
};

}  // MkMatrix
}  // H204GPU
#endif