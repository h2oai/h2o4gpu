/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_vector.cuh"
#include "equil_helper.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "util.h"
#include "timer.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/advance.h>
#include <cmath>
#include <limits>
#include <thrust/fill.h>
#include "../include/cuda_utils.h"

namespace h2o4gpu {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
const NormTypes kNormEquilibrate = kNorm2; 
const NormTypes kNormNormalize   = kNormFro;

template<typename T>
struct GpuData {
  const T *orig_data; // pointer to data on CPU
  cublasHandle_t handle; // handle for data on GPU
  GpuData(const T *orig_data) : orig_data(orig_data) {
    cublasCreate(&handle);
    //    fprintf(stderr,"HEREstart: %ld\n",handle); fflush(stderr);
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
    //    fprintf(stderr,"HEREend: %ld\n",handle); fflush(stderr);

    if(handle!=NULL) cublasDestroy(handle);
    DEBUG_CUDA_CHECK_ERR();
  }
};

cublasOperation_t OpToCublasOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
}

  
template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, const MatrixDense<T>& A);

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDense<T>::Ord ord, T *data);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixDense Implementation /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  // original MatrixDense where only trainX and no trainY or validX or validY
  // Used by elastic_net.cpp to pass CPU data and put on GPU
template <typename T>
MatrixDense<T>::MatrixDense(int sharedA, int wDev, char ord, size_t m, size_t n, const T *data)
  : Matrix<T>(m, n, 0), _sharedA(sharedA), _wDev(wDev), _datatype(0), _dopredict(0), _data(0), _de(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  _me=_wDev; // assume thread same as wDev if not given
  _datay=NULL;
  _vdata=NULL;
  _vdatay=NULL;
  _weight=NULL;

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  DEBUG_FPRINTF(stderr,"MatrixDense1: ord=%c m=%d n=%d\n",ord,(int)m,(int)n);fflush(stderr);
  
#ifdef DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  fprintf(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev); fflush(stderr);
#endif
  
  // Set GPU specific _info.


  PUSH_RANGE("MDnew",MDnew,1);
  GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
  this->_info = reinterpret_cast<void*>(info);
  GpuData<T> *infoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
  this->_infoy = reinterpret_cast<void*>(infoy);
  GpuData<T> *vinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
  this->_vinfo = reinterpret_cast<void*>(vinfo);
  GpuData<T> *vinfoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
  this->_vinfoy = reinterpret_cast<void*>(vinfoy);
  GpuData<T> *weightinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
  this->_weightinfo = reinterpret_cast<void*>(weightinfo);
  POP_RANGE("MDnew",MDnew,1);

  if(!this->_done_alloc){
    this->_done_alloc = true;
    // unlike CPU case, input pointer is always CPU so have to always allocate on GPU when calling this function.  So no use of sharedA related to pointer copy like in CPU case.

    // Copy Matrix to GPU.
    PUSH_RANGE("MDsend",MDsend,1);
    //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
    double t0 = timer<double>();
    cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
    double t1 = timer<double>();
    cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    double t2 = timer<double>();
#ifdef DEBUG    
    printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
    printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
#endif

    cudaMalloc(&_de, (m + n) * sizeof(T));
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_de[0]));
    T fill_value=0.0;
    thrust::fill(dev_ptr, dev_ptr + (m + n), fill_value);
    if(sharedA>0){
      Init(); // does nothing right now
      Equil(1); // JONTODO: Hack for now.  Need to pass equil
    }
    POP_RANGE("MDsend",MDsend,1);
  }
}
template <typename T>
MatrixDense<T>::MatrixDense(char ord, size_t m, size_t n, const T *data)
  : MatrixDense<T>(0, 0, ord, m, n, data){} // assume sharedA=0 and thread=wDev=0 if not given

template <typename T>
MatrixDense<T>::MatrixDense(int sharedA, int wDev, int datatype, char ord, size_t m, size_t n, T *data)
  : Matrix<T>(m, n, 0), _sharedA(sharedA), _wDev(wDev), _datatype(datatype), _dopredict(0), _data(0),_de(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  _me=_wDev; // assume thread=wDev if not given
  _datay=NULL;
  _vdata=NULL;
  _vdatay=NULL;
  _weight=NULL;

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  DEBUG_FPRINTF(stderr,"MatrixDense2: ord=%c m=%d n=%d\n",ord,(int)m,(int)n);fflush(stderr);
  
#ifdef DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  fprintf(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev); fflush(stderr);
#endif
  

  if(datatype==1){
    // input data pointer is already on GPU on this wDev, so just copy pointer
    // no info->orig_data, so send 0 to GpuData
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    POP_RANGE("MDnew",MDnew,1);
    GpuData<T> *infoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_infoy = reinterpret_cast<void*>(infoy);
    GpuData<T> *vinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_vinfo = reinterpret_cast<void*>(vinfo);
    GpuData<T> *vinfoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_vinfoy = reinterpret_cast<void*>(vinfoy);
    GpuData<T> *weightinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_weightinfo = reinterpret_cast<void*>(weightinfo);
    
    // source pointer is on this GPU

    // just copy GPU pointer
    _data = data;
    if(!this->_done_alloc){
      this->_done_alloc = true;
      cudaMalloc(&_de, (m + n) * sizeof(T));
      thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_de[0]));
      T fill_value=0.0;
      thrust::fill(dev_ptr, dev_ptr + (m + n), fill_value);

      Init(); // does nothing right now
      Equil(1); // JONTODO: Hack for now.  Need to pass equil
    }

  }
  else{
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    GpuData<T> *infoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_infoy = reinterpret_cast<void*>(infoy);
    GpuData<T> *vinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_vinfo = reinterpret_cast<void*>(vinfo);
    GpuData<T> *vinfoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_vinfoy = reinterpret_cast<void*>(vinfoy);
    GpuData<T> *weightinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_weightinfo = reinterpret_cast<void*>(weightinfo);
    POP_RANGE("MDnew",MDnew,1);

    if(!this->_done_alloc){
      this->_done_alloc = true;

      // Unlike CPU case, can't pointer copy as going from CPU to GPU
      
      // Copy CPU Matrix to GPU.
      PUSH_RANGE("MDsend",MDsend,1);
      //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
      double t0 = timer<double>();
      cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
      double t1 = timer<double>();
      cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
      cudaMalloc(&_de, (m + n) * sizeof(T));
      thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_de[0]));
      T fill_value=0.0;
      thrust::fill(dev_ptr, dev_ptr + (m + n), fill_value);

      if(sharedA>0){
        Init(); // does nothing right now
        Equil(1); // JONTODO: Hack for now.  Need to pass equil
      }
      double t2 = timer<double>();
#ifdef DEBUG
      printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
      printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
#endif
      POP_RANGE("MDsend",MDsend,1);
    }
  }
}


  // like original MatrixDense, but also feed in CPU data for trainY, validX, and validY
  // Used by elastic_net_ptr.cpp to pass CPU data and put on GPU
template <typename T>
MatrixDense<T>::MatrixDense(int sharedA, int me, int wDev, char ord, size_t m, size_t n, size_t mValid, const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight)
  : Matrix<T>(m, n, mValid), _sharedA(sharedA), _me(me), _wDev(wDev), _datatype(0), _dopredict(0), _data(0), _datay(0), _vdata(0), _vdatay(0), _weight(0), _de(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  DEBUG_FPRINTF(stderr,"MatrixDense3: ord=%c m=%d n=%d mValid=%d\n",ord,(int)m,(int)n,int(mValid));fflush(stderr);
  
#ifdef DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  fprintf(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev); fflush(stderr);
#endif


  // source pointer is on CPU
  // Set GPU specific _info.
  PUSH_RANGE("MDnew",MDnew,1);
  GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
  GpuData<T> *infoy = new GpuData<T>(datay); // new structure (holds pointer to data and GPU handle)
  GpuData<T> *vinfo = new GpuData<T>(vdata); // new structure (holds pointer to data and GPU handle)
  GpuData<T> *vinfoy = new GpuData<T>(vdatay); // new structure (holds pointer to data and GPU handle)
  GpuData<T> *weightinfo = new GpuData<T>(weight); // new structure (holds pointer to data and GPU handle)
  this->_info = reinterpret_cast<void*>(info);
  this->_infoy = reinterpret_cast<void*>(infoy);
  this->_vinfo = reinterpret_cast<void*>(vinfo);
  this->_vinfoy = reinterpret_cast<void*>(vinfoy);
  this->_weightinfo = reinterpret_cast<void*>(weightinfo);
  POP_RANGE("MDnew",MDnew,1);


  if(!this->_done_alloc){
    this->_done_alloc = true;

    // Unlike CPU case, can't pointer copy even if sharedA!=0
    
    // Copy Matrix to GPU.
    PUSH_RANGE("MDsend",MDsend,1);
    //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
    double t0 = timer<double>();
    cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
    cudaMalloc(&_datay, this->_m * sizeof(T)); // allocate on GPU
    cudaMalloc(&_vdata, this->_mvalid * this->_n * sizeof(T)); // allocate on GPU
    cudaMalloc(&_vdatay, this->_mvalid * sizeof(T)); // allocate on GPU
    cudaMalloc(&_weight, this->_m * sizeof(T)); // allocate on GPU
    double t1 = timer<double>();

    cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    if(infoy->orig_data){
      cudaMemcpy(_datay, infoy->orig_data, this->_m * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
      _dopredict=0;
    }
    else{
      _dopredict=1;
    }
    if(vinfo->orig_data){
      cudaMemcpy(_vdata, vinfo->orig_data, this->_mvalid * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    }
    else{
      if(this->_mvalid>0){ fprintf(stderr,"vinfo->orig_data NULL but this->_mvalid>0\n"); fflush(stderr); exit(1); }
    }
    if(vinfoy->orig_data){
      cudaMemcpy(_vdatay, vinfoy->orig_data, this->_mvalid * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    }
    else{
      if(this->_mvalid>0){ fprintf(stderr,"vinfoy->orig_data NULL but this->_mvalid>0\n"); fflush(stderr); exit(1); }
    }
    if(weightinfo->orig_data){
      cudaMemcpy(_weight, weightinfo->orig_data, this->_m * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    }
    else{// if no weights, set as unity weights
      thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_weight[0]));
      T fill_value=1.0;
      thrust::fill(dev_ptr, dev_ptr + m, fill_value);
    }
    cudaMalloc(&_de, (m + n) * sizeof(T));
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_de[0]));
    T fill_value=0.0;
    thrust::fill(dev_ptr, dev_ptr + (m + n), fill_value);

    if(sharedA>0){
      Init(); // does nothing right now
      Equil(1); // JONTODO: Hack for now.  Need to pass equil
    }
    double t2 = timer<double>();
#ifdef DEBUG
    printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
    printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
#endif
    POP_RANGE("MDsend",MDsend,1);
  }
}

  template <typename T>
  MatrixDense<T>::MatrixDense(int wDev, char ord, size_t m, size_t n, size_t mValid, const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight)
    : MatrixDense<T>(0,wDev,wDev,ord,m,n,mValid,data,datay,vdata,vdatay,weight){} // assume sharedA=0 and source thread=wDev if not given

  // like original MatrixDense, but also feed in CPU data for trainY, validX, and validY
  // Used by elastic_net_ptr.cpp to pass CPU data and put on GPU
  // datatype=0: CPU pointer to data
  // datatype=1: GPU pointer to data
template <typename T>
MatrixDense<T>::MatrixDense(int sharedA, int me, int wDev, int datatype, char ord, size_t m, size_t n, size_t mValid, T *data, T *datay, T *vdata, T *vdatay, T *weight)
  : Matrix<T>(m, n, mValid), _sharedA(sharedA), _me(me), _wDev(wDev), _datatype(datatype), _dopredict(0), _data(0), _datay(0), _vdata(0), _vdatay(0), _weight(0), _de(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  DEBUG_FPRINTF(stderr,"%d\n", ord == 'r');
  DEBUG_FPRINTF(stderr,"%d\n", ord == 'c');
  DEBUG_FPRINTF(stderr,"ord=%c m=%d n=%d mValid=%d\n",ord,(int)m,(int)n,int(mValid));
  DEBUG_FPRINTF(stderr,"MatrixDense4: ord=%c m=%d n=%d mValid=%d\n",ord,(int)m,(int)n,int(mValid));

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

#ifdef DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  DEBUG_FPRINTF(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev);
#endif


  if(datatype==1){
    // source pointer is on GPU already
    // Set GPU specific _info.
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *infoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *weightinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    this->_infoy = reinterpret_cast<void*>(infoy);
    this->_vinfo = reinterpret_cast<void*>(vinfo);
    this->_vinfoy = reinterpret_cast<void*>(vinfoy);
    this->_weightinfo = reinterpret_cast<void*>(weightinfo);
    POP_RANGE("MDnew",MDnew,1);


    // Just copy GPU pointer
    _data = data;
    _datay = datay;
    _vdata = vdata;
    _vdatay = vdatay;
    _weight = weight;

    if(_datay) _dopredict=0;
    else _dopredict=1;

    
    if(_weight==NULL){
      DEBUG_FPRINTF(stderr,"datatype=1: making up unity weights: %d %p\n",m,&_weight);
      CUDACHECK(cudaMalloc(&_weight, m * sizeof(T))); // allocate on GPU
      thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_weight[0]));
      T fill_value=1.0;
      thrust::fill(dev_ptr, dev_ptr + m, fill_value);
    }
    
      
    if(!this->_done_alloc){
      this->_done_alloc = true;
      cudaMalloc(&_de, (m + n) * sizeof(T));
      thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_de[0]));
      T fill_value=0.0;
      thrust::fill(dev_ptr, dev_ptr + (m + n), fill_value);
      if(sharedA>0){
        Init(); // does nothing right now
        Equil(1); // JONTODO: Hack for now.  Need to pass equil
      }
    }
  }
  else{
    // source pointer is on CPU
    // Set GPU specific _info.
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *infoy = new GpuData<T>(datay); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfo = new GpuData<T>(vdata); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfoy = new GpuData<T>(vdatay); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *weightinfo = new GpuData<T>(weight); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    this->_infoy = reinterpret_cast<void*>(infoy);
    this->_vinfo = reinterpret_cast<void*>(vinfo);
    this->_vinfoy = reinterpret_cast<void*>(vinfoy);
    this->_weightinfo = reinterpret_cast<void*>(weightinfo);
    POP_RANGE("MDnew",MDnew,1);


    if(!this->_done_alloc){
      this->_done_alloc = true;
      // Copy CPU Matrix to GPU.
      PUSH_RANGE("MDsend",MDsend,1);
      //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
      double t0 = timer<double>();
      cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
      cudaMalloc(&_datay, this->_m * sizeof(T)); // allocate on GPU
      cudaMalloc(&_vdata, this->_mvalid * this->_n * sizeof(T)); // allocate on GPU
      cudaMalloc(&_vdatay, this->_mvalid * sizeof(T)); // allocate on GPU
      cudaMalloc(&_weight, this->_m * sizeof(T)); // allocate on GPU
      double t1 = timer<double>();

      cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
      if(infoy->orig_data){
        cudaMemcpy(_datay, infoy->orig_data, this->_m * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
        _dopredict=0;
      }
      else{
        _dopredict=1;
      }
      cudaMemcpy(_vdata, vinfo->orig_data, this->_mvalid * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
      cudaMemcpy(_vdatay, vinfoy->orig_data, this->_mvalid * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
      if(weightinfo->orig_data){
        cudaMemcpy(_weight, weightinfo->orig_data, this->_m * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
      }
      else{
        DEBUG_FPRINTF(stderr,"datatype=0: making up unity weights: %d\n",m);
        CUDACHECK(cudaMalloc(&_weight, this->_m * sizeof(T))); // allocate on GPU
        thrust::device_ptr<T> dev_ptr=thrust::device_pointer_cast(static_cast<T*>(_weight));
        T fill_value=1.0;
        thrust::fill(dev_ptr, dev_ptr + this->_m, fill_value);
      }
      cudaMalloc(&_de, (m + n) * sizeof(T));
      thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(static_cast<T*>(&_de[0]));
      T fill_value=0.0;
      thrust::fill(dev_ptr, dev_ptr + (m + n), fill_value);
      if(sharedA>0){
        Init(); // does nothing right now
        Equil(1); // JONTODO: Hack for now.  Need to pass equil
      }
      double t2 = timer<double>();
#ifdef DEBUG
      printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
      printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
#endif
      POP_RANGE("MDsend",MDsend,1);
    }
  }
}

template <typename T>
MatrixDense<T>::MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, size_t mValid, T *data, T *datay, T *vdata, T *vdatay, T *weight)
  : MatrixDense<T>(0,wDev,wDev,datatype,ord,m,n,mValid,data,datay,vdata,vdatay,weight){} // assume sharedA=0 and thread=wDev if not given

  // MatrixDense where input actual A object that contains all CPU information, but need to go from 1 GPU to multiple GPU
  // Used by elastic_net_ptr.cpp inside openmp loop for each core


template <typename T>
MatrixDense<T>::MatrixDense(int sharedA, int me, int wDev, const MatrixDense<T>& A)
  : Matrix<T>(A._m, A._n, A._mvalid), _sharedA(sharedA), _me(me), _wDev(wDev), _data(0),_de(0), _ord(A._ord) {

  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  DEBUG_FPRINTF(stderr,"MatrixDense5: ord=%c m=%d n=%d mValid=%d\n",A._ord,A._m,A._n,A._mvalid);

  PUSH_RANGE("MDnew",MDnew,2);
  GpuData<T> *info_A   = reinterpret_cast<GpuData<T>*>(A._info); // cast from void to GpuData
  GpuData<T> *infoy_A  = reinterpret_cast<GpuData<T>*>(A._infoy); // cast from void to GpuData
  GpuData<T> *vinfo_A  = reinterpret_cast<GpuData<T>*>(A._vinfo); // cast from void to GpuData
  GpuData<T> *vinfoy_A = reinterpret_cast<GpuData<T>*>(A._vinfoy); // cast from void to GpuData
  GpuData<T> *weightinfo_A = reinterpret_cast<GpuData<T>*>(A._weightinfo); // cast from void to GpuData

  GpuData<T> *info;
  GpuData<T> *infoy;
  GpuData<T> *vinfo;
  GpuData<T> *vinfoy;
  GpuData<T> *weightinfo;
  if(info_A->orig_data) info = new GpuData<T>(info_A->orig_data); // create new GpuData structure with point to CPU data
  else info = new GpuData<T>(0); // create new GpuData structure with point to CPU data
  if(infoy_A->orig_data) infoy  = new GpuData<T>(infoy_A->orig_data); // create new GpuData structure with point to CPU data
  else infoy = new GpuData<T>(0); // create new GpuData structure with point to CPU data
  if(vinfo_A->orig_data) vinfo  = new GpuData<T>(vinfo_A->orig_data); // create new GpuData structure with point to CPU data
  else vinfo = new GpuData<T>(0); // create new GpuData structure with point to CPU data
  if(vinfoy_A->orig_data) vinfoy = new GpuData<T>(vinfoy_A->orig_data); // create new GpuData structure with point to CPU data
  else vinfoy = new GpuData<T>(0); // create new GpuData structure with point to CPU data
  if(weightinfo_A->orig_data) weightinfo = new GpuData<T>(weightinfo_A->orig_data); // create new GpuData structure with point to CPU data
  else weightinfo = new GpuData<T>(0); // create new GpuData structure with point to CPU data


  this->_info = reinterpret_cast<void*>(info); // back to cast as void
  this->_infoy = reinterpret_cast<void*>(infoy); // back to cast as void
  this->_vinfo = reinterpret_cast<void*>(vinfo); // back to cast as void
  this->_vinfoy = reinterpret_cast<void*>(vinfoy); // back to cast as void
  this->_weightinfo = reinterpret_cast<void*>(weightinfo); // back to cast as void
  POP_RANGE("MDnew",MDnew,2);


  if(!this->_done_alloc){
    this->_done_alloc = true;
    if(A._wDev == _wDev && A._me == _me && (A._sharedA==0 || _sharedA==0)){ // if on same device and same thread, just copy pointer
      DEBUG_FPRINTF(stderr,"ATYPE%d\n",0);
      _data   = A._data;
      _datay  = A._datay;
      _vdata  = A._vdata;
      _vdatay = A._vdatay;
      _weight = A._weight;
      _de = A._de;
      _dopredict = A._dopredict;
      //      Init();
      //      this->_done_equil=1;
    }
    else if(A._wDev == _wDev && A._sharedA!=0 && _sharedA!=0){ // if on same device and sharing memory, then just copy pointer
      DEBUG_FPRINTF(stderr,"ATYPE%d\n",1);
      _data   = A._data;
      _datay  = A._datay;
      _vdata  = A._vdata;
      _vdatay = A._vdatay;
      _weight = A._weight;
      _de = A._de;
      _dopredict = A._dopredict;
      Init();
      this->_done_equil=1;
    }
    else{
      DEBUG_FPRINTF(stderr,"ATYPE%d\n",2);
      // Copy Matrix to from source GPU to this GPU
      PUSH_RANGE("MDcopy",MDcopy,1);
      //GpuData<T> *info = reinterpret_cast<GpuData<T>*>(_info); // cast void -> GpuData
      double t0 = timer<double>();
      if(A._data) cudaMalloc(&_data, A._m * A._n * sizeof(T)); // allocate on GPU
      if(A._datay) cudaMalloc(&_datay, A._m * sizeof(T)); // allocate on GPU
      if(A._vdata) cudaMalloc(&_vdata, A._mvalid * A._n * sizeof(T)); // allocate on GPU
      if(A._vdatay) cudaMalloc(&_vdatay, A._mvalid * sizeof(T)); // allocate on GPU
      if(A._weight) cudaMalloc(&_weight, A._m * sizeof(T)); // allocate on GPU
      double t1 = timer<double>();
      if(A._data) cudaMemcpyPeer(_data, _wDev, A._data, A._wDev, A._m * A._n * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
      if(A._datay){
        cudaMemcpyPeer(_datay, _wDev, A._datay, A._wDev, A._m * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
        _dopredict=0;
      }
      else{
        _dopredict=1;
      }
      if(A._vdata) cudaMemcpyPeer(_vdata, _wDev, A._vdata, A._wDev, A._mvalid * A._n * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
      if(A._vdatay) cudaMemcpyPeer(_vdatay, _wDev, A._vdatay, A._wDev, A._mvalid * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
      if(A._weight) cudaMemcpyPeer(_weight, _wDev, A._weight, A._wDev, A._m * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
      if(A._de) cudaMalloc(&_de, (A._m + A._n) * sizeof(T)); cudaMemcpyPeer(_de, _wDev, A._de, A._wDev, (A._m + A._n) * sizeof(T));
      if(sharedA>0){
        Init();
        Equil(1);
      }
      double t2 = timer<double>();
#ifdef DEBUG
      printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
      printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
#endif
      POP_RANGE("MDcopy",MDcopy,1);
    }
  }

}

template <typename T>
MatrixDense<T>::MatrixDense(int me, int wDev, const MatrixDense<T>& A)
  : MatrixDense<T>(0, me, wDev, A){} // then assume not sharing memory

template <typename T>
MatrixDense<T>::MatrixDense(int wDev, const MatrixDense<T>& A)
  : MatrixDense<T>(wDev, wDev, A){} // then assume thread=wDev for the new matrix (i.e. not input A)


template <typename T>
MatrixDense<T>::MatrixDense(const MatrixDense<T>& A)
  : MatrixDense<T>(A._wDev, A){} // then assume same device as input A

template <typename T>
MatrixDense<T>::~MatrixDense() {

  
  // return;//TODO: Some deconstructor issue FIXME.  Segfaults after adding weights.  Can't find issue.
  
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  if(0){
    GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
    GpuData<T> *infoy = reinterpret_cast<GpuData<T>*>(this->_infoy);
    GpuData<T> *vinfo = reinterpret_cast<GpuData<T>*>(this->_vinfo);
    GpuData<T> *vinfoy = reinterpret_cast<GpuData<T>*>(this->_vinfoy);
    GpuData<T> *weightinfo = reinterpret_cast<GpuData<T>*>(this->_weightinfo);

    if(info) delete info; this->_info = 0;
    if(infoy) delete infoy; this->_infoy = 0;
    if(vinfo) delete vinfo; this->_vinfo = 0;
    if(vinfoy) delete vinfoy; this->_vinfoy = 0;
    if(weightinfo) delete weightinfo; this->_weightinfo = 0;
  }

  //  fprintf(stderr,"HERE1\n"); fflush(stderr);

  if(0){ // Note that this frees these pointers as soon as MatrixDense constructor goes out of scope, and might want more fine-grained control over GPU memory if inside (say) high-level python API
    // If 0 is used, then need to ensure user calls a finish() or something to free memory.  If 0, also allows user to call (say) fit() or fitptr() multiple times
    
    if (this->_done_init && _data) {
      //      fprintf(stderr,"Freeing _data: %p\n",(void*)_data); fflush(stderr);
      cudaFree(_data);
      this->_data = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
    //  fprintf(stderr,"HERE2\n"); fflush(stderr);
    if (this->_done_init && _datay) {
      //      fprintf(stderr,"Freeing _datay: %p\n",(void*)_datay); fflush(stderr);
      cudaFree(_datay);
      this->_datay = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
    //  fprintf(stderr,"HERE3\n"); fflush(stderr);
    if (this->_done_init && _vdata) {
      //      fprintf(stderr,"Freeing _vdata: %p\n",(void*)_vdata); fflush(stderr);
      cudaFree(_vdata);
      this->_vdata = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
    //  fprintf(stderr,"HERE4\n"); fflush(stderr);
    if (this->_done_init && _vdatay) {
      //      fprintf(stderr,"Freeing _vdatay: %p\n",(void*)_vdatay); fflush(stderr);
      cudaFree(_vdatay);
      this->_vdatay = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
    //  fprintf(stderr,"HERE5\n"); fflush(stderr);

    if (this->_done_init && _weight) {
      //      fprintf(stderr,"Freeing _weight: %p\n",(void*)_weight); fflush(stderr);
      cudaFree(_weight);
      this->_weight = 0;
      DEBUG_CUDA_CHECK_ERR();
    }
    //  fprintf(stderr,"HERE6\n"); fflush(stderr);

    if(this->_done_init && _de && !_sharedA){ // JONTODO: When sharedA=1, only free on sourceme thread and sourcewDev device (can store sourcethread for-- sourceme -- data and only free if on source thread)
      //      fprintf(stderr,"Freeing _de: %p\n",(void*)_weight); fflush(stderr);
      cudaFree(_de);
      this->_de=0;
      DEBUG_CUDA_CHECK_ERR();
    }
  }
  
}

      
template <typename T>
int MatrixDense<T>::Init() {
  DEBUG_EXPECT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;
  CUDACHECK(cudaSetDevice(_wDev));

  PUSH_RANGE("MDinit",MDinit,1);
  POP_RANGE("MDinit",MDinit,1);

  DEBUG_CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixDense<T>::GetTrainX(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(_data){
    if(datatype==1){
      cudaMemcpy(*data, _data, size* sizeof(T),cudaMemcpyDeviceToHost);
      CUDA_CHECK_ERR();
    }
    else{
      std::memcpy(*data, _data, size * sizeof(T));
    }
    return(0);
  }
  else return(1);
}
template <typename T>
int MatrixDense<T>::GetTrainY(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(_datay){
    if(datatype==1){
      cudaMemcpy(*data, _datay, size* sizeof(T),cudaMemcpyDeviceToHost);
      CUDA_CHECK_ERR();
    }
    else{
      std::memcpy(*data, _datay, size * sizeof(T));
    }
    return(0);
  }
  else return(1);
}

template <typename T>
int MatrixDense<T>::GetValidX(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(_vdata){
    if(datatype==1){
      cudaMemcpy(*data, _vdata, size* sizeof(T),cudaMemcpyDeviceToHost);
      CUDA_CHECK_ERR();
    }
    else{
      std::memcpy(*data, _vdata, size * sizeof(T));
    }
    return(0);
  }
  else return(1);
}
template <typename T>
int MatrixDense<T>::GetValidY(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(_vdatay){
    if(datatype==1){
      cudaMemcpy(*data, _vdatay, size* sizeof(T),cudaMemcpyDeviceToHost);
      CUDA_CHECK_ERR();
    }
    else{
      std::memcpy(*data, _vdatay, size * sizeof(T));
    }
    return(0);
  }
  else return(1);
}
template <typename T>
int MatrixDense<T>::GetWeight(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(_weight){
    if(datatype==1){
      cudaMemcpy(*data, _weight, size* sizeof(T),cudaMemcpyDeviceToHost);
      CUDA_CHECK_ERR();
    }
    else{
      std::memcpy(*data, _weight, size * sizeof(T));
    }
    return(0);
  }
  else return(1);
}


template <typename T>
int MatrixDense<T>::Mul(char trans, T alpha, const T *x, T beta, T *y) const {

  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init)
    return 1;
  CUDACHECK(cudaSetDevice(_wDev));

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, this->_n);
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, this->_m);

  //  Performs the matrix-vector operations y := alpha*A*x + beta*y or y := alpha*A'*x + beta*y where alpha and beta are scalars, x and y are vectors and A is an m by n matrix
  // _data is A on GPU
  //https://docs.oracle.com/cd/B19306_01/appdev.102/b14258/u_nla.htm#CIAFEAFG
  if (_ord == ROW) {
    cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>(_data, this->_m, this->_n);
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta,
        &y_vec);
  } else {
    cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>(_data, this->_m, this->_n);
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }
  CUDA_CHECK_ERR();

  return 0;
}

  template <typename T>
int MatrixDense<T>::Mulvalid(char trans, T alpha, const T *x, T beta, T *y) const {

  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init)
    return 1;
  CUDACHECK(cudaSetDevice(_wDev));

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, this->_n);
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, this->_mvalid);

  //  Performs the matrix-vector operations y := alpha*A*x + beta*y or y := alpha*A'*x + beta*y where alpha and beta are scalars, x and y are vectors and A is an m by n matrix
  // _vdata is A on GPU
  //https://docs.oracle.com/cd/B19306_01/appdev.102/b14258/u_nla.htm#CIAFEAFG
  if (_ord == ROW) {
    cml::matrix<T, CblasRowMajor> A =
        cml::matrix_view_array<T, CblasRowMajor>(_vdata, this->_mvalid, this->_n);
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta,
        &y_vec);
  } else {
    cml::matrix<T, CblasColMajor> A =
        cml::matrix_view_array<T, CblasColMajor>(_vdata, this->_mvalid, this->_n);
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }
  CUDA_CHECK_ERR();

  return 0;
}

  // col-major order (fortran) A, but still print as row major
template <typename T>
void printMatrix(int m, int n, const T*A, int lda, const char* name)
{
  printf("rows=%d cols=%d lda=%d\n",m,n,lda);
  for(int row = 0 ; row < m ; row++){
    for(int col = 0 ; col < n ; col++){
      T Areg = A[row + col*lda];
      printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
    }
  }
}

  // row-major order (c) A printed as row major
template <typename T>
void printMatrix2(int m, int n, const T*A, int lda, const char* name)
{
  printf("rows=%d cols=%d lda=%d\n",m,n,lda);
  for(int row = 0 ; row < m ; row++){
    for(int col = 0 ; col < n ; col++){
      T Areg = A[col + row*n];
      printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
    }
  }
}

/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include svd_example.cpp 
 *   g++ -fopenmp -o a.out svd_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *
 */

  inline cusolverStatus_t cusolverDngesvd (    cusolverDnHandle_t handle,    signed char jobu,    signed char jobvt,    int m,    int n,    float *A,    int lda,    float *S,    float *U,    int ldu,    float *VT,    int ldvt,    float *work,    int lwork,    float *rwork,    int *devInfo){
  return(cusolverDnSgesvd(handle,    jobu,    jobvt,    m,    n,    A,    lda,    S,    U,    ldu,    VT,    ldvt,    work,    lwork,    rwork,    devInfo));
}
  
inline cusolverStatus_t cusolverDngesvd (    cusolverDnHandle_t handle,    signed char jobu,    signed char jobvt,    int m,    int n,    double *A,    int lda,    double *S,    double *U,    int ldu,    double *VT,    int ldvt,    double *work,    int lwork,    double *rwork,    int *devInfo){
  return(cusolverDnDgesvd(handle,    jobu,    jobvt,    m,    n,    A,    lda,    S,    U,    ldu,    VT,    ldvt,    work,    lwork,    rwork,    devInfo));
}

inline cublasStatus_t cublasgemm(cublasHandle_t handle,                           cublasOperation_t transa, cublasOperation_t transb,                           int m, int n, int k,                           const float           *alpha,                           const float           *A, int lda,                           const float           *B, int ldb,                           const float           *beta,                           float           *C, int ldc){
  return(cublasSgemm_v2(handle,                           transa, transb,                           m, n, k,                   alpha,                                A, lda,                           B, ldb,                      beta,                       C, ldc));
  

}

  
inline cublasStatus_t cublasgemm(cublasHandle_t handle,                           cublasOperation_t transa, cublasOperation_t transb,                           int m, int n, int k,                           const double          *alpha,                           const double          *A, int lda,                           const double          *B, int ldb,                           const double          *beta,                           double          *C, int ldc){
  return(cublasDgemm_v2(handle,                           transa, transb,                           m, n, k,                   alpha,                                A, lda,                           B, ldb,                      beta,                       C, ldc));

}

inline cublasStatus_t cublasdgmm(cublasHandle_t handle,
                                 cublasSideMode_t mode, 
                                 int m, 
                                 int n,
                                 const float *A, 
                                 int lda,
                                 const float *x, 
                                 int incx,
                                 float *C, 
                                 int ldc){
  
  return(cublasSdgmm(handle,
                     mode, 
                     m, 
                     n,
                     A, 
                     lda,
                     x, 
                     incx,
                     C, 
                     ldc));

}
inline cublasStatus_t cublasdgmm(cublasHandle_t handle,
                                 cublasSideMode_t mode, 
                                 int m, 
                                 int n,
                                 const double *A, 
                                 int lda,
                                 const double *x, 
                                 int incx,
                                 double *C, 
                                 int ldc){
  
  return(cublasDdgmm(handle,
                     mode, 
                     m, 
                     n,
                     A, 
                     lda,
                     x, 
                     incx,
                     C, 
                     ldc));

}

inline cublasStatus_t cublasnrm2(cublasHandle_t handle, 
                                 int n, 
                                 const double *x, 
                                 int incx, 
                                 double *result){
    return(cublasDnrm2_v2(handle, 
                      n, 
                      x, 
                      incx, 
                      result));
    
  }
    
inline cublasStatus_t cublasnrm2(cublasHandle_t handle, 
                                 int n, 
                                 const float *x, 
                                 int incx, 
                                 float *result){
    return(cublasSnrm2_v2(handle, 
                      n, 
                      x, 
                      incx, 
                      result));
    
}
    
// // Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// // using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// // TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TILE_DIM    16
#define BLOCK_ROWS  16

//   __global__ void transposeNaive(float *odata, float* idata,
//                                  int width, int height)
//   {
//     int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
//     int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
//     int index_in = xIndex + width * yIndex;
//     int index_out = yIndex + height * xIndex;
//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//       odata[index_out+i] = idata[index_in+i*width];
//     }
//   }
//   __global__ void transposeNaive(double *odata, double* idata,
//                                  int width, int height)
//   {
//     int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
//     int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
//     int index_in = xIndex + width * yIndex;
//     int index_out = yIndex + height * xIndex;
//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//       odata[index_out+i] = idata[index_in+i*width];
//     }
//   } 
//   __global__ void transposeCoalesced(float *odata,
//                                      float *idata, int width, int height)
//   {
//     __shared__ float tile[TILE_DIM][TILE_DIM];
//     int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
//     int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
//     int index_in = xIndex + (yIndex)*width;
//     xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//     yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//     int index_out = xIndex + (yIndex)*height;
//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//       tile[threadIdx.y+i][threadIdx.x] =
//         idata[index_in+i*width];
//     }

//     __syncthreads();

//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//       odata[index_out+i*height] =
//         tile[threadIdx.x][threadIdx.y+i];
//     }
//   } 
//   __global__ void transposeCoalesced(double *odata,
//                                      double *idata, int width, int height)
//   {
//     __shared__ double tile[TILE_DIM][TILE_DIM];
//     int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
//     int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
//     int index_in = xIndex + (yIndex)*width;
//     xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//     yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//     int index_out = xIndex + (yIndex)*height;
//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//       tile[threadIdx.y+i][threadIdx.x] =
//         idata[index_in+i*width];
//     }

//     __syncthreads();

//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//       odata[index_out+i*height] =
//         tile[threadIdx.x][threadIdx.y+i];
//     }
//   }

// in-place transpose for row-major matrix on device of A[m][n]
void cudaintranspose(float *odata, float *idata, int m, int n){

  cudaError_t cudaStat1 = cudaSuccess;
  cudaStat1 = cudaMemcpy(odata, idata, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == cudaStat1);
  
  float const alpha(1.0);
  float const beta(0.0);
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, idata, n, &beta, idata, m, odata, m );
  cublasDestroy(handle);
}
void cudaintranspose(double *odata, double *idata, int m, int n){

  cudaError_t cudaStat1 = cudaSuccess;
  cudaStat1 = cudaMemcpy(odata, idata, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == cudaStat1);
  
  double const alpha(1.0);
  double const beta(0.0);
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, idata, n, &beta, idata, m, odata, m );
  cublasDestroy(handle);
}
#define MIN(a,b) ((a)<(b) ? (a) : (b))

template <typename T>
int MatrixDense<T>::svd1(void) {
  fprintf(stderr,"begin svd inside0\n"); fflush(stderr); fflush(stdout);
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    Init();

  fprintf(stderr,"begin svd inside\n"); fflush(stderr); fflush(stdout);
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;
  cudaError_t cudaStat6 = cudaSuccess;
  int m = this->_m;
  int n = this->_n;
  //    const int m = this->_m;
  //    const int n = this->_n;
  int lda = m;
  /*       | 1 2  |
   *   A = | 4 5  |
   *       | 2 1  |
   */


    
  unsigned char ord='r'; // TODO; should be inputted
      
  // original device vector
  T *d_A0;
  d_A0 = this->_data;

  // device vectors
  T *d_A = NULL;
  T *d_S = NULL;
  T *d_U = NULL;
  T *d_VT = NULL;
  int *devInfo = NULL;
  T *d_work = NULL;
  T *d_rwork = NULL;
  T *d_W = NULL;  // W = S*VT

  int lwork = 0;
  int info_gpu = 0;
  const T h_one = 1;
  const T h_minus_one = -1;

  double t0 = timer<double>();
  // step 1: create cusolverDn/cublas handle
  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);


  fprintf(stderr,"HERE1\n"); fflush(stderr); fflush(stdout);

  // step 2: copy A to device
  //    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(T)*lda*n);

  // svd destroys d_A, so make copy for testing error // OPTMARK
  cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(T)*lda*n);
  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMemcpy(d_A, d_A0, sizeof(T)*lda*n, cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStat1);


  int ldu=m; //lda;
  int ldureal=n; // actual storage
  int ldvt=n;
  if(ord=='r'){
    // transpose
    // execution configuration parameters
    //dim3 grid(n/TILE_DIM, lda/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
    //      transposeCoalesced<<<grid, threads>>>(d_A, d_A0, n, lda);
    //    transposeNaive<<<grid, threads>>>(d_A, d_A0, n, lda);
    cudaintranspose(d_A,d_A0,m,n); // OPTMARK
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    // below debug only for printMatrix2 to view, shouldn't actually swap for use.
    if(0){
      int temp=m;
      m=n;
      n=temp;
      
      lda=m;
      ldu=m; //lda;
      ldureal=n; // actual storage
      ldvt=n;
    }
  }
  else{
    d_A = d_A0;
  }



  
  fprintf(stderr,"HERE PRE\n"); fflush(stderr); fflush(stdout);
  // old host side vectors
  //    T A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
  //    GpuData<T> *info_A   = reinterpret_cast<GpuData<T>*>(this->_info); // cast from void to GpuData
  //    T *A = const_cast<T*>(info_A->orig_data);
#if(0)
  T A[lda*n]; // for debug
  T U[ldureal*m]; // m-by-m unitary matrix 
  T VT[ldvt*n];  // n-by-n unitary matrix
  T S[MIN(n,m)]; // singular value
#endif
  //    T S_exact[n] = {7.065283497082729, 1.040081297712078};
  fprintf(stderr,"HERE POST\n"); fflush(stderr); fflush(stdout);
  
  // now d_A has column-major order matrix
  fprintf(stderr,"HERE2\n"); fflush(stderr); fflush(stdout);
    
#if(0) // debug
    cudaStat1 = cudaMemcpy(A, d_A, sizeof(T)*lda*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    printf("A = (matlab base-1)\n");
    printMatrix(m, n, A, lda, "A");
    printf("=====\n");
    printf("A = (matlab base-1)\n");
    printMatrix2(m, n, A, lda, "A");
    printf("=====\n");
#endif

  fprintf(stderr,"HERE3\n"); fflush(stderr); fflush(stdout);

  cudaStat2 = cudaMalloc ((void**)&d_S  , sizeof(T)*MIN(n,m));
  cudaStat3 = cudaMalloc ((void**)&d_U  , sizeof(T)*ldureal*m);
  cudaStat4 = cudaMalloc ((void**)&d_VT , sizeof(T)*ldvt*n);
  cudaStat5 = cudaMalloc ((void**)&devInfo, sizeof(int));
  cudaStat6 = cudaMalloc ((void**)&d_W  , sizeof(T)*lda*n);
  //    assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat5);
  assert(cudaSuccess == cudaStat6);

  // host->device
  //    cudaStat1 = cudaMemcpy(d_A, A, sizeof(T)*lda*n, cudaMemcpyHostToDevice);
  //    assert(cudaSuccess == cudaStat1);

  // step 3: query working space of SVD
  //The dense matrices are assumed to be stored in column-major order in memory.
  cusolver_status = cusolverDnDgesvd_bufferSize(
                                                cusolverH,
                                                m,
                                                n,
                                                &lwork );
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

  cudaStat1 = cudaMalloc((void**)&d_work , sizeof(T)*lwork);
  assert(cudaSuccess == cudaStat1);
  double t1 = timer<double>();
  fprintf(stderr,"SVD init: %g\n",t1-t0); fflush(stderr); fflush(stdout);

  // step 4: compute SVD
  double t0c = timer<double>();
  signed char jobu = 'A'; // all m columns of U
  signed char jobvt = 'A'; // all n columns of VT
  cusolver_status = cusolverDngesvd(
                                    cusolverH,
                                    jobu,
                                    jobvt,
                                    m,
                                    n,
                                    d_A,
                                    lda,
                                    d_S,
                                    d_U,
                                    ldu,
                                    d_VT,
                                    ldvt,
                                    d_work,
                                    lwork,
                                    d_rwork,
                                    devInfo);


  cudaStat4 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("after gesvd: info_gpu = %d\n", info_gpu); fflush(stdout);
  assert(0 == info_gpu);
  printf("=====\n"); fflush(stdout);

  cudaStat1 = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStat1);
  fprintf(stderr,"BAD: %d\n",cusolver_status); fflush(stderr);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  double t1c = timer<double>();
  fprintf(stderr,"SVD compute: %g\n",t1-t0); fflush(stderr); fflush(stdout);



#if(0)
    /////////////////////////
    // Copy solution device->host
    double t0h = timer<double>();
    cudaStat1 = cudaMemcpy(U , d_U , sizeof(T)*ldureal*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(VT, d_VT, sizeof(T)*ldvt*n, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(S , d_S , sizeof(T)*MIN(n,m), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
  

  if(0){ // debug
    printf("S = (matlab base-1)\n");
    printMatrix(n, 1, S, lda, "S");
    printf("=====\n");
      
    printf("U = (matlab base-1)\n");
    printMatrix(m, m, U, ldureal, "U");
    printf("=====\n");
      
    printf("VT = (matlab base-1)\n");
    printMatrix(n, n, VT, ldvt, "VT");
    printf("=====\n");

    /////////////////////////
    // measure error of singular value
    //      T ds_sup = 0;
    //      for(int j = 0; j < n; j++){
    //        T err = fabs( S[j] - S_exact[j] );
    //        ds_sup = (ds_sup > err)? ds_sup : err;
    //      }
    //      printf("|S - S_exact| = %E \n", ds_sup);
  }
  double t1h = timer<double>();
  fprintf(stderr,"SVD back to host: %g\n",t1h-t0h); fflush(stderr); fflush(stdout);
#endif

  /////////////////////////
  // now check
  double t0c1 = timer<double>();
  // step 5: |A - U*S*VT|
  // W = S*VT
  cublas_status = cublasdgmm(
                             cublasH,
                             CUBLAS_SIDE_LEFT,
                             n,
                             n,
                             d_VT,
                             ldvt,
                             d_S,
                             1,
                             d_W,
                             lda);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);
  double t1c1 = timer<double>();
  fprintf(stderr,"SVD check1: %g\n",t1c1-t0c1); fflush(stderr); fflush(stdout);


 
  // A := -U*W + A
  double t0c2 = timer<double>();
  cudaStat1 = cudaMemcpy(d_A, d_A0, sizeof(T)*lda*n, cudaMemcpyDeviceToDevice); // copy because original d_A was destroyed
  assert(cudaSuccess == cudaStat1);
  cublas_status = cublasgemm(
                             cublasH,
                             CUBLAS_OP_N, // U
                             CUBLAS_OP_N, // W
                             m, // number of rows of A
                             n, // number of columns of A
                             n, // number of columns of U 
                             &h_minus_one, /* host pointer */
                             d_U, // U
                             ldu,
                             d_W, // W
                             lda,
                             &h_one, /* hostpointer */
                             d_A,
                             lda);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);
  double t1c2 = timer<double>();
  fprintf(stderr,"SVD check2: %g\n",t1c2-t0c2); fflush(stderr); fflush(stdout);

  double t0c3 = timer<double>();
  T dR_fro = 0.0;
  cublas_status = cublasnrm2(
                             cublasH, lda*n, d_A, 1, &dR_fro);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  printf("|A - U*S*VT| = %E \n", dR_fro); fflush(stdout);
  double t1c3 = timer<double>();
  fprintf(stderr,"SVD check3: %g\n",t1c3-t0c3); fflush(stderr); fflush(stdout);

    
  // free resources
  double t0f = timer<double>();
  //if (d_A    ) cudaFree(d_A);
  if (d_S    ) cudaFree(d_S);
  if (d_U    ) cudaFree(d_U);
  if (d_VT   ) cudaFree(d_VT);
  if (devInfo) cudaFree(devInfo);
  if (d_work ) cudaFree(d_work);
  if (d_rwork) cudaFree(d_rwork);
  if (d_W    ) cudaFree(d_W);

  if (cublasH ) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);
  //    cudaDeviceReset();
  double t1f = timer<double>();
  fprintf(stderr,"SVD free: %g\n",t1f-t0f); fflush(stderr); fflush(stdout);

  fprintf(stderr,"end svd inside\n"); fflush(stderr); fflush(stdout);

  return 0;
}

// Equilibration (precondition) matrix using Sinkhorn Knopp method wrapped to allow any norm
// See https://arxiv.org/pdf/1610.03871.pdf for more information
template <typename T>
int MatrixDense<T>::Equil(bool equillocal) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  if (this->_done_equil) return 0;
  else this->_done_equil=1;
  
  CUDACHECK(cudaSetDevice(_wDev));

  // Extract cublas handle from _info.
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  T *d = _de;
  T *e = d + this->_m;

  
  // Number of elements in matrix.
  size_t num_el = this->_m * this->_n;

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign;
  size_t num_sign_bytes = (num_el + 7) / 8;
  cudaMalloc(&sign, num_sign_bytes);
  CUDA_CHECK_ERR();

  size_t num_chars = num_el / 8;
  size_t grid_size = cml::calc_grid_dim(num_chars, cml::kBlockSize);
  if(equillocal){
    // Fill sign bits, assigning each thread a multiple of 8 elements.
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      __SetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
                                                SquareF<T>());
    } else {
      __SetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
                                                AbsF<T>());
    }
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    // If numel(A) is not a multiple of 8, then we need to set the last couple
    // of sign bits too. 
    if (num_el > num_chars * 8) {
      if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
        __SetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
                                  num_el - num_chars * 8, SquareF<T>());
      } else {
        __SetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
                                  num_el - num_chars * 8, AbsF<T>());
      }
      wrapcudaDeviceSynchronize();
      CUDA_CHECK_ERR();
    }
  }
  
  // Perform Sinkhorn-Knopp equilibration to obtain a doubly stochastic matrix.
  SinkhornKnopp(this, d, e, equillocal);
  wrapcudaDeviceSynchronize();

  if(equillocal){
    // Transform A = sign(A) .* sqrt(A) if 2-norm equilibration was performed,
    // or A = sign(A) .* A if the 1-norm was equilibrated.
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      __UnSetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
                                                  SqrtF<T>());
    } else {
      __UnSetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
                                                  IdentityF<T>());
    }
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    // Deal with last few entries if num_el is not a multiple of 8.
    if (num_el > num_chars * 8) {
      if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
        __UnSetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
                                    num_el - num_chars * 8, SqrtF<T>());
      } else {
        __UnSetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars, 
                                    num_el - num_chars * 8, IdentityF<T>());
      }
      wrapcudaDeviceSynchronize();
      CUDA_CHECK_ERR();
    }
  }
  
  // Compute D := sqrt(D), E := sqrt(E), if 2-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    thrust::transform(thrust::device_pointer_cast(d),
        thrust::device_pointer_cast(d + this->_m),
        thrust::device_pointer_cast(d), SqrtF<T>());
    thrust::transform(thrust::device_pointer_cast(e),
        thrust::device_pointer_cast(e + this->_n),
        thrust::device_pointer_cast(e), SqrtF<T>());
    wrapcudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Compute A := D * A * E.
  MultDiag(d, e, this->_m, this->_n, _ord, _data);
  wrapcudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // Scale A to have norm of 1 (in the kNormNormalize norm).
  T normA = NormEst(hdl, kNormNormalize, *this);
  CUDA_CHECK_ERR();
  wrapcudaDeviceSynchronize();
  cml::vector<T> a_vec = cml::vector_view_array(_data, num_el);
  cml::vector_scale(&a_vec, 1 / normA);
  wrapcudaDeviceSynchronize();

  // Scale d and e to account for normalization of A.
  cml::vector<T> d_vec = cml::vector_view_array<T>(d, this->_m);
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, this->_n);
  cml::vector_scale(&d_vec, 1 / sqrt(normA));
  cml::vector_scale(&e_vec, 1 / sqrt(normA));
  wrapcudaDeviceSynchronize();

  DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA,
      cml::blas_nrm2(hdl, &d_vec), cml::blas_nrm2(hdl, &e_vec));

  cudaFree(sign);
  CUDA_CHECK_ERR();

  return 0;
}

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example
// structure used to accumulate the moments and other
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
  T n;
  T min;
  T max;
  T mean;
  T M2;
  T M3;
  T M4;

  // initialize to the identity element
  void initialize()
  {
    n = mean = M2 = M3 = M4 = 0;
    min = std::numeric_limits<T>::max();
    max = std::numeric_limits<T>::min();
  }

  T variance()   { return M2 / (n - 1); }
  T variance_n() { return M2 / n; }
  T skewness()   { return std::sqrt(n) * M3 / std::pow(M2, (T) 1.5); }
  T kurtosis()   { return n * M4 / (M2 * M2); }
};

  // stats_unary_op is a functor that takes in a value x and
  // returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
  __host__ __device__
  summary_stats_data<T> operator()(const T& x) const
  {
    summary_stats_data<T> result;
    result.n    = 1;
    result.min  = x;
    result.max  = x;
    result.mean = x;
    result.M2   = 0;
    result.M3   = 0;
    result.M4   = 0;

    return result;
  }
};

  // summary_stats_binary_op is a functor that accepts two summary_stats_data
  // structs and returns a new summary_stats_data which are an
  // approximation to the summary_stats for
  // all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op
  : public thrust::binary_function<const summary_stats_data<T>&,
                                   const summary_stats_data<T>&,
                                   summary_stats_data<T> >
{
  __host__ __device__
  summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
  {
    summary_stats_data<T> result;

    // precompute some common subexpressions
    T n  = x.n + y.n;
    T n2 = n  * n;
    T n3 = n2 * n;

    T delta  = y.mean - x.mean;
    T delta2 = delta  * delta;
    T delta3 = delta2 * delta;
    T delta4 = delta3 * delta;

    //Basic number of samples (n), min, and max
    result.n   = n;
    result.min = thrust::min(x.min, y.min);
    result.max = thrust::max(x.max, y.max);

    result.mean = x.mean + delta * y.n / n;

    result.M2  = x.M2 + y.M2;
    result.M2 += delta2 * x.n * y.n / n;

    result.M3  = x.M3 + y.M3;
    result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
    result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

    result.M4  = x.M4 + y.M4;
    result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
    result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
    result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

    return result;
  }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
  typedef typename std::iterator_traits<Iterator>::value_type T;

  std::cout << name << ": ";
  thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
  std::cout << "\n";
}

template<typename T>
struct absolute_value : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};
// --- Operator for testing nan values
template<typename T>
struct isnan_test {
    __host__ __device__ bool operator()(const T a) const {
        return isnan(a) || isinf(a);
    }
};

// check properties of input data
template <typename T>
int MatrixDense<T>::Stats(int intercept, T *min, T *max, T *mean, T *var, T *sd, T *skew, T *kurt, T &lambda_max0)
{
  CUDACHECK(cudaSetDevice(_wDev));

  if(_data!=NULL) {// check for nan or inf in data
	  thrust::device_ptr<T> begin = thrust::device_pointer_cast(_data);
	  thrust::device_ptr<T> end = thrust::device_pointer_cast(_data+this->_m*this->_n);
	  bool h_result = thrust::transform_reduce(begin, end, isnan_test<T>(), 0, thrust::plus<bool>());
	  if(h_result==true){
		  fprintf(stderr,"Data matrix (trainX) has nan/inf or missing was not encoded\n");
	  	  fflush(stderr);
	  	  exit(1);
	  }
  }
  if(_datay!=NULL) {// check for nan or inf in data
	  thrust::device_ptr<T> begin = thrust::device_pointer_cast(_datay);
	  thrust::device_ptr<T> end = thrust::device_pointer_cast(_datay+this->_m);
	  bool h_result = thrust::transform_reduce(begin, end, isnan_test<T>(), 0, thrust::plus<bool>());
	  if(h_result==true){
		  fprintf(stderr,"Data training predictions/labels (trainY) has nan/inf or missing was not encoded\n");
	  	  fflush(stderr);
	  	  exit(1);
	  }
  }
  if(_vdata!=NULL) {// check for nan or inf in data
	  thrust::device_ptr<T> begin = thrust::device_pointer_cast(_vdata);
	  thrust::device_ptr<T> end = thrust::device_pointer_cast(_vdata+this->_mvalid*this->_n);
	  bool h_result = thrust::transform_reduce(begin, end, isnan_test<T>(), 0, thrust::plus<bool>());
	  if(h_result==true){
		  fprintf(stderr,"Validation Data matrix (validX) has nan/inf or missing was not encoded\n");
	  	  fflush(stderr);
	  	  exit(1);
	  }
  }
  if(_vdatay!=NULL) {// check for nan or inf in data
	  thrust::device_ptr<T> begin = thrust::device_pointer_cast(_vdatay);
	  thrust::device_ptr<T> end = thrust::device_pointer_cast(_vdatay+this->_mvalid);
	  bool h_result = thrust::transform_reduce(begin, end, isnan_test<T>(), 0, thrust::plus<bool>());
	  if(h_result==true){
		  fprintf(stderr,"Validation Data training predictions/labels (validY) has nan/inf or missing was not encoded\n");
	  	  fflush(stderr);
	  	  exit(1);
	  }
  }
  if(_weight!=NULL) {// check for nan or inf in data
	  thrust::device_ptr<T> begin = thrust::device_pointer_cast(_weight);
	  thrust::device_ptr<T> end = thrust::device_pointer_cast(_weight+this->_m);
	  bool h_result = thrust::transform_reduce(begin, end, isnan_test<T>(), 0, thrust::plus<bool>());
	  if(h_result==true){
		  fprintf(stderr,"Weight Training Data has nan/inf or missing was not encoded\n");
	  	  fflush(stderr);
	  	  exit(1);
	  }
  }

  // nothing else to do if _datay==NULL
  if(_datay==NULL) return(0);

  // setup arguments
  summary_stats_unary_op<T>  unary_op;
  summary_stats_binary_op<T> binary_op;
  summary_stats_data<T>      init;
  
  init.initialize();
  int len=0;
  
  // cast GPU pointer as thrust pointer
  thrust::device_ptr<T> dataybegin=thrust::device_pointer_cast(_datay);
  len=this->_m;
  thrust::device_ptr<T> datayend=thrust::device_pointer_cast(_datay+len);
  

  // compute summary statistics
  summary_stats_data<T> resulty = thrust::transform_reduce(dataybegin, datayend, unary_op, init, binary_op);

  min[0]=resulty.min;
  max[0]=resulty.max;
  mean[0]=resulty.mean;
  var[0]=resulty.variance();
  sd[0]=std::sqrt(resulty.variance_n());
  skew[0]=resulty.skewness();
  kurt[0]=resulty.kurtosis();

#ifdef DEBUG
  std::cout <<"******Summary Statistics of Response Train*****"<<std::endl;
//  print_range("The data", dataybegin, datayend);
  std::cout <<"Count              : "<< resulty.n << std::endl;
  std::cout <<"Minimum            : "<< min[0]<<std::endl;
  std::cout <<"Maximum            : "<< max[0]<<std::endl;
  std::cout <<"Mean               : "<< mean[0]<< std::endl;
  std::cout <<"Variance           : "<< var[0]<< std::endl;
  std::cout <<"Standard Deviation : "<< sd[0]<< std::endl;
  std::cout <<"Skewness           : "<< skew[0]<< std::endl;
  std::cout <<"Kurtosis           : "<< kurt[0]<< std::endl;
#endif

  // cast GPU pointer as thrust pointer
  thrust::device_ptr<T> vdataybegin=thrust::device_pointer_cast(_vdatay);
  len=this->_mvalid;
  thrust::device_ptr<T> vdatayend=thrust::device_pointer_cast(_vdatay+len);

  // compute summary statistics
  summary_stats_data<T> vresulty = thrust::transform_reduce(vdataybegin, vdatayend, unary_op, init, binary_op);

  
  min[1]=vresulty.min;
  max[1]=vresulty.max;
  mean[1]=vresulty.mean;
  var[1]=vresulty.variance();
  sd[1]=std::sqrt(vresulty.variance_n());
  skew[1]=vresulty.skewness();
  kurt[1]=vresulty.kurtosis();

#ifdef DEBUG
  std::cout <<"******Summary Statistics of Response Valid*****"<<std::endl;
  //  print_range("The data", vdataybegin, vdatayend);
  std::cout <<"Count              : "<< vresulty.n << std::endl;
  std::cout <<"Minimum            : "<< min[1]<<std::endl;
  std::cout <<"Maximum            : "<< max[1]<<std::endl;
  std::cout <<"Mean               : "<< mean[1]<< std::endl;
  std::cout <<"Variance           : "<< var[1]<< std::endl;
  std::cout <<"Standard Deviation : "<< sd[1]<< std::endl;
  std::cout <<"Skewness           : "<< skew[1]<< std::endl;
  std::cout <<"Kurtosis           : "<< kurt[1]<< std::endl;
#endif

  if(1){ // normal usage
    // Get Cublas handle
    GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
    cublasHandle_t hdl = info->handle;

    // Set up views for raw vectors.
    cml::vector<T> y_vec = cml::vector_view_array(_datay, this->_m); // b
    cml::vector<T> weight_vec;
    if(_weight) weight_vec = cml::vector_view_array(_weight, this->_m); // weight
    else{
      weight_vec = cml::vector_calloc<T>(this->_m); // weight make up
      cml::vector_add_constant(&weight_vec, static_cast<T>(1.0)); // make unity weights
    }
    cml::vector<T> ytemp = cml::vector_calloc<T>(this->_m); // b
    cml::vector<T> xtemp = cml::vector_calloc<T>(this->_n); // x
    cml::vector_memcpy(&ytemp, &y_vec); // y_vec->ytemp
    cml::vector_add_constant(&ytemp, -static_cast<T>(intercept)*mean[0]); // ytemp -> ytemp - intercept*mean[0]
    cml::vector_mul(&ytemp,&weight_vec); // ytemp*weight -> ytemp

    // Compute A^T . b
    if (_ord == MatrixDense<T>::ROW) {
      const cml::matrix<T, CblasRowMajor> A = cml::matrix_view_array<T, CblasRowMajor>(_data, this->_m, this->_n); // just view
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &ytemp, static_cast<T>(0.), &xtemp); // A.ytemp -> xtemp
    }
    else{
      const cml::matrix<T, CblasColMajor> A = cml::matrix_view_array<T, CblasColMajor>(_data, this->_m, this->_n); // just view
      cml::blas_gemv(hdl, CUBLAS_OP_T, static_cast<T>(1.), &A, &ytemp, static_cast<T>(0.), &xtemp); // A.ytemp -> xtemp
    }

    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(&xtemp.data[0]);

    lambda_max0 = thrust::transform_reduce(thrust::device,
                                           dev_ptr, dev_ptr + this->_n-intercept,
                                           absolute_value<T>(),
                                           static_cast<T>(0.0),
                                           thrust::maximum<T>());
  }
  else{
    lambda_max0 = 7000; // test
  }
  CUDA_CHECK_ERR();

  return 0;
}

  

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, const MatrixDense<T>& A) {
  switch (norm_type) {
    case kNorm2: {
      return Norm2Est(hdl, &A);
    }
    case kNormFro: {
      const cml::vector<T> a = cml::vector_view_array(A.Data(),
          A.Rows() * A.Cols());
      return cml::blas_nrm2(hdl, &a) / std::sqrt(std::min(A.Rows(), A.Cols()));
    }
    case kNorm1:
      // 1-norm normalization doens't make make sense since it treats rows and
      // columns differently.
    default:
      ASSERT(false);
      return static_cast<T>(0.);
  }
}


  

// Performs A := D * A * E for A in row major
template <typename T>
void __global__ __MultRow(size_t m, size_t n, const T *d, const T *e, T *data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t t = tid; t < m * n; t += gridDim.x * blockDim.x)
    data[t] *= d[t / n] * e[t % n];
}

// Performs A := D * A * E for A in col major
template <typename T>
void __global__ __MultCol(size_t m, size_t n, const T *d, const T *e, T *data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t t = tid; t < m * n; t += gridDim.x * blockDim.x)
    data[t] *= d[t % m] * e[t / m];
}

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDense<T>::Ord ord, T *data) {
  if (ord == MatrixDense<T>::ROW) {
    size_t grid_dim_row = cml::calc_grid_dim(m * n, cml::kBlockSize);
    __MultRow<<<grid_dim_row, cml::kBlockSize>>>(m, n, d, e, data);
  } else {
    size_t grid_dim_row = cml::calc_grid_dim(m * n, cml::kBlockSize);
    __MultCol<<<grid_dim_row, cml::kBlockSize>>>(m, n, d, e, data);
  }
}

}  // namespace

// Explicit template instantiation.
#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE==1
template class MatrixDense<double>;
#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE==1
template class MatrixDense<float>;
#endif




  // upload data function.  Uploads to a single GPU.
  // mimics otherwise similar MatrixDense constructor, but has no destruction of uploaded data pointers
template <typename T>
int makePtr_dense(int sharedA, int me, int wDev, size_t m, size_t n, size_t mValid, const char ord, const T *data, const T *datay, const T *vdata, const T *vdatay, const T *weight, void **_data, void **_datay, void **_vdata, void **_vdatay, void **_weight){
    checkwDev(wDev);
    CUDACHECK(cudaSetDevice(wDev));

    DEBUG_FPRINTF(stderr,"makePtr_dense: %d\n",0);

#ifdef DEBUG
    //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
    cudaDeviceProp props;
    CUDACHECK(cudaGetDeviceProperties(&props, wDev));
    DEBUG_FPRINTF(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev);
#endif

    // Copy Matrix to GPU (unlike CPU case, cannot copy just pointer because always assume input is CPU and output is GPU)
    double t0 = timer<double>();
    PUSH_RANGE("MDsendsource",MDsendsource,1);

    if(data){
      CUDACHECK(cudaMalloc(_data, m * n * sizeof(T))); // allocate on GPU
      CUDACHECK(cudaMemcpy(*_data, data, m * n * sizeof(T),cudaMemcpyHostToDevice)); // copy from orig CPU data to GPU
      //      fprintf(stderr,"_data: %p\n",(void*)*_data); fflush(stderr);
    }
    else *_data=NULL;

    if(datay){
      CUDACHECK(cudaMalloc(_datay, m * sizeof(T))); // allocate on GPU
      CUDACHECK(cudaMemcpy(*_datay, datay, m * sizeof(T),cudaMemcpyHostToDevice)); // copy from orig CPU data to GPU
      //      fprintf(stderr,"_datay: %p\n",(void*)*_datay); fflush(stderr);
    }
    else *_datay=NULL;

    if(vdata){
      CUDACHECK(cudaMalloc(_vdata, mValid * n * sizeof(T))); // allocate on GPU
      CUDACHECK(cudaMemcpy(*_vdata, vdata, mValid * n * sizeof(T),cudaMemcpyHostToDevice)); // copy from orig CPU data to GPU
      //      fprintf(stderr,"_vdata: %p\n",(void*)*_vdata); fflush(stderr);
    }
    else *_vdata=NULL;

    if(vdatay){
      CUDACHECK(cudaMalloc(_vdatay, mValid * sizeof(T))); // allocate on GPU
      CUDACHECK(cudaMemcpy(*_vdatay, vdatay, mValid * sizeof(T),cudaMemcpyHostToDevice)); // copy from orig CPU data to GPU
      //      fprintf(stderr,"_vdatay: %p\n",(void*)*_vdatay); fflush(stderr);
    }
    else *_vdatay=NULL;

    //    fprintf(stderr,"weight=%p\n",weight); fflush(stderr);
    if(weight){
      CUDACHECK(cudaMalloc(_weight, m * sizeof(T))); // allocate on GPU
      CUDACHECK(cudaMemcpy(*_weight, weight, m * sizeof(T),cudaMemcpyHostToDevice)); // copy from orig CPU data to GPU
    }
    else{
      DEBUG_FPRINTF(stderr,"making up unity weights: %d\n",m);
      CUDACHECK(cudaMalloc(_weight, m * sizeof(T))); // allocate on GPU
      thrust::device_ptr<T> dev_ptr=thrust::device_pointer_cast(static_cast<T*>(*_weight));
      T fill_value=1.0;
      thrust::fill(dev_ptr, dev_ptr + m, fill_value);
      //      fprintf(stderr,"_weight: %p\n",(void*)*_weight); fflush(stderr);
    }
    
    POP_RANGE("MDsendsource",MDsendsource,1);
    double t2 = timer<double>();
    DEBUG_FPRINTF(stdout,"Time to allocate and copy the data matrix on the GPU: %f\n", t2-t0);
    cudaDeviceSynchronize();
    
    DEBUG_FPRINTF(stderr,"pointer data   %p\n",(void*)*_data);
    DEBUG_FPRINTF(stderr,"pointer datay  %p\n",(void*)*_datay);
    DEBUG_FPRINTF(stderr,"pointer vdata  %p\n",(void*)*_vdata);
    DEBUG_FPRINTF(stderr,"pointer vdaty  %p\n",(void*)*_vdatay);
    DEBUG_FPRINTF(stderr,"pointer weight %p\n",(void*)*_weight);


    return(0);
}

  

  template int makePtr_dense<double>(int sharedA, int me, int wDev, size_t m, size_t n, size_t mValid, const char ord,
                                     const double *data, const double *datay, const double *vdata, const double *vdatay, const double *weight,
                                     void **_data, void **_datay, void **_vdata, void **_vdatay, void **_weight);
  template int makePtr_dense<float>(int sharedA, int me, int wDev, size_t m, size_t n, size_t mValid, const char ord,
                                    const float *data, const float *datay, const float *vdata, const float *vdatay, const float *weight,
                                    void **_data, void **_datay, void **_vdata, void **_vdatay, void **_weight);



template <typename T>
int modelFree1(T *aptr){

  if(aptr!=NULL){
    // for now, freed during ~
    //cudaFree(aptr);
    //CUDA_CHECK_ERR();
  }
  return(0);
}


  template int modelFree1<float>(float *aptr);
  template int modelFree1<double>(double *aptr);
  


}  // namespace h2o4gpu

#ifdef __cplusplus
extern "C" {
#endif

    int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        void**a, void**b, void**c, void**d, void **e) {
      return h2o4gpu::makePtr_dense<double>(sharedA, sourceme, sourceDev, mTrain, n, mValid, ord, trainX, trainY, validX, validY, weight, a, b, c, d, e);
    }
    int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       void**a, void**b, void**c, void**d, void **e) {
      return h2o4gpu::makePtr_dense<float>(sharedA, sourceme, sourceDev, mTrain, n, mValid, ord, trainX, trainY, validX, validY, weight, a, b, c, d, e);
    }

   int modelfree1_double(double *aptr){
    return h2o4gpu::modelFree1<double>(aptr);
  }
  int modelfree1_float(float *aptr){
    return h2o4gpu::modelFree1<float>(aptr);
  }

#ifdef __cplusplus
}
#endif



