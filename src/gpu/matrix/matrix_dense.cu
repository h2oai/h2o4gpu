#include <cublas_v2.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_vector.cuh"
#include "equil_helper.cuh"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "util.h"
#include "timer.h"

extern int checkwDev(int wDev);

namespace pogs {

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
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
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
MatrixDense<T>::MatrixDense(int wDev, char ord, size_t m, size_t n, const T *data)
  : Matrix<T>(m, n, 0), _wDev(wDev), _datatype(0),_data(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  _datay=NULL;
  _vdata=NULL;
  _vdatay=NULL;

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  fprintf(stderr,"MatrixDense1: ord=%c m=%d n=%d\n",ord,(int)m,(int)n);
  
#ifdef _DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  fprintf(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev); fflush(stderr);
#endif
  
  // Set GPU specific _info.


  PUSH_RANGE("MDnew",MDnew,1);
  GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
  this->_info = reinterpret_cast<void*>(info);
  POP_RANGE("MDnew",MDnew,1);

  // Copy Matrix to GPU.
  PUSH_RANGE("MDsend",MDsend,1);
  //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
  double t0 = timer<double>();
  cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
  double t1 = timer<double>();
  cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
  double t2 = timer<double>();
  printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
  printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
  POP_RANGE("MDsend",MDsend,1);
}
template <typename T>
MatrixDense<T>::MatrixDense(char ord, size_t m, size_t n, const T *data)
  : MatrixDense<T>(0, ord, m, n, data){}

  // datatype=0: data CPU pointer to _data CPU pointer // NA
  // datatype=1: data GPU pointer to _data GPU pointer
  // datatype=2: data CPU pointer to _data GPU pointer // NA
  // datatype=3: data CPU pointer to _data GPU pointer // NA
template <typename T>
MatrixDense<T>::MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, T *data)
  : Matrix<T>(m, n, 0), _wDev(wDev), _datatype(datatype),_data(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  _datay=NULL;
  _vdata=NULL;
  _vdatay=NULL;

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  fprintf(stderr,"MatrixDense2: ord=%c m=%d n=%d\n",ord,(int)m,(int)n);
  
#ifdef _DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  fprintf(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev); fflush(stderr);
#endif
  

  if(datatype==1){
    // no info->orig_data, so send 0 to GpuData
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    POP_RANGE("MDnew",MDnew,1);
    
  // source pointer is on this GPU

    // just copy GPU pointer
    _data = data;
  }
  else{
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    POP_RANGE("MDnew",MDnew,1);

  // Copy CPU Matrix to GPU.
    PUSH_RANGE("MDsend",MDsend,1);
    //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
    double t0 = timer<double>();
    cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
    double t1 = timer<double>();
    cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    double t2 = timer<double>();
    printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
    printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
    POP_RANGE("MDsend",MDsend,1);
  }
}


  // like original MatrixDense, but also feed in CPU data for trainY, validX, and validY
  // Used by elastic_net_mapd.cpp to pass CPU data and put on GPU
template <typename T>
MatrixDense<T>::MatrixDense(int wDev, char ord, size_t m, size_t n, size_t mValid, const T *data, const T *datay, const T *vdata, const T *vdatay)
  : Matrix<T>(m, n, mValid), _wDev(wDev), _datatype(0),_data(0), _datay(0), _vdata(0), _vdatay(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  fprintf(stderr,"MatrixDense3: ord=%c m=%d n=%d mValid=%d\n",ord,(int)m,(int)n,int(mValid));
  
#ifdef _DEBUG
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
  this->_info = reinterpret_cast<void*>(info);
  this->_infoy = reinterpret_cast<void*>(infoy);
  this->_vinfo = reinterpret_cast<void*>(vinfo);
  this->_vinfoy = reinterpret_cast<void*>(vinfoy);
  POP_RANGE("MDnew",MDnew,1);


  // Copy Matrix to GPU.
  PUSH_RANGE("MDsend",MDsend,1);
  //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
  double t0 = timer<double>();
  cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
  cudaMalloc(&_datay, this->_m * sizeof(T)); // allocate on GPU
  cudaMalloc(&_vdata, this->_mvalid * this->_n * sizeof(T)); // allocate on GPU
  cudaMalloc(&_vdatay, this->_mvalid * sizeof(T)); // allocate on GPU
  double t1 = timer<double>();
  cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
  cudaMemcpy(_datay, infoy->orig_data, this->_m * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
  cudaMemcpy(_vdata, vinfo->orig_data, this->_mvalid * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
  cudaMemcpy(_vdatay, vinfoy->orig_data, this->_mvalid * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
  double t2 = timer<double>();
  printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
  printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
  POP_RANGE("MDsend",MDsend,1);
}
  // like original MatrixDense, but also feed in CPU data for trainY, validX, and validY
  // Used by elastic_net_mapd.cpp to pass CPU data and put on GPU
  // datatype=0: CPU pointer to data
  // datatype=1: GPU pointer to data
template <typename T>
MatrixDense<T>::MatrixDense(int wDev, int datatype, char ord, size_t m, size_t n, size_t mValid, T *data, T *datay, T *vdata, T *vdatay)
  : Matrix<T>(m, n, mValid), _wDev(wDev), _datatype(datatype),_data(0), _datay(0), _vdata(0), _vdatay(0) {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));

  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  
  fprintf(stderr,"ord=%c m=%d n=%d mValid=%d\n",ord,(int)m,(int)n,int(mValid));
  
#ifdef _DEBUG
  //    CUDACHECK(cudaSetDeviceFlags(cudaDeviceMapHost)); // TODO: MapHostMemory
  cudaDeviceProp props;
  CUDACHECK(cudaGetDeviceProperties(&props, _wDev));
  fprintf(stderr,"Using: Compute %d.%d CUDA device: [%s] with id=%2d\n", props.major, props.minor, props.name,wDev); fflush(stderr);
#endif


  if(datatype==1){
    // source pointer is on GPU already
    // Set GPU specific _info.
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *infoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfo = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfoy = new GpuData<T>(0); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    this->_infoy = reinterpret_cast<void*>(infoy);
    this->_vinfo = reinterpret_cast<void*>(vinfo);
    this->_vinfoy = reinterpret_cast<void*>(vinfoy);
    POP_RANGE("MDnew",MDnew,1);


    // Just copy pointer
    _data = data;
    _datay = datay;
    _vdata = vdata;
    _vdatay = vdatay;
  }
  else{
    // source pointer is on CPU
    // Set GPU specific _info.
    PUSH_RANGE("MDnew",MDnew,1);
    GpuData<T> *info = new GpuData<T>(data); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *infoy = new GpuData<T>(datay); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfo = new GpuData<T>(vdata); // new structure (holds pointer to data and GPU handle)
    GpuData<T> *vinfoy = new GpuData<T>(vdatay); // new structure (holds pointer to data and GPU handle)
    this->_info = reinterpret_cast<void*>(info);
    this->_infoy = reinterpret_cast<void*>(infoy);
    this->_vinfo = reinterpret_cast<void*>(vinfo);
    this->_vinfoy = reinterpret_cast<void*>(vinfoy);
    POP_RANGE("MDnew",MDnew,1);


    // Copy CPU Matrix to GPU.
    PUSH_RANGE("MDsend",MDsend,1);
    //  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info); // cast void -> GpuData
    double t0 = timer<double>();
    cudaMalloc(&_data, this->_m * this->_n * sizeof(T)); // allocate on GPU
    cudaMalloc(&_datay, this->_m * sizeof(T)); // allocate on GPU
    cudaMalloc(&_vdata, this->_mvalid * this->_n * sizeof(T)); // allocate on GPU
    cudaMalloc(&_vdatay, this->_mvalid * sizeof(T)); // allocate on GPU
    double t1 = timer<double>();
    cudaMemcpy(_data, info->orig_data, this->_m * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    cudaMemcpy(_datay, infoy->orig_data, this->_m * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    cudaMemcpy(_vdata, vinfo->orig_data, this->_mvalid * this->_n * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    cudaMemcpy(_vdatay, vinfoy->orig_data, this->_mvalid * sizeof(T),cudaMemcpyHostToDevice); // copy from orig CPU data to GPU
    double t2 = timer<double>();
    printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
    printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
    POP_RANGE("MDsend",MDsend,1);
  }
}


  // MatrixDense where input actual A object that contains all CPU information, but need to go from 1 GPU to multiple GPU
  // Used by elastic_net_mapd.cpp inside openmp loop for each core
template <typename T>
MatrixDense<T>::MatrixDense(int wDev, const MatrixDense<T>& A)
  : Matrix<T>(A._m, A._n, A._mvalid), _wDev(wDev), _data(0), _ord(A._ord) {

  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  
  PUSH_RANGE("MDnew",MDnew,2);
  GpuData<T> *info_A   = reinterpret_cast<GpuData<T>*>(A._info); // cast from void to GpuData
  GpuData<T> *infoy_A  = reinterpret_cast<GpuData<T>*>(A._infoy); // cast from void to GpuData
  GpuData<T> *vinfo_A  = reinterpret_cast<GpuData<T>*>(A._vinfo); // cast from void to GpuData
  GpuData<T> *vinfoy_A = reinterpret_cast<GpuData<T>*>(A._vinfoy); // cast from void to GpuData
  
  GpuData<T> *info;
  GpuData<T> *infoy;
  GpuData<T> *vinfo;
  GpuData<T> *vinfoy;
  if(A._data) info = new GpuData<T>(info_A->orig_data); // create new GpuData structure with point to CPU data
  if(A._datay) infoy  = new GpuData<T>(infoy_A->orig_data); // create new GpuData structure with point to CPU data
  if(A._vdata) vinfo  = new GpuData<T>(vinfo_A->orig_data); // create new GpuData structure with point to CPU data
  if(A._vdatay) vinfoy = new GpuData<T>(vinfoy_A->orig_data); // create new GpuData structure with point to CPU data
  
  this->_info = reinterpret_cast<void*>(info); // back to cast as void
  this->_infoy = reinterpret_cast<void*>(infoy); // back to cast as void
  this->_vinfo = reinterpret_cast<void*>(vinfo); // back to cast as void
  this->_vinfoy = reinterpret_cast<void*>(vinfoy); // back to cast as void
  POP_RANGE("MDnew",MDnew,2);


  if(A._wDev == _wDev){ // if on same device, just copy pointer
    _data   = A._data;
    _datay  = A._datay;
    _vdata  = A._vdata;
    _vdatay = A._vdatay;
  }
  else{
    // Copy Matrix to from source GPU to this GPU
    PUSH_RANGE("MDcopy",MDcopy,1);
    //GpuData<T> *info = reinterpret_cast<GpuData<T>*>(_info); // cast void -> GpuData
    double t0 = timer<double>();
    if(A._data) cudaMalloc(&_data, A._m * A._n * sizeof(T)); // allocate on GPU
    if(A._datay) cudaMalloc(&_datay, A._m * sizeof(T)); // allocate on GPU
    if(A._vdata) cudaMalloc(&_vdata, A._mvalid * A._n * sizeof(T)); // allocate on GPU
    if(A._vdatay) cudaMalloc(&_vdatay, A._mvalid * sizeof(T)); // allocate on GPU
    double t1 = timer<double>();
    if(A._data) cudaMemcpyPeer(_data, _wDev, A._data, A._wDev, A._m * A._n * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
    if(A._datay) cudaMemcpyPeer(_datay, _wDev, A._datay, A._wDev, A._m * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
    if(A._vdata) cudaMemcpyPeer(_vdata, _wDev, A._vdata, A._wDev, A._mvalid * A._n * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
    if(A._vdatay) cudaMemcpyPeer(_vdatay, _wDev, A._vdatay, A._wDev, A._mvalid * sizeof(T)); // dest: _data destid: _wDev  source: A._data sourceid: A._wDev
    double t2 = timer<double>();
    printf("Time to allocate the data matrix on the GPU: %f\n", t1-t0);
    printf("Time to copy the data matrix to the GPU    : %f\n", t2-t1);
    POP_RANGE("MDcopy",MDcopy,1);
  }
}

template <typename T>
MatrixDense<T>::MatrixDense(const MatrixDense<T>& A)
  : MatrixDense<T>(A._wDev, A){}

template <typename T>
MatrixDense<T>::~MatrixDense() {
  checkwDev(_wDev);
  CUDACHECK(cudaSetDevice(_wDev));
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  delete info;
  this->_info = 0;

  if (this->_done_init && _data) {
    cudaFree(_data);
    this->_data = 0;
    DEBUG_CUDA_CHECK_ERR();
  }
  if (this->_done_init && _datay) {
    cudaFree(_datay);
    this->_datay = 0;
    DEBUG_CUDA_CHECK_ERR();
  }
  if (this->_done_init && _vdata) {
    cudaFree(_vdata);
    this->_vdata = 0;
    DEBUG_CUDA_CHECK_ERR();
  }
  if (this->_done_init && _vdatay) {
    cudaFree(_vdatay);
    this->_vdatay = 0;
    DEBUG_CUDA_CHECK_ERR();
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
void MatrixDense<T>::GetTrainX(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(datatype==1){
    cudaMemcpy(*data, _data, size* sizeof(T),cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR();
  }
  else{
    std::memcpy(*data, _data, size * sizeof(T));
  }

  return;
}
template <typename T>
void MatrixDense<T>::GetTrainY(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(datatype==1){
    cudaMemcpy(*data, _datay, size* sizeof(T),cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR();
  }
  else{
    std::memcpy(*data, _datay, size * sizeof(T));
  }

  return;
}

template <typename T>
void MatrixDense<T>::GetValidX(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(datatype==1){
    cudaMemcpy(*data, _vdata, size* sizeof(T),cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR();
  }
  else{
    std::memcpy(*data, _vdata, size * sizeof(T));
  }

  return;
}
template <typename T>
void MatrixDense<T>::GetValidY(int datatype, size_t size, T**data) const {

  CUDACHECK(cudaSetDevice(_wDev));

  if(datatype==1){
    cudaMemcpy(*data, _vdatay, size* sizeof(T),cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR();
  }
  else{
    std::memcpy(*data, _vdatay, size * sizeof(T));
  }


  return;
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


  // Equilibration (precondition) matrix using Sinkhorn Knopp method wrapped to allow any norm
  // See https://arxiv.org/pdf/1610.03871.pdf for more information
template <typename T>
int MatrixDense<T>::Equil(T *d, T *e, bool equillocal) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  CUDACHECK(cudaSetDevice(_wDev));

  // Extract cublas handle from _info.
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  // Number of elements in matrix.
  size_t num_el = this->_m * this->_n;

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign;
  size_t num_sign_bytes = (num_el + 7) / 8;
  cudaMalloc(&sign, num_sign_bytes);
  CUDA_CHECK_ERR();

  // Fill sign bits, assigning each thread a multiple of 8 elements.
  size_t num_chars = num_el / 8;
  size_t grid_size = cml::calc_grid_dim(num_chars, cml::kBlockSize);
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

  // Perform Sinkhorn-Knopp equilibration to obtain a doubly stochastic matrix.
  SinkhornKnopp(this, d, e, equillocal);
  wrapcudaDeviceSynchronize();

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
#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
template class MatrixDense<double>;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class MatrixDense<float>;
#endif

}  // namespace pogs

