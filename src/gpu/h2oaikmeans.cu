#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <thrust/device_vector.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include "h2oaikmeans.h"
#include "kmeans.h"
#include <random>
#include "include/kmeans_general.h"

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)


template<typename T>
void fill_array(T& array, int m, int n) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      array[i * n + j] = (i % 2)*3 + j;
    }
  }
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int m, int n) {
  thrust::host_vector<T> host_array(m*n);
  for(int i = 0; i < m * n; i++) {
    host_array[i] = (T)rand()/(T)RAND_MAX;
  }
  array = host_array;
}

template<typename T>
void nonrandom_data(const char ord, thrust::device_vector<T>& array, const T *srcdata, int q, int n, int npergpu, int d) {
  thrust::host_vector<T> host_array(npergpu*d);
  if(ord=='c'){
    fprintf(stderr,"COL ORDER -> ROW ORDER\n"); fflush(stderr);
    int indexi,indexj;
    for(int i = 0; i < npergpu * d; i++) {
#if(1)
      indexi = i%d; // col
      indexj = i/d +  q*npergpu; // row (shifted by which gpu)
      //      host_array[i] = srcdata[indexi*n + indexj];
      host_array[i] = srcdata[indexi*n + indexj];
#else
      indexj = i%d;
      indexi = i/d;
      //      host_array[i] = srcdata[indexi*n + indexj];
      host_array[i] = srcdata[indexi*d + indexj];
#endif
    }
  }
  else{
    fprintf(stderr,"ROW ORDER not changed\n"); fflush(stderr);
    for(int i = 0; i < npergpu * d; i++) {
      host_array[i] = srcdata[q*npergpu*d + i]; // shift by which gpu
    }
  }
  array = host_array;
}

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
  thrust::host_vector<int> host_labels(n);
  for(int i = 0; i < n; i++) {
    host_labels[i] = rand() % k;
  }
  labels = host_labels;
}
void nonrandom_labels(const char ord, thrust::device_vector<int>& labels, const int *srclabels, int q, int n, int npergpu) {
  thrust::host_vector<int> host_labels(npergpu);
  int d=1; // only 1 dimension
  if(ord=='c'){
    fprintf(stderr,"labels COL ORDER -> ROW ORDER\n"); fflush(stderr);
    int indexi,indexj;
    for(int i = 0; i < npergpu; i++) {
#if(1)
      indexi = i%d; // col
      indexj = i/d +  q*npergpu; // row (shifted by which gpu)
      //      host_labels[i] = srclabels[indexi*n + indexj];
      host_labels[i] = srclabels[indexi*n + indexj];
#else
      indexj = i%d;
      indexi = i/d;
      //      host_labels[i] = srclabels[indexi*n + indexj];
      host_labels[i] = srclabels[indexi*d + indexj];
#endif
    }
  }
  else{
    fprintf(stderr,"labels ROW ORDER not changed\n"); fflush(stderr);
    for(int i = 0; i < npergpu; i++) {
      host_labels[i] = srclabels[q*npergpu*d + i]; // shift by which gpu
    }
  }
  labels = host_labels;
}

template<typename T>
void random_centroids(const char ord, thrust::device_vector<T>& array, const T *srcdata, int q, int n, int npergpu, int d, int k) {
  thrust::host_vector<T> host_array(k*d);
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());
  //  std::uniform_int_distribution<>dis(0, npergpu-1); // random i in range from 0..npergpu-1
    std::uniform_int_distribution<>dis(0, n-1); // random i in range from 0..n-1 (i.e. only 1 gpu gets centroids)
  if(ord=='c'){
    if(VERBOSE){
      fprintf(stderr,"COL ORDER -> ROW ORDER\n"); fflush(stderr);
    }
    for(int i = 0; i < k; i++) { // rows
      int reali = dis(gen); // + q*npergpu; // row sampled (called indexj above)
      for(int j = 0; j < d; j++) { // cols
        host_array[i*d+j] = srcdata[reali + j*n];
#if(DEBUG)
        fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,reali,host_array[i*d+j]); fflush(stderr);
#endif
      }
    }
  }
  else{
    if(VERBOSE){
      fprintf(stderr,"ROW ORDER not changed\n"); fflush(stderr);
    }
    for(int i = 0; i < k; i++) { // rows
      int reali = dis(gen); // + q*npergpu ; // row sampled
      for(int j = 0; j < d; j++) { // cols
        host_array[i*d+j] = srcdata[reali*d + j];
      }
    }
  }
  array = host_array;
}

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace h2oaikmeans {
    volatile std::atomic_int flag(0);
    inline void my_function(int sig){ // can be called asynchronously
      fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
      flag = 1;
    }

    template <typename T>
    H2OAIKMeans<T>::H2OAIKMeans(const T* A, int k, size_t n, size_t d)
    {
      _A = A; _k = k; _n = n; _d = d;
    }

    template <typename T>
    int H2OAIKMeans<T>::Solve() {
      int max_iterations = 10000;
      int n = 260753;  // rows
      int d = 298;  // cols
      int k = 100;  // clusters
      double thresh = 1e-3;  // relative improvement

      int n_gpu;
      cudaGetDeviceCount(&n_gpu);
      std::cout << n_gpu << " gpus." << std::endl;

      thrust::device_vector<T> *data[MAX_NGPUS];
      thrust::device_vector<int> *labels[MAX_NGPUS];
      thrust::device_vector<T> *centroids[MAX_NGPUS];
      thrust::device_vector<T> *distances[MAX_NGPUS];
      for (int q = 0; q < n_gpu; q++) {
        cudaSetDevice(q);
        data[q] = new thrust::device_vector<T>(n/n_gpu*d);
        labels[q] = new thrust::device_vector<int>(n/n_gpu);
        centroids[q] = new thrust::device_vector<T>(k * d);
        distances[q] = new thrust::device_vector<T>(n);
      }

      std::cout << "Generating random data" << std::endl;
      std::cout << "Number of points: " << n << std::endl;
      std::cout << "Number of dimensions: " << d << std::endl;
      std::cout << "Number of clusters: " << k << std::endl;
      std::cout << "Max. number of iterations: " << max_iterations << std::endl;
      std::cout << "Stopping threshold: " << thresh << std::endl;

      for (int q = 0; q < n_gpu; q++) {
        random_data<T>(*data[q], n/n_gpu, d);
        random_labels(*labels[q], n/n_gpu, k);
      }

      double t0 = timer<double>();
      kmeans::kmeans<T>(&flag, n, d, k, data, labels, centroids, distances, n_gpu, max_iterations, true, thresh);
      double time = static_cast<double>(timer<double>() - t0);
      std::cout << "  Time: " << time << " s" << std::endl;

      for (int q = 0; q < n_gpu; q++) {
        delete(data[q]);
        delete(labels[q]);
        delete(centroids[q]);
        delete(distances[q]);
      }
      return 0;
    }

    template <typename T>
    int makePtr_dense(int n_gputry, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, T threshold, const T* srcdata, const int* srclabels, void ** res) {
      int n=rows;
      int d=cols;
      signal(SIGINT, my_function);
      signal(SIGTERM, my_function);

      int printsrcdata=0;
      if(printsrcdata){
        for(unsigned int ii=0;ii<n;ii++){
          for(unsigned int jj=0;jj<d;jj++){
            fprintf(stderr,"%2g ",srcdata[ii*d+jj]);
          }
          fprintf(stderr," |  ");
        }
        fflush(stderr);
      }


      int n_gpu;
      cudaGetDeviceCount(&n_gpu);
      if(n_gputry<n_gpu) n_gpu=n_gputry; // no more than visible
      std::cout << n_gpu << " gpus." << std::endl;

      
      double t0t = timer<double>();
      thrust::device_vector<T> *data[n_gpu];
      thrust::device_vector<int> *labels[n_gpu];
      thrust::device_vector<T> *centroids[n_gpu];
      thrust::device_vector<T> *distances[n_gpu];
      for (int q = 0; q < n_gpu; q++) {
        CUDACHECK(cudaSetDevice(q));
        data[q] = new thrust::device_vector<T>(n/n_gpu*d);
        labels[q] = new thrust::device_vector<int>(n/n_gpu*d);
        centroids[q] = new thrust::device_vector<T>(k * d);
        distances[q] = new thrust::device_vector<T>(n);
      }

      std::cout << "Number of points: " << n << std::endl;
      std::cout << "Number of dimensions: " << d << std::endl;
      std::cout << "Number of clusters: " << k << std::endl;
      std::cout << "Max. number of iterations: " << max_iterations << std::endl;
      std::cout << "Stopping threshold: " << threshold << std::endl;

      for (int q = 0; q < n_gpu; q++) {
        CUDACHECK(cudaSetDevice(q));
        std::cout << "Copying data to device: " << q << std::endl;
        //        fprintf(stderr,"q=%d %p %p %p\n",q,&srcdata[q*n/n_gpu*d],&srcdata[(q+1)*n/n_gpu*d],&(data[q]->data[0])); fflush(stderr);

        //        std::vector<T> vdata(&srcdata[q*n/n_gpu*d],&srcdata[(q+1)*n/n_gpu*d]);
        //        thrust::copy(vdata.begin(),vdata.end(),data[q]->begin());
        //random_labels(*labels[q], n/n_gpu, k);
        nonrandom_data(ord, *data[q], &srcdata[0], q, n, n/n_gpu, d);
        nonrandom_labels(ord, *labels[q], &srclabels[0], q, n, n/n_gpu);
      }
      // get non-random centroids on 1 gpu, then share with rest.
      if(init_from_labels==0){
        int master_device=0;
        int q=master_device;
        CUDACHECK(cudaSetDevice(q));
        random_centroids(ord, *centroids[q], &srcdata[0], q, n, n/n_gpu, d, k);
        size_t bytecount = d*k; // all centroids

        // copy centroids to rest of gpus asynchronously
        std::vector<cudaStream_t *> streams;
        streams.resize(n_gpu);
        for (int q = 1; q < n_gpu; q++) {
          CUDACHECK(cudaSetDevice(q));
          std::cout << "Copying centroid data to device: " << q << std::endl;

          streams[q] = reinterpret_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t)));
          cudaStreamCreate(streams[q]);
          cudaMemcpyPeerAsync(thrust::raw_pointer_cast(&(*centroids[q])[0]),q,thrust::raw_pointer_cast(&(*centroids[master_device])[0]),master_device,bytecount,*(streams[q]));
        }
        for (int q = 1; q < n_gpu; q++) {
          cudaSetDevice(q);
          cudaStreamSynchronize(*(streams[q]));
        }
        for (int q = 1; q < n_gpu; q++) {
          cudaSetDevice(q);
          cudaStreamDestroy(*(streams[q]));
        }
      }
      
      
      double timetransfer = static_cast<double>(timer<double>() - t0t);

      double t0 = timer<double>();
      kmeans::kmeans<T>(&flag, n,d,k,data,labels,centroids,distances,n_gpu,max_iterations,init_from_labels,threshold);
      double timefit = static_cast<double>(timer<double>() - t0);
      std::cout << "  Time fit: " << timefit << " s" << std::endl;
      fprintf(stderr,"Timetransfer: %g Timefit: %g\n",timetransfer,timefit); fflush(stderr);

      // copy result of centroids (sitting entirely on each device) back to host
      thrust::host_vector<T> *ctr = new thrust::host_vector<T>(*centroids[0]);
      // TODO FIXME: When do delete this ctr memory?
      //      cudaMemcpy(ctr->data().get(), centroids[0]->data().get(), sizeof(T)*k*d, cudaMemcpyDeviceToHost);
      *res = ctr->data();

      // debug
      int printcenters=0;
      if(printcenters){
        for(unsigned int ii=0;ii<k;ii++){
          fprintf(stderr,"ii=%d of k=%d ",ii,k);
          for(unsigned int jj=0;jj<d;jj++){
            fprintf(stderr,"%g ",(*ctr)[d*ii+jj]);
          }
          fprintf(stderr,"\n");
          fflush(stderr);
        }
      }

      // done with GPU data
      for (int q = 0; q < n_gpu; q++) {
        delete(data[q]);
        delete(labels[q]);
        delete(centroids[q]);
        delete(distances[q]);
      }

      return 0;
    }
  template int makePtr_dense<float>(int n_gpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, float threshold, const float *srcdata, const int *srclabels, void **a);
  template int makePtr_dense<double>(int n_gpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, double threshold, const double *srcdata, const int *srclabels, void **a);


// Explicit template instantiation.
#if !defined(H2OAIGLM_DOUBLE) || H2OAIGLM_DOUBLE==1
  template class H2OAIKMeans<double>;
#endif

#if !defined(H2OAIGLM_SINGLE) || H2OAIGLM_SINGLE==1
  template class H2OAIKMeans<float>;
#endif

}  // namespace h2oaikmeans

#ifdef __cplusplus
extern "C" {
#endif

  int make_ptr_float_kmeans(int n_gpu, size_t mTrain, size_t n, const char ord, int k, int max_iterations, int init_from_labels, float threshold, const float* srcdata, const int* srclabels, void** res) {
    return h2oaikmeans::makePtr_dense<float>(n_gpu, mTrain, n, ord, k, max_iterations, init_from_labels, threshold, srcdata, srclabels, res);
}
  int make_ptr_double_kmeans(int n_gpu, size_t mTrain, size_t n, const char ord, int k, int max_iterations, int init_from_labels, double threshold, const double* srcdata, const int* srclabels, void** res) {
    return h2oaikmeans::makePtr_dense<double>(n_gpu, mTrain, n, ord, k, max_iterations, init_from_labels, threshold, srcdata, srclabels, res);
}

#ifdef __cplusplus
}
#endif
