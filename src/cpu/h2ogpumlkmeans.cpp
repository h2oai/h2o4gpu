#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <iostream>
#include <cstdlib>
#include "h2ogpumlkmeans.h"
#include <random>
#include <algorithm>
#include <vector>
//#include "mkl.h"
#include "cblas.h"
#include <atomic>
#include <csignal>

#define VERBOSE 1

#include "h2ogpumlkmeans_kmeanscpu.h"


template<typename T>
void random_data(int verbose, std::vector<T>& array, int m, int n) {
  for(int i = 0; i < m * n; i++) {
    array[i] = (T)rand()/(T)RAND_MAX;
  }
}

template<typename T>
void nonrandom_data(int verbose, const char ord, std::vector<T>& array, const T *srcdata, int q, int n, int npercpu, int d) {
  if(ord=='c'){
	if(verbose){ fprintf(stderr,"COL ORDER -> ROW ORDER\n"); fflush(stderr); }
    int indexi,indexj;
    for(int i = 0; i < npercpu * d; i++) {
#if(1)
      indexi = i%d; // col
      indexj = i/d +  q*npercpu; // row (shifted by which cpu)
      //      array[i] = srcdata[indexi*n + indexj];
      array[i] = srcdata[indexi*n + indexj];
#else
      indexj = i%d;
      indexi = i/d;
      //      array[i] = srcdata[indexi*n + indexj];
      array[i] = srcdata[indexi*d + indexj];
#endif
    }
#if(DEBUG)
    for(int i = 0; i < npercpu; i++) {
      for(int j = 0; j < d; j++) {
        fprintf(stderr,"q=%d initdata[%d,%d]=%g\n",q,i,j,array[i*d+j]); fflush(stderr);
      }
    }
#endif
  }
  else{
    if(verbose) { fprintf(stderr,"ROW ORDER not changed\n"); fflush(stderr); }
    for(int i = 0; i < npercpu * d; i++) {
      array[i] = srcdata[q*npercpu*d + i]; // shift by which cpu
    }
  }
}
template<typename T>
void nonrandom_data_new(int verbose, std::vector<int> v, const char ord, std::vector<T>& array, const T *srcdata, int q, int n, int npercpu, int d) {

  if(ord=='c'){
    if(verbose){ fprintf(stderr,"COL ORDER -> ROW ORDER\n"); fflush(stderr); }
     for(int i = 0; i < npercpu; i++) {
      for(int j=0;j<d;j++){
        array[i*d + j] = srcdata[v[q*npercpu + i] + j*n]; // shift by which cpu
      }
    }
#if(DEBUG)
    for(int i = 0; i < npercpu; i++) {
      for(int j = 0; j < d; j++) {
        fprintf(stderr,"q=%d initdata[%d,%d]=%g\n",q,i,j,array[i*d+j]); fflush(stderr);
      }
    }
#endif
  }
  else{
    if(verbose) { fprintf(stderr,"ROW ORDER not changed\n"); fflush(stderr); }
     for(int i = 0; i < npercpu; i++) {
      for(int j=0;j<d;j++){
        array[i*d + j] = srcdata[v[q*npercpu + i]*d + j]; // shift by which cpu
      }
    }
  }
}

void random_labels(int verbose, std::vector<int>& labels, int n, int k) {
  for(int i = 0; i < n; i++) {
    labels[i] = rand() % k;
  }
}
void nonrandom_labels(int verbose, const char ord, std::vector<int>& labels, const int *srclabels, int q, int n, int npercpu) {
  int d=1; // only 1 dimension
  if(ord=='c'){
    if(verbose) { fprintf(stderr,"labels COL ORDER -> ROW ORDER\n"); fflush(stderr); }
    int indexi,indexj;
    for(int i = 0; i < npercpu; i++) {
#if(1)
      indexi = i%d; // col
      indexj = i/d +  q*npercpu; // row (shifted by which cpu)
      //      labels[i] = srclabels[indexi*n + indexj];
      labels[i] = srclabels[indexi*n + indexj];
#else
      indexj = i%d;
      indexi = i/d;
      //      labels[i] = srclabels[indexi*n + indexj];
      labels[i] = srclabels[indexi*d + indexj];
#endif
    }
  }
  else{
    if(verbose) { fprintf(stderr,"labels ROW ORDER not changed\n"); fflush(stderr); }
    for(int i = 0; i < npercpu; i++) {
      labels[i] = srclabels[q*npercpu*d + i]; // shift by which cpu
    }
  }
}

template<typename T>
void random_centroids(int verbose, const char ord, std::vector<T>& array, const T *srcdata, int q, int n, int npercpu, int d, int k) {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());
  //  std::uniform_int_distribution<>dis(0, npercpu-1); // random i in range from 0..npercpu-1
  std::uniform_int_distribution<>dis(0, n-1); // random i in range from 0..n-1 (i.e. only 1 cpu gets centroids)
  
  if(ord=='c'){
    if(verbose){
      if(verbose) { fprintf(stderr,"COL ORDER -> ROW ORDER\n"); fflush(stderr); }
    }
    for(int i = 0; i < k; i++) { // rows
      int reali = dis(gen); // + q*npercpu; // row sampled (called indexj above)
      for(int j = 0; j < d; j++) { // cols
        array[i*d+j] = srcdata[reali + j*n];
#if(DEBUG)
        fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,reali,array[i*d+j]); fflush(stderr);
#endif
      }
    }
  }
  else{
    if(verbose){
      fprintf(stderr,"ROW ORDER not changed\n"); fflush(stderr);
    }
    for(int i = 0; i < k; i++) { // rows
      int reali = dis(gen); // + q*npercpu ; // row sampled
      for(int j = 0; j < d; j++) { // cols
        array[i*d+j] = srcdata[reali*d + j];
      }
    }
  }
}
template<typename T>
void random_centroids_new(int verbose, std::vector<int> v, const char ord, std::vector<T>& array, const T *srcdata, int q, int n, int npercpu, int d, int k) {

  if(ord=='c'){
    if(VERBOSE){
      fprintf(stderr,"COL ORDER -> ROW ORDER\n"); fflush(stderr);
    }
    for(int i = 0; i < k; i++) { // rows
      for(int j = 0; j < d; j++) { // cols
        array[i*d+j] = srcdata[v[i] + j*n];
#if(DEBUG)
        fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,v[i],array[i*d+j]); fflush(stderr);
#endif
      }
    }
  }
  else{
    if(VERBOSE){
      fprintf(stderr,"ROW ORDER not changed\n"); fflush(stderr);
    }
    for(int i = 0; i < k; i++) { // rows
      for(int j = 0; j < d; j++) { // cols
        array[i*d+j] = srcdata[v[i]*d + j];
      }
    }
  }
}

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace h2ogpumlkmeans {
    volatile std::atomic_int flag(0);
    inline void my_function(int sig){ // can be called asynchronously
      fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
      flag = 1;
    }

    template <typename T>
    H2OGPUMLKMeansCPU<T>::H2OGPUMLKMeansCPU(const T* A, int k, int n, int d)
    {
      _A = A; _k = k; _n = n; _d = d;
    }

    template <typename T>
    int makePtr_dense(int verbose, int cpu_idtry, int n_cputry, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, T threshold, const T* srcdata, const int* srclabels, void ** res) {

      if(rows>std::numeric_limits<int>::max()){
        fprintf(stderr,"rows>%d now implemented\n",std::numeric_limits<int>::max());
        fflush(stderr);
        exit(0);
      }
      
      int n=rows;
      int d=cols;
      std::signal(SIGINT, my_function);
      std::signal(SIGTERM, my_function);

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


      int n_cpu=1; // ignore try
      int cpu_id=0; // ignore try
      int n_cpuvis=n_cpu; // fake
      
      // setup CPU list to use
      std::vector<int> dList(n_cpu);
      for(int idx=0;idx<n_cpu;idx++){
        int device_idx = (cpu_id + idx) % n_cpuvis;
        dList[idx] = device_idx;
      }
      
        
      double t0t = timer<double>();
      std::vector<T> *data[n_cpu];
      std::vector<int> *labels[n_cpu];
      std::vector<T> *centroids[n_cpu];
      std::vector<T> *distances[n_cpu];
      for (int q = 0; q < n_cpu; q++) {
        data[q] = new std::vector<T>(n/n_cpu*d);
        labels[q] = new std::vector<int>(n/n_cpu*d);
        centroids[q] = new std::vector<T>(k * d);
        distances[q] = new std::vector<T>(n);
      }
      

      std::cout << "Number of points: " << n << std::endl;
      std::cout << "Number of dimensions: " << d << std::endl;
      std::cout << "Number of clusters: " << k << std::endl;
      std::cout << "Max. number of iterations: " << max_iterations << std::endl;
      std::cout << "Stopping threshold: " << threshold << std::endl;

      // setup random sequence for sampling data
      //      std::random_device rd;
      //      std::mt19937 g(rd());
      std::vector<int> v(n);
      std::iota (std::begin(v), std::end(v), 0); // Fill with 0, 1, ..., 99.
      std::random_shuffle(v.begin(), v.end());

      for (int q = 0; q < n_cpu; q++) {
        if(init_labels==0){ // random
          random_labels(verbose, *labels[q], n/n_cpu, k);
        }
        else{
          nonrandom_labels(verbose, ord, *labels[q], &srclabels[0], q, n, n/n_cpu);
        }
        if(init_data==0){ // random (for testing)
          random_data<T>(verbose, *data[q], n/n_cpu, d);
        }
        else if(init_data==1){ // shard by row
          nonrandom_data(verbose, ord, *data[q], &srcdata[0], q, n, n/n_cpu, d);
        }
        else{ // shard by randomly (without replacement) selected by row
          nonrandom_data_new(verbose, v, ord, *data[q], &srcdata[0], q, n, n/n_cpu, d);
        }
      }
      // get non-random centroids on 1 cpu, then share with rest.
      if(init_from_labels==0){
        int masterq=0;
        //random_centroids(verbose, ord, *centroids[masterq], &srcdata[0], masterq, n, n/n_cpu, d, k);
        random_centroids_new(verbose, v, ord, *centroids[masterq], &srcdata[0], masterq, n, n/n_cpu, d, k);
#if(DEBUG)
        std::vector<T> h_centroidq=*centroids[q];
        for(int ii=0;ii<k*d;ii++){
          fprintf(stderr,"q=%d initcent[%d]=%g\n",q,ii,h_centroidq[ii]); fflush(stderr);
        }
#endif
      }
      double timetransfer = static_cast<double>(timer<double>() - t0t);

      
      double t0 = timer<double>();
      int masterq=0;
      int status=kmeans::kmeans<T>(verbose, &flag, n,d,k,*data[masterq],*labels[masterq],*centroids[masterq],max_iterations,init_from_labels,threshold);
      if(status) return(status);
      double timefit = static_cast<double>(timer<double>() - t0);

      
      std::cout << "  Time fit: " << timefit << " s" << std::endl;
      fprintf(stderr,"Timetransfer: %g Timefit: %g\n",timetransfer,timefit); fflush(stderr);

      // copy result of centroids (sitting entirely on each device) back to host
      std::vector<T> *ctr = new std::vector<T>(*centroids[0]);
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

      // done with CPU data
      for (int q = 0; q < n_cpu; q++) {
        delete(data[q]);
        delete(labels[q]);
        delete(centroids[q]);
        delete(distances[q]);
      }

      return 0;
    }
  template int makePtr_dense<float>(int verbose, int cpu_id, int n_cpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, float threshold, const float *srcdata, const int *srclabels, void **a);
  template int makePtr_dense<double>(int verbose, int cpu_id, int n_cpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, double threshold, const double *srcdata, const int *srclabels, void **a);


// Explicit template instantiation.
#if !defined(H2OGPUML_DOUBLE) || H2OGPUML_DOUBLE==1
  template class H2OGPUMLKMeansCPU<double>;
#endif

#if !defined(H2OGPUML_SINGLE) || H2OGPUML_SINGLE==1
  template class H2OGPUMLKMeansCPU<float>;
#endif

}  // namespace h2ogpumlkmeans

#ifdef __cplusplus
extern "C" {
#endif

  int make_ptr_float_kmeans(int verbose, int cpu_id, int n_cpu, size_t mTrain, size_t n, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, float threshold, const float* srcdata, const int* srclabels, void** res) {
    return h2ogpumlkmeans::makePtr_dense<float>(verbose, cpu_id, n_cpu, mTrain, n, ord, k, max_iterations, init_from_labels, init_labels, init_data, threshold, srcdata, srclabels, res);
}
  int make_ptr_double_kmeans(int verbose, int cpu_id, int n_cpu, size_t mTrain, size_t n, const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data, double threshold, const double* srcdata, const int* srclabels, void** res) {
    return h2ogpumlkmeans::makePtr_dense<double>(verbose, cpu_id, n_cpu, mTrain, n, ord, k, max_iterations, init_from_labels, init_labels, init_data, threshold, srcdata, srclabels, res);
  }

#ifdef __cplusplus
}
#endif

