#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <thrust/device_vector.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include "h2oaikmeans.h"
#include "kmeans.h"

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

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
  thrust::host_vector<int> host_labels(n);
  for(int i = 0; i < n; i++) {
    host_labels[i] = rand() % k;
  }
  labels = host_labels;
}

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace h2oaikmeans {

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

      thrust::device_vector<T> *data[16];
      thrust::device_vector<int> *labels[16];
      thrust::device_vector<T> *centroids[16];
      thrust::device_vector<T> *distances[16];
      for (int q = 0; q < n_gpu; q++) {
        cudaSetDevice(q);
        data[q] = new thrust::device_vector<T>(n/n_gpu*d);
        labels[q] = new thrust::device_vector<int>(n/n_gpu*d);
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
      kmeans::kmeans<T>(n, d, k, data, labels, centroids, distances, n_gpu, max_iterations, true, thresh);
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
    int makePtr_dense(int n_gpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, T threshold, const T* srcdata, const int* srclabels, void ** res) {
      int n=rows;
      int d=cols;

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

      bool init_from_labels=true;
#if(1)
      for (int q = 0; q < n_gpu; q++) {
        CUDACHECK(cudaSetDevice(q));
        std::cout << "Copying data to device: " << q << std::endl;
#if(0)
        CUDACHECK(cudaMemcpy(data[q]->data().get(), &srcdata[q*n/n_gpu*d], sizeof(T)*n/n_gpu*d, cudaMemcpyHostToDevice));
        std::cout << "Done copying data to device: " << q << " of bytes size " << sizeof(T)*n/n_gpu*d << std::endl;
        CUDACHECK(cudaMemcpy(labels[q]->data().get(), &srclabels[q*n/n_gpu*d], sizeof(T)*n/n_gpu*d, cudaMemcpyHostToDevice));
        std::cout << "Done copying labels to device: " << q << " of bytes size " << sizeof(T)*n/n_gpu*d << std::endl;
#else
        thrust::copy(&srcdata[q*n/n_gpu*d],&srcdata[(q+1)*n/n_gpu*d],data[q]->begin());
        //        thrust::copy(&srclabels[q*n/n_gpu],&srclabels[(q+1)*n/n_gpu],labels[q]->begin()); // why 
        random_labels(*labels[q], n/n_gpu, k); // why only portion of lables set?
#endif
      }
#else
      for (int q = 0; q < n_gpu; q++) {
        random_data<T>(*data[q], n/n_gpu, d);
        //        random_labels(*labels[q], n/n_gpu*d, k);
        random_labels(*labels[q], n/n_gpu, k); // why only portion of lables set?
      }
#endif

      double t0 = timer<double>();
      kmeans::kmeans<T>(n,d,k,data,labels,centroids,distances,n_gpu,max_iterations,init_from_labels,threshold);
      double time = static_cast<double>(timer<double>() - t0);
      std::cout << "  Time: " << time << " s" << std::endl;

      // copy result of centroids (sitting entirely on each device) back to host
      thrust::host_vector<T> *ctr = new thrust::host_vector<T>(*centroids[0]);
      //      cudaMemcpy(ctr->data().get(), centroids[0]->data().get(), sizeof(T)*k*d, cudaMemcpyDeviceToHost);
      *res = ctr->data();
      
      for (int q = 0; q < n_gpu; q++) {
        delete(data[q]);
        delete(labels[q]);
        //        delete(centroids[q]);
        //        delete(distances[q]);
      }

      return 0;
    }
    template int
    makePtr_dense<float>(int n_gpu, size_t rows, size_t cols, const char ord, int k, int max_iterations, float threshold, const float *srcdata, const int *srclabels, void **a);


// Explicit template instantiation.
//#if !defined(H2OAIGLM_DOUBLE) || H2OAIGLM_DOUBLE==1
//template class H2OAIKMeans<double>;
//#endif

#if !defined(H2OAIGLM_SINGLE) || H2OAIGLM_SINGLE==1
    template class H2OAIKMeans<float>;
#endif

}  // namespace h2oaikmeans

#ifdef __cplusplus
extern "C" {
#endif

  int make_ptr_float_kmeans(int n_gpu, size_t mTrain, size_t n, const char ord, int k, int max_iterations, float threshold, const float* srcdata, const int* srclabels, void** res) {
    return h2oaikmeans::makePtr_dense<float>(n_gpu, mTrain, n, ord, k, max_iterations, threshold, srcdata, srclabels, res);
}

#ifdef __cplusplus
}
#endif
