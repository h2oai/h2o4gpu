/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include <unistd.h>
#include "h2o4gpukmeans.h"
#include "kmeans_impl.h"
#include "kmeans_general.h"
#include <random>
#include <algorithm>
#include <vector>
#include <csignal>

#define CUDACHECK(cmd) do {                           \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));\
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

/**
 * METHODS FOR DATA COPYING AND GENERATION
 */

template<typename T>
void random_data(int verbose, thrust::device_vector<T> &array, int m, int n) {
  thrust::host_vector<T> host_array(m * n);
  for (int i = 0; i < m * n; i++) {
    host_array[i] = (T) rand() / (T) RAND_MAX;
  }
  array = host_array;
}

/**
 * Copies data from srcdata to array
 * @tparam T
 * @param verbose Logging level
 * @param ord Column on row order of data
 * @param array Destination array
 * @param srcdata Source data
 * @param q Shard number (from 0 to n_gpu)
 * @param n
 * @param npergpu
 * @param d
 */
template<typename T>
void copy_data(int verbose, const char ord, thrust::device_vector<T> &array, const T *srcdata,
               int q, int n, int npergpu, int d) {
  thrust::host_vector<T> host_array(npergpu * d);
  if (ord == 'c') {
    log_debug(verbose, "Copy data COL ORDER -> ROW ORDER");

    int indexi, indexj;
    for (int i = 0; i < npergpu * d; i++) {
      indexi = i % d; // col
      indexj = i / d + q * npergpu; // row (shifted by which gpu)
      host_array[i] = srcdata[indexi * n + indexj];
    }
  } else {
    log_debug(verbose, "Copy data ROW ORDER not changed");

    for (int i = 0; i < npergpu * d; i++) {
      host_array[i] = srcdata[q * npergpu * d + i]; // shift by which gpu
    }
  }
  array = host_array;
}

/**
 * Like copy_data but shuffles the data according to mapping from v
 * @tparam T
 * @param verbose
 * @param v
 * @param ord
 * @param array
 * @param srcdata
 * @param q
 * @param n
 * @param npergpu
 * @param d
 */
template<typename T>
void copy_data_shuffled(int verbose, std::vector<int> v, const char ord, thrust::device_vector<T> &array,
                        const T *srcdata, int q, int n, int npergpu, int d) {
  thrust::host_vector<T> host_array(npergpu * d);
  if (ord == 'c') {
    log_debug(verbose, "Copy data shuffle COL ORDER -> ROW ORDER");

    for (int i = 0; i < npergpu; i++) {
      for (int j = 0; j < d; j++) {
        host_array[i * d + j] = srcdata[v[q * npergpu + i] + j * n]; // shift by which gpu
      }
    }
  } else {
    log_debug(verbose, "Copy data shuffle ROW ORDER not changed");

    for (int i = 0; i < npergpu; i++) {
      for (int j = 0; j < d; j++) {
        host_array[i * d + j] = srcdata[v[q * npergpu + i] * d + j]; // shift by which gpu
      }
    }
  }
  array = host_array;
}

template<typename T>
void copy_centroids_shuffled(int verbose, std::vector<int> v, const char ord, thrust::device_vector<T> &array,
                             const T *srcdata, int n, int k, int d) {
  copy_data_shuffled(verbose, v, ord, array, srcdata, 0, n, k, d);
}

/**
 * Copies centroids from initial training set randomly.
 * @tparam T
 * @param verbose
 * @param seed
 * @param ord
 * @param array
 * @param srcdata
 * @param q
 * @param n
 * @param npergpu
 * @param d
 * @param k
 */
template<typename T>
void random_centroids(int verbose, int seed, const char ord,
                      thrust::device_vector<T> &array, const T *srcdata,
                      int q, int n, int npergpu, int d, int k) {
  thrust::host_vector<T> host_array(k * d);
  if (seed < 0) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, n - 1); // random i in range from 0..n-1 (i.e. only 1 gpu gets centroids)

  if (ord == 'c') {
    log_debug(verbose, "Random centroids COL ORDER -> ROW ORDER");
    for (int i = 0; i < k; i++) { // rows
      int reali = dis(gen); // + q*npergpu; // row sampled (called indexj above)
      for (int j = 0; j < d; j++) { // cols
        host_array[i * d + j] = srcdata[reali + j * n];
      }
    }
  } else {
    log_debug(verbose, "Random centroids ROW ORDER not changed");
    for (int i = 0; i < k; i++) { // rows
      int reali = dis(gen); // + q*npergpu ; // row sampled
      for (int j = 0; j < d; j++) { // cols
        host_array[i * d + j] = srcdata[reali * d + j];
      }
    }
  }
  array = host_array;
}

/**
 * KMEANS METHODS FIT, PREDICT, TRANSFORM
 */

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace h2o4gpukmeans {
volatile std::atomic_int flaggpu(0);

inline void my_function_gpu(int sig) { // can be called asynchronously
  fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
  flaggpu = 1;
}

std::vector<int> kmeans_init(int verbose, int *final_n_gpu, int n_gputry, int gpu_idtry, int rows) {
  if (rows > std::numeric_limits<int>::max()) {
    fprintf(stderr, "rows > %d not implemented\n", std::numeric_limits<int>::max());
    fflush(stderr);
    exit(0);
  }

  std::signal(SIGINT, my_function_gpu);
  std::signal(SIGTERM, my_function_gpu);

  // no more gpus than visible gpus
  int n_gpuvis;
  cudaGetDeviceCount(&n_gpuvis);
  int n_gpu = std::min(n_gpuvis, n_gputry);

  // no more than rows
  n_gpu = std::min(n_gpu, rows);

  if (verbose) {
    std::cout << n_gpu << " gpus." << std::endl;
  }

  int gpu_id = gpu_idtry % n_gpuvis;

  // setup GPU list to use
  std::vector<int> dList(n_gpu);
  for (int idx = 0; idx < n_gpu; idx++) {
    int device_idx = (gpu_id + idx) % n_gpuvis;
    dList[idx] = device_idx;
  }

  *final_n_gpu = n_gpu;
  return dList;
}

template<typename T>
H2O4GPUKMeans<T>::H2O4GPUKMeans(const T *A, int k, int n, int d) {
  _A = A;
  _k = k;
  _n = n;
  _d = d;
}

template<typename T>
int kmeans_fit(int verbose, int seed, int gpu_idtry, int n_gputry,
               size_t rows, size_t cols, const char ord,
               int k, int max_iterations, int init_from_data,
               T threshold,
               const T *srcdata, void **pred_centroids, void **pred_labels) {
  log_debug(verbose, "KMeans - Start fitting");

  // init random seed if use the C function rand()
  if (seed >= 0) {
    srand(seed);
  } else {
    srand(unsigned(time(NULL)));
  }

  // no more clusters than rows
  if (k > rows) {
    k = static_cast<int>(rows);
    fprintf(stderr, "Number of clusters adjusted to be equal to number of rows.\n");
    fflush(stderr);
  }

  int n_gpu;
  std::vector<int> dList = kmeans_init(verbose, &n_gpu, n_gputry, gpu_idtry, rows);

  double t0t = timer<double>();
  thrust::device_vector<T> *data[n_gpu];
  thrust::device_vector<int> *labels[n_gpu];
  thrust::device_vector<T> *d_centroids[n_gpu];
  thrust::device_vector<T> *distances[n_gpu];

  log_debug(verbose, "KMeans - Before allocation");

  for (int q = 0; q < n_gpu; q++) {
    CUDACHECK(cudaSetDevice(dList[q]));
    data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);
    labels[q] = new thrust::device_vector<int>(rows / n_gpu);
    d_centroids[q] = new thrust::device_vector<T>(k * cols);
    distances[q] = new thrust::device_vector<T>(rows / n_gpu);
  }

  if (verbose >= H2O4GPU_LOG_INFO) {
    std::cout << "Number of points: " << rows << std::endl;
    std::cout << "Number of dimensions: " << cols << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;
    std::cout << "Max. number of iterations: " << max_iterations << std::endl;
    std::cout << "Stopping threshold: " << threshold << std::endl;
  }

  std::vector<int> v(rows);
  std::iota(std::begin(v), std::end(v), 0); // Fill with 0, 1, ..., rows.

  if (seed >= 0) {
    std::shuffle(v.begin(), v.end(), std::default_random_engine(seed));
  } else {
    std::random_shuffle(v.begin(), v.end());
  }

  // Copy the data to devices
  for (int q = 0; q < n_gpu; q++) {
    CUDACHECK(cudaSetDevice(dList[q]));
    if (verbose) { std::cout << "Copying data to device: " << dList[q] << std::endl; }

    copy_data(verbose, ord, *data[q], &srcdata[0], q, rows, rows / n_gpu, cols);
  }

  // Get random points as centroids
  int masterq = 0;
  CUDACHECK(cudaSetDevice(dList[masterq]));
  copy_centroids_shuffled(verbose, v, ord, *d_centroids[masterq], &srcdata[0], rows, k, cols);
  int bytecount = cols * k * sizeof(T); // all centroids

  // Copy centroids to all devices
  std::vector < cudaStream_t * > streams;
  streams.resize(n_gpu);
  for (int q = 0; q < n_gpu; q++) {
    if (q == masterq) continue;

    CUDACHECK(cudaSetDevice(dList[q]));
    std::cout << "Copying centroid data to device: " << dList[q] << std::endl;

    streams[q] = reinterpret_cast<cudaStream_t *>(malloc(sizeof(cudaStream_t)));
    cudaStreamCreate(streams[q]);
    cudaMemcpyPeerAsync(thrust::raw_pointer_cast(&(*d_centroids[q])[0]),
                        dList[q],
                        thrust::raw_pointer_cast(&(*d_centroids[masterq])[0]),
                        dList[masterq],
                        bytecount,
                        *(streams[q]));
  }
  for (int q = 0; q < n_gpu; q++) {
    if (q == masterq) continue;
    cudaSetDevice(dList[q]);
    cudaStreamDestroy(*(streams[q]));
#if(DEBUGKMEANS)
    thrust::host_vector<T> h_centroidq=*d_centroids[q];
    for(int ii=0;ii<k*d;ii++){
        fprintf(stderr,"q=%d initcent[%d]=%g\n",q,ii,h_centroidq[ii]); fflush(stderr);
    }
#endif
  }

  double timetransfer = static_cast<double>(timer<double>() - t0t);

  log_debug(verbose, "KMeans - Before kmeans() call");

  double t0 = timer<double>();

  int status = kmeans::kmeans<T>(verbose, &flaggpu, rows, cols, k, data, labels, d_centroids, distances, dList, n_gpu,
                                 max_iterations, threshold, true);

  if (status) {
    fprintf(stderr, "KMeans status was %d\n", status);
    fflush(stderr);
    return (status);
  }

  double timefit = static_cast<double>(timer<double>() - t0);

  if (verbose) {
    std::cout << "  Time fit: " << timefit << " s" << std::endl;
    fprintf(stderr, "Timetransfer: %g Timefit: %g\n", timetransfer, timefit);
    fflush(stderr);
  }

  // copy result of centroids (sitting entirely on each device) back to host
  thrust::host_vector<T> *ctr = new thrust::host_vector<T>(*d_centroids[0]);
  // TODO FIXME: When do delete this ctr memory?
  // cudaMemcpy(ctr->data().get(), centroids[0]->data().get(), sizeof(T)*k*d, cudaMemcpyDeviceToHost);
  *pred_centroids = ctr->data();

  // copy assigned labels
  thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(0);
  for (int q = 0; q < n_gpu; q++) {
    h_labels->insert(h_labels->end(), labels[q]->begin(), labels[q]->end());
  }

  *pred_labels = h_labels->data();

  // debug
  if (verbose >= H2O4GPU_LOG_VERBOSE) {
    for (unsigned int ii = 0; ii < k; ii++) {
      fprintf(stderr, "ii=%d of k=%d ", ii, k);
      for (unsigned int jj = 0; jj < cols; jj++) {
        fprintf(stderr, "%g ", (*ctr)[cols * ii + jj]);
      }
      fprintf(stderr, "\n");
      fflush(stderr);
    }
  }

  for (int q = 0; q < n_gpu; q++) {
    delete (data[q]);
    delete (labels[q]);
    delete (d_centroids[q]);
    delete (distances[q]);
  }

  return 0;
}

template<typename T>
int kmeans_predict(int verbose, int gpu_idtry, int n_gputry,
                   size_t rows, size_t cols,
                   const char ord, int k,
                   const T *srcdata, const T *centroids, void **pred_labels) {
  // Print centroids
  if (verbose >= H2O4GPU_LOG_VERBOSE) {
    std::cout << std::endl;
    for (int i = 0; i < cols * k; i++) {
      std::cout << centroids[i] << " ";
      if (i % cols == 1) {
        std::cout << std::endl;
      }
    }
  }

  int n_gpu;
  std::vector<int> dList = kmeans_init(verbose, &n_gpu, n_gputry, gpu_idtry, rows);

  thrust::device_vector<T> *d_data[n_gpu];
  thrust::device_vector<int> *d_labels[n_gpu];
  thrust::device_vector<T> *d_centroids[n_gpu];
  thrust::device_vector<T> *pairwise_distances[n_gpu];
  thrust::device_vector<T> *data_dots[n_gpu];
  thrust::device_vector<T> *centroid_dots[n_gpu];
  thrust::device_vector<T> *distances[n_gpu];
  int *d_changes[n_gpu];

  for (int q = 0; q < n_gpun_gpu; q++) {
    // TODO everything from here till "distances[q]" is exactly the same as in transform
    CUDACHECK(cudaSetDevice(dList[q]));
    kmeans::detail::labels_init();

    data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);
    centroid_dots[q] = new thrust::device_vector<T>(k);
    pairwise_distances[q] = new thrust::device_vector<T>(rows / n_gpu * k);

    d_centroids[q] = new thrust::device_vector<T>(k * cols);
    d_data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);

    copy_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, cols);

    copy_data(verbose, ord, *d_data[q], &srcdata[0], q, rows, rows / n_gpu, cols);

    kmeans::detail::make_self_dots(rows / n_gpu, cols, *d_data[q], *data_dots[q]);

    kmeans::detail::calculate_distances(verbose, q, rows / n_gpu, cols, k,
                                        *d_data[q], *d_centroids[q], *data_dots[q],
                                        *centroid_dots[q], *pairwise_distances[q]);

    distances[q] = new thrust::device_vector<T>(rows / n_gpu);
    d_labels[q] = new thrust::device_vector<int>(rows / n_gpu);
    cudaMalloc(&d_changes[q], sizeof(int));

    kmeans::detail::relabel(rows / n_gpu, k, *pairwise_distances[q], *d_labels[q], *distances[q], d_changes[q]);
  }

  // Move the resulting labels into host memory from all devices
  thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(0);
  for (int q = 0; q < n_gpu; q++) {
    h_labels->insert(h_labels->end(), d_labels[q]->begin(), d_labels[q]->end());
  }

  *pred_labels = h_labels->data();

  for (int q = 0; q < n_gpu; q++) {
    safe_cuda(cudaSetDevice(dList[q]));
    safe_cuda(cudaFree(d_changes[q]));
    kmeans::detail::labels_close();
    delete (d_labels[q]);
    delete (pairwise_distances[q]);
    delete (data_dots[q]);
    delete (centroid_dots[q]);
    delete (d_centroids[q]);
    delete (d_data[q]);
    delete (distances[q]);
  }

  return 0;
}

template<typename T>
int kmeans_transform(int verbose,
                     int gpu_idtry, int n_gputry,
                     size_t rows, size_t cols, const char ord, int k,
                     const T *srcdata, const T *centroids,
                     void **preds) {
  // Print centroids
  if (verbose >= H2O4GPU_LOG_VERBOSE) {
    std::cout << std::endl;
    for (int i = 0; i < cols * k; i++) {
      std::cout << centroids[i] << " ";
      if (i % cols == 1) {
        std::cout << std::endl;
      }
    }
  }

  int n_gpu;
  std::vector<int> dList = kmeans_init(verbose, &n_gpu, n_gputry, gpu_idtry, rows);

  thrust::device_vector<T> *d_data[n_gpu];
  thrust::device_vector<T> *d_centroids[n_gpu];
  thrust::device_vector<T> *d_pairwise_distances[n_gpu];
  thrust::device_vector<T> *data_dots[n_gpu];
  thrust::device_vector<T> *centroid_dots[n_gpu];

  for (int q = 0; q < n_gpu; q++) {
    CUDACHECK(cudaSetDevice(dList[q]));
    kmeans::detail::labels_init();

    data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);
    centroid_dots[q] = new thrust::device_vector<T>(k);
    d_pairwise_distances[q] = new thrust::device_vector<T>(rows / n_gpu * k);

    d_centroids[q] = new thrust::device_vector<T>(k * cols);
    d_data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);

    copy_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, cols);

    copy_data(verbose, ord, *d_data[q], &srcdata[0], q, rows, rows / n_gpu, cols);

    kmeans::detail::make_self_dots(rows / n_gpu, cols, *d_data[q], *data_dots[q]);

    kmeans::detail::calculate_distances(verbose, q, rows / n_gpu, cols, k,
                                        *d_data[q], *d_centroids[q], *data_dots[q],
                                        *centroid_dots[q], *d_pairwise_distances[q]);
  }

  // Move the resulting labels into host memory from all devices
  thrust::host_vector<T> *h_pairwise_distances = new thrust::host_vector<T>(0);
  for (int q = 0; q < n_gpu; q++) {
    h_pairwise_distances->insert(h_pairwise_distances->end(),
                                 d_pairwise_distances[q]->begin(),
                                 d_pairwise_distances[q]->end());
  }
  *preds = h_pairwise_distances->data();

  // Print centroids
  if (verbose >= H2O4GPU_LOG_VERBOSE) {
    std::cout << std::endl;
    for (int i = 0; i < rows * cols; i++) {
      std::cout << h_pairwise_distances->data()[i] << " ";
      if (i % cols == 1) {
        std::cout << std::endl;
      }
    }
  }

  for (int q = 0; q < n_gpu; q++) {
    safe_cuda(cudaSetDevice(dList[q]));
    kmeans::detail::labels_close();
    delete (d_pairwise_distances[q]);
    delete (data_dots[q]);
    delete (centroid_dots[q]);
    delete (d_centroids[q]);
    delete (d_data[q]);
  }

  return 0;
}

template<typename T>
int makePtr_dense(int dopredict, int verbose, int seed, int gpu_idtry, int n_gputry, size_t rows, size_t cols,
                  const char ord, int k, int max_iterations, int init_from_data,
                  T threshold, const T *srcdata, const T *centroids,
                  void **pred_centroids, void **pred_labels) {
  if (dopredict == 0) {
    return kmeans_fit(verbose, seed, gpu_idtry, n_gputry, rows, cols,
                      ord, k, max_iterations, init_from_data, threshold,
                      srcdata, pred_centroids, pred_labels);
  } else {
    return kmeans_predict(verbose, gpu_idtry, n_gputry, rows, cols,
                          ord, k,
                          srcdata, centroids, pred_labels);
  }
}

template int
makePtr_dense<float>(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t rows, size_t cols,
                     const char ord, int k, int max_iterations, int init_from_data,
                     float threshold, const float *srcdata,
                     const float *centroids, void **pred_centroids, void **pred_labels);

template int
makePtr_dense<double>(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t rows, size_t cols,
                      const char ord, int k, int max_iterations, int init_from_data,
                      double threshold, const double *srcdata,
                      const double *centroids, void **pred_centroids, void **pred_labels);

template int kmeans_fit<float>(int verbose, int seed, int gpu_idtry, int n_gputry,
                               size_t rows, size_t cols,
                               const char ord, int k, int max_iterations,
                               int init_from_data, float threshold,
                               const float *srcdata,
                               void **pred_centroids, void **pred_labels);

template int kmeans_fit<double>(int verbose, int seed, int gpu_idtry, int n_gputry,
                                size_t rows, size_t cols,
                                const char ord, int k, int max_iterations,
                                int init_from_data, double threshold,
                                const double *srcdata,
                                void **pred_centroids, void **pred_labels);

template int kmeans_predict<float>(int verbose, int gpu_idtry, int n_gputry,
                                   size_t rows, size_t cols,
                                   const char ord, int k,
                                   const float *srcdata, const float *centroids, void **pred_labels);

template int kmeans_predict<double>(int verbose, int gpu_idtry, int n_gputry,
                                    size_t rows, size_t cols,
                                    const char ord, int k,
                                    const double *srcdata, const double *centroids, void **pred_labels);

template int kmeans_transform<float>(int verbose,
                                     int gpu_id, int n_gpu,
                                     size_t m, size_t n, const char ord, int k,
                                     const float *src_data, const float *centroids,
                                     void **preds);

template int kmeans_transform<double>(int verbose,
                                      int gpu_id, int n_gpu,
                                      size_t m, size_t n, const char ord, int k,
                                      const double *src_data, const double *centroids,
                                      void **preds);

// Explicit template instantiation.
#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE == 1

template
class H2O4GPUKMeans<double>;

#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE == 1

template
class H2O4GPUKMeans<float>;

#endif

}  // namespace h2o4gpukmeans

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Interface for other languages
 */

// Fit and Predict
int make_ptr_float_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                          const char ord, int k, int max_iterations, int init_from_data,
                          float threshold, const float *srcdata,
                          const float *centroids, void **pred_centroids, void **pred_labels) {
  return h2o4gpukmeans::makePtr_dense<float>(dopredict, verbose, seed, gpu_id, n_gpu, mTrain, n, ord, k,
                                             max_iterations, init_from_data, threshold,
                                             srcdata, centroids, pred_centroids, pred_labels);
}

int make_ptr_double_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                           const char ord, int k, int max_iterations, int init_from_data,
                           double threshold, const double *srcdata,
                           const double *centroids, void **pred_centroids, void **pred_labels) {
  return h2o4gpukmeans::makePtr_dense<double>(dopredict, verbose, seed, gpu_id, n_gpu, mTrain, n, ord, k,
                                              max_iterations, init_from_data, threshold,
                                              srcdata, centroids, pred_centroids, pred_labels);
}

// Transform
int kmeans_transform_float(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const float *src_data, const float *centroids,
                           void **preds) {
  return h2o4gpukmeans::kmeans_transform<float>(verbose, gpu_id, n_gpu, m, n, ord, k, src_data, centroids, preds);
}

int kmeans_transform_double(int verbose,
                            int gpu_id, int n_gpu,
                            size_t m, size_t n, const char ord, int k,
                            const double *src_data, const double *centroids,
                            void **preds) {
  return h2o4gpukmeans::kmeans_transform<double>(verbose, gpu_id, n_gpu, m, n, ord, k, src_data, centroids, preds);
}

#ifdef __cplusplus
}
#endif
