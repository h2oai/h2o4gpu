/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include <unistd.h>
#include "h2o4gpukmeans.h"
#include "kmeans_impl.h"
#include "kmeans_general.h"
#include "kmeans_h2o4gpu.h"
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include <csignal>
#include "../../common/utils.h"
#include <math.h>

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
  if (ord == 'c') {
    thrust::host_vector<T> host_array(npergpu * d);
    log_debug(verbose, "Copy data COL ORDER -> ROW ORDER");

    int indexi, indexj;
    for (int i = 0; i < npergpu * d; i++) {
      indexi = i % d; // col
      indexj = i / d + q * npergpu; // row (shifted by which gpu)
      host_array[i] = srcdata[indexi * n + indexj];
    }
    array = host_array;
  } else {
    log_debug(verbose, "Copy data ROW ORDER not changed");
    thrust::host_vector<T> host_array(srcdata + q * npergpu * d, srcdata + q * npergpu * d + npergpu * d);
    array = host_array;
  }
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

template<typename T>
int kmeans_fit(int verbose, int seed, int gpu_idtry, int n_gputry,
               size_t rows, size_t cols, const char ord,
               int k, int max_iterations, int init_from_data,
               T threshold,
               const T *srcdata, void **pred_centroids, void **pred_labels);

template<typename T>
int pick_point_idx_weighted(
    int seed,
    std::vector<T> *data,
    thrust::host_vector<T> weights) {
  T weighted_sum = 0;

  for(int i = 0; i < weights.size(); i++) {
    if(data) {
      weighted_sum += (data->data()[i] * weights.data()[i]);
    } else {
      weighted_sum += weights.data()[i];
    }
  }

  T best_prob = 0.0;
  int best_prob_idx = 0;

  std::mt19937 mt(seed);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  int i = 0;
  for(i = 0; i <= weights.size(); i++) {
    if(weights.size() == i) {
      break;
    }

    T prob_threshold = (T) dist(mt);

    T data_val = weights.data()[i];
    if (data) {
      data_val *= data->data()[i];
    }

    T prob_x = (data_val / weighted_sum);

    if(prob_x > prob_threshold) {
      break;
    }

    if (prob_x >= best_prob) {
      best_prob = prob_x;
      best_prob_idx = i;
    }
  }

  return weights.size() == i ? best_prob_idx : i;
}

/**
 * Copies cols records, starting at position idx*cols from data to centroids. Removes them afterwards from data.
 * Removes record from weights at position idx.
 * @tparam T
 * @param idx
 * @param cols
 * @param data
 * @param weights
 * @param centroids
 */
template<typename T>
void add_centroid(int idx, int cols,
                  thrust::host_vector<T> &data,
                  thrust::host_vector<T> &weights,
                  thrust::host_vector<T> &centroids) {
  for (int i = 0; i < cols; i++) {
    centroids.push_back(data[idx * cols + i]);
  }
  for (int i = cols - 1; i >= 0; i--) {
    data.erase(data.begin() + idx * cols + i);
  }
  weights.erase(weights.begin() + idx);
}

/**
 * K-Means++ algorithm
 * @tparam T
 * @param seed
 * @param data
 * @param weights
 * @param k
 * @param cols
 * @param centroids
 */
template<typename T>
void kmeans_plus_plus(
    int verbose,
    int seed,
    thrust::host_vector<T> data,
    thrust::host_vector<T> weights,
    int k,
    int cols,
    thrust::host_vector<T> &centroids) {

  int centroid_idx = pick_point_idx_weighted(
      seed,
      (std::vector<T> *) NULL,
      weights
  );

  add_centroid(centroid_idx, cols, data, weights, centroids);

  log_verbose(verbose, "KMeans++ - Allocating memory %d | %d | %d", data.size(), cols, centroids.size());

  std::vector<T> best_pairwise_distances(data.size() / cols); // one for each row in data
  std::vector<T> std_data(data.begin(), data.end());
  std::vector<T> std_centroids(centroids.begin(), centroids.end());

  compute_distances(std_data,
                    std_centroids,
                    best_pairwise_distances,
                    data.size() / cols, cols, 1);

  for (int iter = 0; iter < k - 1; iter++) {
    log_verbose(verbose, "KMeans++ - Iteraton %d/%d.", iter, k-1);

    centroid_idx = pick_point_idx_weighted(
        seed,
        &best_pairwise_distances,
        weights
    );

    add_centroid(centroid_idx, cols, data, weights, centroids);

    best_pairwise_distances.erase(best_pairwise_distances.begin() + centroid_idx);

    // TODO necessary?
    std_data = std::vector<T>(data.begin(), data.end());
    std_centroids = std::vector<T>(centroids.begin() + cols * (iter + 1), centroids.end());

    int centroids_nr = std_centroids.size() / cols;
    std::vector<T> curr_pairwise_distances( centroids_nr * (std_data.size() / cols));

    compute_distances(std_data,
                      std_centroids,
                      curr_pairwise_distances,
                      std_data.size() / cols, cols, 1);

    for (int i = 0; i < curr_pairwise_distances.size(); i++) {
      best_pairwise_distances[i] = std::min(curr_pairwise_distances[i], best_pairwise_distances[i]);
    }
  }
}

/**
 * K-Means|| initialization method implementation as described in "Scalable K-Means++".
 *
 * This is a probabilistic method, which tries to choose points as much spread out as possible as centroids.
 *
 * In case it finds more than k centroids a K-Means++ algorithm is ran on potential centroids to pick k best suited ones.
 *
 * http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
 *
 * @tparam T
 * @param verbose
 * @param seed
 * @param ord
 * @param data
 * @param data_dots
 * @param centroids
 * @param rows
 * @param cols
 * @param k
 * @param num_gpu
 * @param threshold
 */
template<typename T>
thrust::host_vector<T> kmeans_parallel(int verbose, int seed, const char ord,
                     thrust::device_vector<T> **data,
                     thrust::device_vector<T> **data_dots,
                     size_t rows, int cols, int k, int num_gpu, T threshold) {
  if (seed < 0) {
    std::random_device rd;
    int seed = rd();
  }

  size_t rows_per_gpu = rows / num_gpu;

  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, rows - 1);

  // Find the position (GPU idx and idx on that GPU) of the initial centroid
  int first_center = dis(gen);
  int first_center_idx = first_center % rows_per_gpu;
  int first_center_gpu = first_center / rows_per_gpu;

  log_verbose(verbose, "KMeans|| - Initial centroid %d on GPU %d.", first_center_idx, first_center_gpu);

  // Copies the initial centroid to potential centroids vector. That vector will store all potential centroids found
  // in the previous iteration.
  thrust::host_vector<T> h_potential_centroids(cols);
  std::vector<thrust::host_vector<T>> h_potential_centroids_per_gpu(num_gpu);

  CUDACHECK(cudaSetDevice(first_center_gpu));

  thrust::copy(
      (*data[first_center_gpu]).begin() + first_center_idx * cols,
      (*data[first_center_gpu]).begin() + (first_center_idx + 1) * cols,
      h_potential_centroids.begin()
  );

  thrust::host_vector<T> h_all_potential_centroids = h_potential_centroids;

  // Initial the cost-to-potential-centroids and cost-to-closest-potential-centroid matrices. Initial cost is +infinity
  std::vector<thrust::device_vector<T>> d_min_costs(num_gpu);
  for (int q = 0; q < num_gpu; q++) {
    CUDACHECK(cudaSetDevice(q));
    d_min_costs[q].resize(rows_per_gpu);
    thrust::fill(d_min_costs[q].begin(), d_min_costs[q].end(), std::numeric_limits<T>::max());
  }

  double t0 = timer<double>();

  // The original white paper claims 8 should be enough
  int max_iter = std::min(8, (int)(2 + log(k)) );
  for (int counter = 0; counter < max_iter; counter++) {
    log_verbose(verbose, "KMeans|| - Iteration %d.", counter);
    T total_min_cost = 0.0;

    int new_potential_centroids = 0;
    for (int i = 0; i < num_gpu; i++) {
      CUDACHECK(cudaSetDevice(i));

      thrust::device_vector<T> d_potential_centroids = h_potential_centroids;

      int potential_k_rows = d_potential_centroids.size() / cols;

      // Compute all the costs to each potential centroid from previous iteration
      thrust::device_vector<T> centroid_dots(potential_k_rows);

      // Get info about available memory
      // This part of the algo can be very memory consuming
      // We might need to batch it
      // TODO not using batch_calculate_distances because nvcc is complaining about using the __device__
      // lambda inside a lambda
      size_t free_byte;
      size_t total_byte;
      CUDACHECK(cudaMemGetInfo( &free_byte, &total_byte ));
      free_byte *= 0.8;

      size_t required_byte = rows_per_gpu * potential_k_rows * sizeof(T);

      size_t runs = std::ceil( required_byte / (double)free_byte );
      size_t offset = 0;
      size_t rows_per_run = rows_per_gpu / runs;
      thrust::device_vector<T> d_all_costs(rows_per_run * potential_k_rows);
      for(int run = 0; run < runs; run++) {
        if( run + 1 == runs ) {
          rows_per_run = rows_per_gpu % rows_per_run;
        }

        thrust::fill_n(d_all_costs.begin(), d_all_costs.size(), (T)0.0);

        kmeans::detail::calculate_distances(verbose, 0, rows_per_run, cols, potential_k_rows,
                                            *data[i], offset,
                                            d_potential_centroids,
                                            *data_dots[i],
                                            centroid_dots,
                                            d_all_costs);

        // Find the closest potential center cost for each row
        auto min_cost_counter = thrust::make_counting_iterator(0);
        auto all_costs_ptr = thrust::raw_pointer_cast(d_all_costs.data());
        auto min_costs_ptr = thrust::raw_pointer_cast(d_min_costs[i].data() + offset);
        T max = std::numeric_limits<T>::max();
        thrust::for_each(min_cost_counter, min_cost_counter + rows_per_run, [=]
        __device__(int idx){
          T best = max;
          for (int j = 0; j < potential_k_rows; j++) {
            best = min(best, std::abs(all_costs_ptr[j * rows_per_run + idx]));
          }

          min_costs_ptr[idx] = min(min_costs_ptr[idx], best);
        });
        offset += rows_per_run;
      }

      CUDACHECK(cudaDeviceSynchronize());

      total_min_cost += thrust::reduce(
          d_min_costs[i].begin(),
          d_min_costs[i].end()
      );

    }

    if(total_min_cost == (T) 0.0) {
      continue;
    }

    std::set<int> copy_from_gpus;
    for (int i = 0; i < num_gpu; i++) {
      CUDACHECK(cudaSetDevice(i));

      // Count how many potential centroids there are using probabilities
      // The further the row is from the closest cluster center the higher the probability
      auto pot_cent_filter_counter = thrust::make_counting_iterator(0);
      auto min_costs_ptr = thrust::raw_pointer_cast(d_min_costs[i].data());
      int pot_cent_num = thrust::count_if(
          pot_cent_filter_counter,
          pot_cent_filter_counter + rows_per_gpu, [=]__device__(int idx){
            thrust::default_random_engine rng(seed);
            thrust::uniform_real_distribution<> dist(0.0, 1.0);
            int device;
            cudaGetDevice(&device);
            rng.discard(idx + device * rows_per_gpu);
            T prob_threshold = (T) dist(rng);

            T prob_x = (( 2.0 * k * min_costs_ptr[idx]) / total_min_cost);

            return prob_x > prob_threshold;
          }
      );

      log_debug(verbose, "KMeans|| - Potential centroids on GPU %d = %d.", i, pot_cent_num);

      if (pot_cent_num > 0) {
        copy_from_gpus.insert(i);

        // Copy all potential cluster centers
        thrust::device_vector<T> d_new_potential_centroids(pot_cent_num * cols);

        auto range = thrust::make_counting_iterator(0);
        thrust::copy_if(
            (*data[i]).begin(), (*data[i]).end(), range,
            d_new_potential_centroids.begin(), [=] __device__(int idx){
              int row = idx / cols;
              thrust::default_random_engine rng(seed);
              thrust::uniform_real_distribution<> dist(0.0, 1.0);
              int device;
              cudaGetDevice(&device);
              rng.discard(row + device * rows_per_gpu);
              T prob_threshold = (T) dist(rng);

              T prob_x = (( 2.0 * k * min_costs_ptr[row]) / total_min_cost);

              return prob_x > prob_threshold;
        });

        h_potential_centroids_per_gpu[i].clear();
        h_potential_centroids_per_gpu[i].resize(d_new_potential_centroids.size());

        new_potential_centroids += d_new_potential_centroids.size();

        thrust::copy(
            d_new_potential_centroids.begin(),
            d_new_potential_centroids.end(),
            h_potential_centroids_per_gpu[i].begin()
        );

      }

    }

    log_verbose(verbose, "KMeans|| - New potential centroids %d.", new_potential_centroids);

    // Gather potential cluster centers from all GPUs
    if (new_potential_centroids > 0) {
      h_potential_centroids.clear();
      h_potential_centroids.resize(new_potential_centroids);

      int old_pot_centroids_size = h_all_potential_centroids.size();
      h_all_potential_centroids.resize(old_pot_centroids_size + new_potential_centroids);

      int offset = 0;
      for (int i = 0; i < num_gpu; i++) {
        if(copy_from_gpus.find(i) != copy_from_gpus.end()) {
          thrust::copy(
              h_potential_centroids_per_gpu[i].begin(),
              h_potential_centroids_per_gpu[i].end(),
              h_potential_centroids.begin() + offset
          );
          offset += h_potential_centroids_per_gpu[i].size();
        }
      }

      thrust::copy(
          h_potential_centroids.begin(),
          h_potential_centroids.end(),
          h_all_potential_centroids.begin() + old_pot_centroids_size
      );
    }
  }

  double timeloop = static_cast<double>(timer<double>() - t0);

  thrust::host_vector<T> final_centroids(0);
  int potential_centroids_num = h_all_potential_centroids.size() / cols;

  if (potential_centroids_num <= k) {
    final_centroids.resize(k * cols);
    thrust::copy(
        h_all_potential_centroids.begin(),
        h_all_potential_centroids.end(),
        final_centroids.begin()
    );
    // TODO what if potential_centroids_num < k ?? we don't want 0s
  } else {
    // If we found more than k potential cluster centers we need to take only a subset
    // This is done using a weighted k-means++ method, since the set should be very small
    // it should converge very fast and is all done on the CPU.
    thrust::host_vector<T> weights(potential_centroids_num);

    double tc0 = timer<double>();

    // Weights correspond to the number of data points assigned to each potential cluster center
    count_pts_per_centroid(
        verbose, num_gpu,
        rows_per_gpu, cols,
        data, data_dots,
        h_all_potential_centroids,
        weights
    );

    double timecount = static_cast<double>(timer<double>() - tc0);

    double tkpp = timer<double>();

    kmeans_plus_plus(
        verbose,
        seed,
        h_all_potential_centroids,
        weights,
        k, cols,
        final_centroids
    );

    double timekpp = static_cast<double>(timer<double>() - tkpp);

    log_verbose(verbose, "KMeans|| - Time loop: %g Time count: %g Time kpp: %g.", timeloop, timecount, timekpp);
  }

  return final_centroids;
}

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
  thrust::device_vector<T> *data_dots[n_gpu];

  for (int q = 0; q < n_gpu; q++) {
    CUDACHECK(cudaSetDevice(dList[q]));
    data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);
    d_centroids[q] = new thrust::device_vector<T>(k * cols);
    data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);

    kmeans::detail::labels_init();
  }

  log_debug(verbose, "Number of points: %d", rows);
  log_debug(verbose, "Number of dimensions: %d", cols);
  log_debug(verbose, "Number of clusters: %d", k);
  log_debug(verbose, "Max. number of iterations: %d", max_iterations);
  log_debug(verbose, "Stopping threshold: %d", threshold);

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

    // Pre-compute the data matrix norms
    kmeans::detail::make_self_dots(rows / n_gpu, cols, *data[q], *data_dots[q]);
  }

  // Get random points as centroids
  int bytecount = cols * k * sizeof(T); // all centroids
  if (0 == init_from_data) {
    log_debug(verbose, "KMeans - Using random initialization.");

    int masterq = 0;
    CUDACHECK(cudaSetDevice(dList[masterq]));
    copy_centroids_shuffled(verbose, v, ord, *d_centroids[masterq], &srcdata[0], rows, k, cols);

    // Copy centroids to all devices
    std::vector < cudaStream_t * > streams;
    streams.resize(n_gpu);
    for (int q = 0; q < n_gpu; q++) {
      if (q == masterq) continue;

      CUDACHECK(cudaSetDevice(dList[q]));
      if (verbose > 0) {
        std::cout << "Copying centroid data to device: " << dList[q] << std::endl;
      }

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
  } else if (1 == init_from_data) { // kmeans||
    log_debug(verbose, "KMeans - Using K-Means|| initialization.");

    thrust::host_vector<T> final_centroids = kmeans_parallel(verbose, seed, ord, data, data_dots, rows, cols, k, n_gpu, threshold);

    for (int q = 0; q < n_gpu; q++) {
      CUDACHECK(cudaSetDevice(dList[q]));
      cudaMemcpy(
          thrust::raw_pointer_cast(&(*d_centroids[q])[0]),
          thrust::raw_pointer_cast(&final_centroids[0]),
          bytecount,
          cudaMemcpyHostToDevice);
    }

  }

  for (int q = 0; q < n_gpu; q++) {
    CUDACHECK(cudaSetDevice(dList[q]));
    labels[q] = new thrust::device_vector<int>(rows / n_gpu);
  }

  double timetransfer = static_cast<double>(timer<double>() - t0t);

  double t0 = timer<double>();

  int iter = kmeans::kmeans<T>(verbose, &flaggpu, rows, cols, k, data, labels, d_centroids, data_dots,
                    dList, n_gpu, max_iterations, threshold, true);

  if (iter < 0) {
    log_error(verbose, "KMeans algorithm failed.");
    return iter;
  }

  double timefit = static_cast<double>(timer<double>() - t0);

  double t1 = timer<double>();

  // copy result of centroids (sitting entirely on each device) back to host
  thrust::host_vector<T> *ctr = new thrust::host_vector<T>(*d_centroids[0]);
  // TODO FIXME: When do delete this ctr memory?
  // cudaMemcpy(ctr->data().get(), centroids[0]->data().get(), sizeof(T)*k*d, cudaMemcpyDeviceToHost);
  *pred_centroids = ctr->data();

  // copy assigned labels
  thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(rows);
  int offset = 0;
  for (int q = 0; q < n_gpu; q++) {
    h_labels->insert(h_labels->begin() + offset, labels[q]->begin(), labels[q]->end());
    offset += labels[q]->size();
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
    CUDACHECK(cudaSetDevice(dList[q]));
    delete (data[q]);
    delete (labels[q]);
    delete (d_centroids[q]);
    delete (data_dots[q]);
    kmeans::detail::labels_close();
  }

  double timecleanup = static_cast<double>(timer<double>() - t1);

  if (verbose) {
    std::cout << "  Time fit: " << timefit << " s" << std::endl;
    fprintf(stderr, "Timetransfer: %g Timefit: %g Timecleanup: %g\n", timetransfer, timefit, timecleanup);
    fflush(stderr);
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
  thrust::device_vector<T> *d_centroids[n_gpu];
  thrust::device_vector<T> *data_dots[n_gpu];
  thrust::device_vector<T> *centroid_dots[n_gpu];
  thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(0);

  for (int q = 0; q < n_gpu; q++) {
    CUDACHECK(cudaSetDevice(dList[q]));
    kmeans::detail::labels_init();

    data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);
    centroid_dots[q] = new thrust::device_vector<T>(k);

    d_centroids[q] = new thrust::device_vector<T>(k * cols);
    d_data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);

    copy_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, cols);

    copy_data(verbose, ord, *d_data[q], &srcdata[0], q, rows, rows / n_gpu, cols);

    kmeans::detail::make_self_dots(rows / n_gpu, cols, *d_data[q], *data_dots[q]);

    thrust::device_vector<int> d_labels(rows / n_gpu);

    kmeans::detail::batch_calculate_distances(verbose, q, rows / n_gpu, cols, k,
                                      *d_data[q], *d_centroids[q], *data_dots[q], *centroid_dots[q],
                                      [&](int n, size_t offset, thrust::device_vector<T> &pairwise_distances) {
                                        kmeans::detail::relabel(n, k, pairwise_distances, d_labels, offset);
                                      }
    );

    h_labels->insert(h_labels->end(), d_labels.begin(), d_labels.end());
  }

  *pred_labels = h_labels->data();

  for (int q = 0; q < n_gpu; q++) {
    safe_cuda(cudaSetDevice(dList[q]));
    kmeans::detail::labels_close();
    delete (data_dots[q]);
    delete (centroid_dots[q]);
    delete (d_centroids[q]);
    delete (d_data[q]);
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

    // TODO batch this
    kmeans::detail::calculate_distances(verbose, q, rows / n_gpu, cols, k,
                                        *d_data[q], 0, *d_centroids[q], *data_dots[q],
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
