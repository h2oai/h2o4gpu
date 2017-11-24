/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <thrust/device_vector.h>
#include "../include/cub/cub.cuh"
#include <iostream>
#include <sstream>
#include <cublas_v2.h>
#include <cfloat>
#include "kmeans_general.h"
#include <thrust/fill.h>

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}


inline cudaError_t throw_on_cuda_error(cudaError_t code, const char *file,
                                       int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }

  return code;
}

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

inline cublasStatus_t throw_on_cublas_error(cublasStatus_t code, const char *file,
                                       int line) {


  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"cublas error: %s %s %d\n", cudaGetErrorEnum(code), file, line);
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }

  return code;
}


extern cudaStream_t cuda_stream[MAX_NGPUS];

template<unsigned int i>
extern __global__ void debugMark(){};

namespace kmeans {
  namespace detail {

    void labels_init();
    void labels_close();

    template<typename T>
      void memcpy(thrust::host_vector<T, std::allocator<T> > &H,
          thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(H.data()),
            thrust::raw_pointer_cast(D.data()),
            sizeof(T) * D.size(), cudaMemcpyDeviceToHost, cuda_stream[dev_num]));
      }

    template<typename T>
      void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D,
          thrust::host_vector<T, std::allocator<T> > &H) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(D.data()),
            thrust::raw_pointer_cast(H.data()),
            sizeof(T) * H.size(), cudaMemcpyHostToDevice, cuda_stream[dev_num]));
      }
    template<typename T>
      void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Do,
          thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Di) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(Do.data()),
            thrust::raw_pointer_cast(Di.data()),
            sizeof(T) * Di.size(), cudaMemcpyDeviceToDevice, cuda_stream[dev_num]));
      }
    template<typename T>
      void memzero(thrust::device_vector<T, thrust::device_malloc_allocator<T> >& D) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        safe_cuda(cudaMemsetAsync(thrust::raw_pointer_cast(D.data()), 0, sizeof(T)*D.size(), cuda_stream[dev_num]));
      }
    void streamsync(int dev_num);

    //n: number of points
    //d: dimensionality of points
    //data: points, laid out in row-major order (n rows, d cols)
    //dots: result vector (n rows)
    // NOTE:
    //Memory accesses in this function are uncoalesced!!
    //This is because data is in row major order
    //However, in k-means, it's called outside the optimization loop
    //on the large data array, and inside the optimization loop it's
    //called only on a small array, so it doesn't really matter.
    //If this becomes a performance limiter, transpose the data somewhere
    template<typename T>
      __global__ void self_dots(int n, int d, T* data, T* dots) {
        T accumulator = 0;
        int global_id = blockDim.x * blockIdx.x + threadIdx.x;

        if (global_id < n) {
          for (int i = 0; i < d; i++) {
            T value = data[i + global_id * d];
            accumulator += value * value;
          }
          dots[global_id] = accumulator;
        }
      }


    template<typename T>
      void make_self_dots(int n, int d, thrust::device_vector<T>& data, thrust::device_vector<T>& dots) {
        int dev_num;
#define MAX_BLOCK_THREADS0 256
        const int GRID_SIZE=(n-1)/MAX_BLOCK_THREADS0+1;
        safe_cuda(cudaGetDevice(&dev_num));
        self_dots<<<GRID_SIZE, MAX_BLOCK_THREADS0, 0, cuda_stream[dev_num]>>>(n, d, thrust::raw_pointer_cast(data.data()),
            thrust::raw_pointer_cast(dots.data()));
#if(CHECK)
        gpuErrchk( cudaGetLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif

      }

#define MAX_BLOCK_THREADS 32
    template<typename T>
      __global__ void all_dots(int n, int k, T* data_dots, T* centroid_dots, T* dots) {
        __shared__ T local_data_dots[MAX_BLOCK_THREADS];
        __shared__ T local_centroid_dots[MAX_BLOCK_THREADS];
        //        if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) printf("inside %d %d %d\n",threadIdx.x,blockIdx.x,blockDim.x);

        int data_index = threadIdx.x + blockIdx.x * blockDim.x;
        if ((data_index < n) && (threadIdx.y == 0)) {
          local_data_dots[threadIdx.x] = data_dots[data_index];
        }

        int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
        if ((centroid_index < k) && (threadIdx.y == 1)) {
          local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
        }

        __syncthreads();

        centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
        //        printf("data_index=%d centroid_index=%d\n",data_index,centroid_index);
        if ((data_index < n) && (centroid_index < k)) {
          dots[data_index + centroid_index * n] = local_data_dots[threadIdx.x] +
            local_centroid_dots[threadIdx.y];
        }
      }


    template<typename T>
      void make_all_dots(int n, int k, size_t offset, thrust::device_vector<T>& data_dots,
          thrust::device_vector<T>& centroid_dots,
          thrust::device_vector<T>& dots) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        const int BLOCK_THREADSX = MAX_BLOCK_THREADS; // BLOCK_THREADSX*BLOCK_THREADSY<=1024 on modern arch's (sm_61)
        const int BLOCK_THREADSY = MAX_BLOCK_THREADS;
        const int GRID_SIZEX=(n-1)/BLOCK_THREADSX+1; // on old arch's this has to be less than 2^16=65536
        const int GRID_SIZEY=(k-1)/BLOCK_THREADSY+1; // this has to be less than 2^16=65536
        //        printf("pre all_dots: %d %d %d %d\n",GRID_SIZEX,GRID_SIZEY,BLOCK_THREADSX,BLOCK_THREADSY); fflush(stdout);
        all_dots<<<
          dim3(GRID_SIZEX,GRID_SIZEY),
          dim3(BLOCK_THREADSX, BLOCK_THREADSY), 0,
          cuda_stream[dev_num]>>>(n, k, thrust::raw_pointer_cast(data_dots.data() + offset),
              thrust::raw_pointer_cast(centroid_dots.data()),
              thrust::raw_pointer_cast(dots.data()));
#if(CHECK)
        gpuErrchk( cudaGetLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif
      };

    template<typename T>
      void calculate_distances(int verbose, int q, size_t n, int d, int k,
                               thrust::device_vector<T>& data,
                               size_t data_offset,
                               thrust::device_vector<T>& centroids,
                               thrust::device_vector<T>& data_dots,
                               thrust::device_vector<T>& centroid_dots,
                               thrust::device_vector<T>& pairwise_distances);

    template<typename T, typename F>
    void batch_calculate_distances(int verbose, int q, size_t n, int d, int k,
                                           thrust::device_vector<T> &data,
                                           thrust::device_vector<T> &centroids,
                                           thrust::device_vector<T> &data_dots,
                                           thrust::device_vector<T> &centroid_dots,
                                           F functor) {
      // Get info about available memory
      // This part of the algo can be very memory consuming
      // We might need to batch it
      size_t free_byte;
      size_t total_byte;
      CUDACHECK(cudaMemGetInfo( &free_byte, &total_byte ));
      free_byte *= 0.8;

      size_t required_byte = n * k * sizeof(T);

      size_t runs = std::ceil( required_byte / (double)free_byte );

      log_verbose(verbose,
                  "Batch calculate distance - Rows %ld | K %ld | Data size %d",
                  n, k, sizeof(T)
      );

      log_verbose(verbose,
                  "Batch calculate distance - Free memory %zu | Required memory %lf | Runs %d",
                  free_byte, required_byte, runs
      );

      size_t offset = 0;
      size_t rows_per_run = n / runs;
      thrust::device_vector<T> pairwise_distances(rows_per_run * k);
      for(int run = 0; run < runs; run++) {
        if( run + 1 == runs ) {
          rows_per_run = n % rows_per_run;
          pairwise_distances.resize(rows_per_run * k, (T)0.0);
        } else {
            thrust::fill_n(pairwise_distances.begin(), pairwise_distances.size(), (T)0.0);
        }

        log_verbose(verbose,
                    "Batch calculate distance - Allocated"
        );

        kmeans::detail::calculate_distances(verbose, 0, rows_per_run, d, k,
                                            data, offset,
                                            centroids,
                                            data_dots,
                                            centroid_dots,
                                            pairwise_distances);

        log_verbose(verbose,
                    "Batch calculate distance - Distances calculated"
        );

        functor(rows_per_run, offset, pairwise_distances);

        log_verbose(verbose,
                    "Batch calculate distance - Functor ran"
        );

        offset += rows_per_run;
      }
    }

    template<typename T>
      __global__ void make_new_labels(int n, int k, T* pairwise_distances, int* labels) {
      T min_distance = FLT_MAX; //std::numeric_limits<T>::max(); // might be ok TODO FIXME
      T min_idx = -1;
        int global_id = threadIdx.x + blockIdx.x * blockDim.x;
        if (global_id < n) {
          for(int c = 0; c < k; c++) {
            T distance = pairwise_distances[c * n + global_id];
            if (distance < min_distance) {
              min_distance = distance;
              min_idx = c;
            }
          }
          labels[global_id] = min_idx;
        }
      }


    template<typename T>
      void relabel(int n, int k,
                   thrust::device_vector<T>& pairwise_distances,
                   thrust::device_vector<int>& labels,
                   size_t offset) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
#define MAX_BLOCK_THREADS2 256
        const int GRID_SIZE=(n-1)/MAX_BLOCK_THREADS2+1;
        make_new_labels<<<GRID_SIZE, MAX_BLOCK_THREADS2,0,cuda_stream[dev_num]>>>(
            n, k,
            thrust::raw_pointer_cast(pairwise_distances.data()),
            thrust::raw_pointer_cast(labels.data() + offset));
#if(CHECK)
        gpuErrchk( cudaGetLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif
      }

  }
}
namespace mycub {

  extern void *d_key_alt_buf[MAX_NGPUS];
  extern unsigned int key_alt_buf_bytes[MAX_NGPUS];
  extern void *d_value_alt_buf[MAX_NGPUS];
  extern unsigned int value_alt_buf_bytes[MAX_NGPUS];
  extern void *d_temp_storage[MAX_NGPUS];
  extern size_t temp_storage_bytes[MAX_NGPUS];
  extern void *d_temp_storage2[MAX_NGPUS];
  extern size_t temp_storage_bytes2[MAX_NGPUS];
  extern bool cub_initted;

  void sort_by_key_int(thrust::device_vector<int>& keys, thrust::device_vector<int>& values);

  template <typename T, typename U>
    void sort_by_key(thrust::device_vector<T>& keys, thrust::device_vector<U>& values) {
      int dev_num;
      safe_cuda(cudaGetDevice(&dev_num));
      cudaStream_t this_stream = cuda_stream[dev_num];
      int SIZE = keys.size();
      if (key_alt_buf_bytes[dev_num] < sizeof(T)*SIZE) {
        if (d_key_alt_buf[dev_num]) safe_cuda(cudaFree(d_key_alt_buf[dev_num]));
        safe_cuda(cudaMalloc(&d_key_alt_buf[dev_num], sizeof(T)*SIZE));
        key_alt_buf_bytes[dev_num] = sizeof(T)*SIZE;
        std::cout << "Malloc key_alt_buf" << std::endl;
      }
      if (value_alt_buf_bytes[dev_num] < sizeof(U)*SIZE) {
        if (d_value_alt_buf[dev_num]) safe_cuda(cudaFree(d_value_alt_buf[dev_num]));
        safe_cuda(cudaMalloc(&d_value_alt_buf[dev_num], sizeof(U)*SIZE));
        value_alt_buf_bytes[dev_num] = sizeof(U)*SIZE;
        std::cout << "Malloc value_alt_buf" << std::endl;
      }
      cub::DoubleBuffer<T> d_keys(thrust::raw_pointer_cast(keys.data()), (T*)d_key_alt_buf[dev_num]);
      cub::DoubleBuffer<U> d_values(thrust::raw_pointer_cast(values.data()), (U*)d_value_alt_buf[dev_num]);
      cudaError_t err;

      // Determine temporary device storage requirements for sorting operation
      //if (temp_storage_bytes[dev_num] == 0) {
      void *d_temp;
      size_t temp_bytes;
      err = cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_bytes, d_keys,
          d_values, SIZE, 0, sizeof(T)*8, this_stream);
      // Allocate temporary storage for sorting operation
      safe_cuda(cudaMalloc(&d_temp, temp_bytes));
      d_temp_storage[dev_num] = d_temp;
      temp_storage_bytes[dev_num] = temp_bytes;
      std::cout << "Malloc temp_storage. " << temp_storage_bytes[dev_num] << " bytes" << std::endl;
      std::cout << "d_temp_storage[" << dev_num << "] = " << d_temp_storage[dev_num] << std::endl;
      if (err) {
        std::cout <<"Error " << err << " in SortPairs 1" << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
      }
      //}
      // Run sorting operation
      err = cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys,
          d_values, SIZE, 0, sizeof(T)*8, this_stream);
      if (err) std::cout <<"Error in SortPairs 2" << std::endl;
      //cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys,
      //                                d_values, SIZE, 0, sizeof(T)*8, this_stream);

    }
  template <typename T>
    void sum_reduce(thrust::device_vector<T>& values, T* sum) {
      int dev_num;
      safe_cuda(cudaGetDevice(&dev_num));
      if (!d_temp_storage2[dev_num]) {
        cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num], thrust::raw_pointer_cast(values.data()),
            sum, values.size(), cuda_stream[dev_num]);
        // Allocate temporary storage for sorting operation
        safe_cuda(cudaMalloc(&d_temp_storage2[dev_num], temp_storage_bytes2[dev_num]));
      }
      cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num], thrust::raw_pointer_cast(values.data()),
          sum, values.size(), cuda_stream[dev_num]);
    }
  void cub_init();
  void cub_close();

  void cub_init(int dev);
  void cub_close(int dev);
}
