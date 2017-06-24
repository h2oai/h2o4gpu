// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cublas_v2.h>
#include <cfloat>
#include "kmeans_general.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
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
        cudaGetDevice(&dev_num);
        cudaMemcpyAsync(thrust::raw_pointer_cast(H.data()),
            thrust::raw_pointer_cast(D.data()),
            sizeof(T) * D.size(), cudaMemcpyDeviceToHost, cuda_stream[dev_num]);
      }

    template<typename T>
      void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D,
          thrust::host_vector<T, std::allocator<T> > &H) {
        int dev_num;
        cudaGetDevice(&dev_num);
        cudaMemcpyAsync(thrust::raw_pointer_cast(D.data()),
            thrust::raw_pointer_cast(H.data()),
            sizeof(T) * H.size(), cudaMemcpyHostToDevice, cuda_stream[dev_num]);
      }
    template<typename T>
      void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Do,
          thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Di) {
        int dev_num;
        cudaGetDevice(&dev_num);
        cudaMemcpyAsync(thrust::raw_pointer_cast(Do.data()),
            thrust::raw_pointer_cast(Di.data()),
            sizeof(T) * Di.size(), cudaMemcpyDeviceToDevice, cuda_stream[dev_num]);
      }
    template<typename T>
      void memzero(thrust::device_vector<T, thrust::device_malloc_allocator<T> >& D) {
        int dev_num;
        cudaGetDevice(&dev_num);
        cudaMemsetAsync(thrust::raw_pointer_cast(D.data()), 0, sizeof(T)*D.size(), cuda_stream[dev_num]);
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
        cudaGetDevice(&dev_num);
        self_dots<<<GRID_SIZE, MAX_BLOCK_THREADS0, 0, cuda_stream[dev_num]>>>(n, d, thrust::raw_pointer_cast(data.data()),
            thrust::raw_pointer_cast(dots.data()));
#if(DEBUG)
        gpuErrchk( cudaPeekAtLastError() );
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
      void make_all_dots(int n, int k, thrust::device_vector<T>& data_dots,
          thrust::device_vector<T>& centroid_dots,
          thrust::device_vector<T>& dots) {
        int dev_num;
        cudaGetDevice(&dev_num);
        const int BLOCK_THREADSX = MAX_BLOCK_THREADS; // BLOCK_THREADSX*BLOCK_THREADSY<=1024 on modern arch's (sm_61)
        const int BLOCK_THREADSY = MAX_BLOCK_THREADS;
        const int GRID_SIZEX=(n-1)/BLOCK_THREADSX+1; // on old arch's this has to be less than 2^16=65536
        const int GRID_SIZEY=(k-1)/BLOCK_THREADSY+1; // this has to be less than 2^16=65536
        //        printf("pre all_dots: %d %d %d %d\n",GRID_SIZEX,GRID_SIZEY,BLOCK_THREADSX,BLOCK_THREADSY); fflush(stdout);
        all_dots<<<
          dim3(GRID_SIZEX,GRID_SIZEY),
          dim3(BLOCK_THREADSX, BLOCK_THREADSY), 0,
          cuda_stream[dev_num]>>>(n, k, thrust::raw_pointer_cast(data_dots.data()),
              thrust::raw_pointer_cast(centroid_dots.data()),
              thrust::raw_pointer_cast(dots.data()));
#if(DEBUG)
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif
      };

    template<typename T>
      void calculate_distances(int q, int n, int d, int k,
          thrust::device_vector<T>& data,
          thrust::device_vector<T>& centroids,
          thrust::device_vector<T>& data_dots,
          thrust::device_vector<T>& centroid_dots,
          thrust::device_vector<T>& pairwise_distances);

    template<typename T>
      __global__ void make_new_labels(int n, int k, T* pairwise_distances,
          int* labels, int* changes,
          T* distances) {
      T min_distance = FLT_MAX; //std::numeric_limits<T>::max(); // might be ok TODO FIXME
        T min_idx = -1;
        int global_id = threadIdx.x + blockIdx.x * blockDim.x;
        if (global_id < n) {
          int old_label = labels[global_id];
          for(int c = 0; c < k; c++) {
            T distance = pairwise_distances[c * n + global_id];
            if (distance < min_distance) {
              min_distance = distance;
              min_idx = c;
            }
          }
          labels[global_id] = min_idx;
          distances[global_id] = min_distance;
          if (old_label != min_idx) {
            atomicAdd(changes, 1);
          }
        }
      }


    template<typename T>
      void relabel(int n, int k,
          thrust::device_vector<T>& pairwise_distances,
          thrust::device_vector<int>& labels,
          thrust::device_vector<T>& distances,
          int *d_changes) {
        int dev_num;
        cudaGetDevice(&dev_num);
        cudaMemsetAsync(d_changes, 0, sizeof(int), cuda_stream[dev_num]);
#define MAX_BLOCK_THREADS2 256
        const int GRID_SIZE=(n-1)/MAX_BLOCK_THREADS2+1;
        make_new_labels<<<GRID_SIZE, MAX_BLOCK_THREADS2,0,cuda_stream[dev_num]>>>(
            n, k,
            thrust::raw_pointer_cast(pairwise_distances.data()),
            thrust::raw_pointer_cast(labels.data()),
            d_changes,
            thrust::raw_pointer_cast(distances.data()));
#if(DEBUG)
        gpuErrchk( cudaPeekAtLastError() );
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
      cudaGetDevice(&dev_num);
      cudaStream_t this_stream = cuda_stream[dev_num]; 
      int SIZE = keys.size();
      if (key_alt_buf_bytes[dev_num] < sizeof(T)*SIZE) {
        if (d_key_alt_buf[dev_num]) cudaFree(d_key_alt_buf[dev_num]);
        cudaMalloc(&d_key_alt_buf[dev_num], sizeof(T)*SIZE);
        key_alt_buf_bytes[dev_num] = sizeof(T)*SIZE;
        std::cout << "Malloc key_alt_buf" << std::endl;
      }
      if (value_alt_buf_bytes[dev_num] < sizeof(U)*SIZE) {
        if (d_value_alt_buf[dev_num]) cudaFree(d_value_alt_buf[dev_num]);
        cudaMalloc(&d_value_alt_buf[dev_num], sizeof(U)*SIZE);
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
      cudaMalloc(&d_temp, temp_bytes);
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
      cudaGetDevice(&dev_num);
      if (!d_temp_storage2[dev_num]) {
        cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num], thrust::raw_pointer_cast(values.data()),
            sum, values.size(), cuda_stream[dev_num]); 
        // Allocate temporary storage for sorting operation
        cudaMalloc(&d_temp_storage2[dev_num], temp_storage_bytes2[dev_num]);
      }
      cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num], thrust::raw_pointer_cast(values.data()),
          sum, values.size(), cuda_stream[dev_num]); 
    }
  void cub_init();
  void cub_close();
}
