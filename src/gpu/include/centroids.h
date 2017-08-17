// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <thrust/device_vector.h>
#include "kmeans_labels.h"
#include "kmeans_general.h"

#if(1)
  __device__ double myatomicAdd(double* address, double val)
{
      unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
      unsigned long long int old = *address_as_ull, assumed;
      do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
      } while (assumed != old);
      return __longlong_as_double(old);
}

  __device__ float myatomicAdd(float* address, float val)
{
  return(atomicAdd(address,val));
}
#else
//template<typename T>
//__device__ T myatomicAdd(T* address, T val)
//{
//  myatomicAdd(address,val);
//}

//  __device__ float myatomicAdd(float* address, float val)
//{
//  return(atomicAdd<float>(address,val));
//}
//  __device__ double myatomicAdd(double* address, double val)
   //{
   //  return(atomicAdd<double>(address,val));
   //}
#define myatomicAdd(A,B) atomicAdd(A,B)
#endif

namespace kmeans {
  namespace detail {

    template<typename T>
      __device__ __forceinline__ void update_centroid(int label, int dimension,
          int d,
          T accumulator, T* centroids,
          int count, int* counts) {
        int index = label * d + dimension;
        T* target = centroids + index;
        myatomicAdd(target, accumulator);
        if (dimension == 0) {
          atomicAdd(counts + label, count);
        }             
      }


    template<typename T>
      __global__ void calculate_centroids(int n, int d, int k,
          T* data,
          int* ordered_labels,
          int* ordered_indices,
          T* centroids,
          int* counts) {
        int in_flight = blockDim.y * gridDim.y;
        int labels_per_row = (n - 1) / in_flight + 1; 
        for(int dimension = threadIdx.x; dimension < d; dimension += blockDim.x) {
          T accumulator = 0;
          int count = 0;
          int global_id = threadIdx.y + blockIdx.y * blockDim.y;
          int start = global_id * labels_per_row;
          int end = (global_id + 1) * labels_per_row;
          end = (end > n) ? n : end;
          int prior_label;
          if (start < n) {
            prior_label = ordered_labels[start];

            for(int label_number = start; label_number < end; label_number++) {
              int label = ordered_labels[label_number];
              if (label != prior_label) {
                update_centroid(prior_label, dimension,
                    d,
                    accumulator, centroids,
                    count, counts);
                accumulator = 0;
                count = 0;
              }

              T value = data[dimension + ordered_indices[label_number] * d];
              accumulator += value;
              prior_label = label;
              count++;
            }
            update_centroid(prior_label, dimension,
                d,
                accumulator, centroids,
                count, counts);
          }
        }
      }

    template<typename T>
      __global__ void scale_centroids(int d, int k, int* counts, T* centroids) {
        int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
        int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
        if ((global_id_x < d) && (global_id_y < k)) {
          int count = counts[global_id_y];
          //To avoid introducing divide by zero errors
          //If a centroid has no weight, we'll do no normalization
          //This will keep its coordinates defined.
          if (count < 1) {
            count = 1;
          }
          T scale = 1.0/T(count);
          centroids[global_id_x + d * global_id_y] *= scale;
        }
      }

    template<typename T>
      void find_centroids(int q, int n, int d, int k,
          thrust::device_vector<T>& data,
          thrust::device_vector<int>& labels,
          thrust::device_vector<T>& centroids,
          thrust::device_vector<int>& range,
          thrust::device_vector<int>& indices,
          thrust::device_vector<int>& counts) {
        int dev_num;
        cudaGetDevice(&dev_num);
        memcpy(indices,range);
        //Bring all labels with the same value together
#if 0
        thrust::sort_by_key(labels.begin(),
            labels.end(),
            indices.begin());
#else
        mycub::sort_by_key_int(labels, indices);
#endif

#if(CHECK)
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif
#if(DEBUG)
        thrust::host_vector<int> h_labels = labels;
        thrust::host_vector<int> h_indices = indices;
        for(int i=0;i<n;i++){
          fprintf(stderr,"after q=%d labels[%d]=%d indices[%d]=%d\n",q,i,h_labels[i],i,h_indices[i]);
        }
        fflush(stderr);
#endif

        
        
        if(0){ // no, if centroid has no members, then this will leave centroid at (arbitrarily) zero position for all dimensions.  Rather keep original position in case no members, so can gracefully add or remove members toward convergence.
          // Required for ngpu>1 where average centroids.  I.e., if gpu_id=1 has a centroid that lost members, then the blind average drives the average centroid position toward zero.  This likely reduces its members, leading to run away case of centroids heading to zero.
          //Initialize centroids to all zeros
          memzero(centroids);
        }

        //Initialize counts to all zeros
        memzero(counts);

        //Calculate centroids 
        int n_threads_x = 64; // TODO FIXME
        int n_threads_y = 16; // TODO FIXME
        //XXX Number of blocks here is hard coded at 30
        //This should be taken care of more thoughtfully.
        calculate_centroids<<<dim3(1, 30), dim3(n_threads_x, n_threads_y), // TODO FIXME
          0, cuda_stream[dev_num]>>>
            (n, d, k,
             thrust::raw_pointer_cast(data.data()),
             thrust::raw_pointer_cast(labels.data()),
             thrust::raw_pointer_cast(indices.data()),
             thrust::raw_pointer_cast(centroids.data()),
             thrust::raw_pointer_cast(counts.data()));
#if(CHECK)
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif
#if(DEBUG)
        thrust::host_vector<T> h_centroids = centroids;
        for(int i=0;i<k*d;i++){
          fprintf(stderr,"after q=%d centroids[%d]=%g\n",q,i,h_centroids[i]);
        }
        fflush(stderr);
#endif
        //Scale centroids
        scale_centroids<<<dim3((d-1)/32+1, (k-1)/32+1), dim3(32, 32), // TODO FIXME
          0, cuda_stream[dev_num]>>>
            (d, k,
             thrust::raw_pointer_cast(counts.data()),
             thrust::raw_pointer_cast(centroids.data()));
#if(CHECK)
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
#endif
#if(DEBUG)
        h_centroids = centroids;
        for(int i=0;i<k*d;i++){
          fprintf(stderr,"afters q=%d centroids[%d]=%g\n",q,i,h_centroids[i]);
        }
        fflush(stderr);
#endif
      }

  }
}
#endif
