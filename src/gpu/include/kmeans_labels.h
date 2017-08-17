// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once

#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>
#include <cublas_v2.h>
#include <cfloat>
#include "kmeans_general.h"

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *cudaGetErrorEnum(cublasStatus_t error);
#endif

extern cudaStream_t cuda_stream[MAX_NGPUS];

template<unsigned int i>
extern __global__ void debugMark();

namespace kmeans {
    namespace detail {

        void labels_init();

        void labels_close();

        template<typename T>
        void memcpy(thrust::host_vector <T, std::allocator<T>> &H,
                    thrust::device_vector <T, thrust::device_malloc_allocator<T>> &D);

        template<typename T>
        void memcpy(thrust::device_vector <T, thrust::device_malloc_allocator<T>> &D,
                    thrust::host_vector <T, std::allocator<T>> &H);

        template<typename T>
        void memcpy(thrust::device_vector <T, thrust::device_malloc_allocator<T>> &Do,
                    thrust::device_vector <T, thrust::device_malloc_allocator<T>> &Di);

        template<typename T>
        void memzero(thrust::device_vector <T, thrust::device_malloc_allocator<T>> &D);

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
        __global__ void self_dots(int n, int d, T *data, T *dots);


        template<typename T>
        void make_self_dots(int n, int d, thrust::device_vector <T> &data, thrust::device_vector <T> &dots);

#define MAX_BLOCK_THREADS 32

        template<typename T>
        __global__ void all_dots(int n, int k, T *data_dots, T *centroid_dots, T *dots);


        template<typename T>
        void make_all_dots(int n, int k, thrust::device_vector <T> &data_dots,
                           thrust::device_vector <T> &centroid_dots,
                           thrust::device_vector <T> &dots);

        template<typename T>
        void calculate_distances(int verbose, int q, int n, int d, int k,
                                 thrust::device_vector <T> &data,
                                 thrust::device_vector <T> &centroids,
                                 thrust::device_vector <T> &data_dots,
                                 thrust::device_vector <T> &centroid_dots,
                                 thrust::device_vector <T> &pairwise_distances);

        template<typename T>
        __global__ void make_new_labels(int n, int k, T *pairwise_distances,
                                        int *labels, int *changes,
                                        T *distances);


        template<typename T>
        void relabel(int n, int k,
                     thrust::device_vector <T> &pairwise_distances,
                     thrust::device_vector<int> &labels,
                     thrust::device_vector <T> &distances,
                     int *d_changes);

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

    void sort_by_key_int(thrust::device_vector<int> &keys, thrust::device_vector<int> &values);

    template<typename T, typename U>
    void sort_by_key(thrust::device_vector <T> &keys, thrust::device_vector <U> &values);

    template<typename T>
    void sum_reduce(thrust::device_vector <T> &values, T *sum);

    void cub_init();

    void cub_close();
}
