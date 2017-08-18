// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#include <unistd.h>
#include "kmeans_labels.h"
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>
#include <cfloat>
#include "include/kmeans_general.h"

extern cudaStream_t cuda_stream[MAX_NGPUS];

cudaStream_t cuda_stream[MAX_NGPUS];
namespace kmeans {
    namespace detail {
        cublasHandle_t cublas_handle[MAX_NGPUS];

        void labels_init() {
            cublasStatus_t stat;
            cudaError_t err;
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            stat = cublasCreate(&detail::cublas_handle[dev_num]);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cout << "CUBLAS initialization failed" << std::endl;
                exit(1);
            }
            err = safe_cuda(cudaStreamCreate(&cuda_stream[dev_num]));
            if (err != cudaSuccess) {
                std::cout << "Stream creation failed" << std::endl;

            }
            cublasSetStream(cublas_handle[dev_num], cuda_stream[dev_num]);
            mycub::cub_init();
        }

        void labels_close() {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cublas(cublasDestroy(cublas_handle[dev_num]));
            safe_cuda(cudaStreamDestroy(cuda_stream[dev_num]));
            mycub::cub_close();
        }

        void streamsync(int dev_num) {
            cudaStreamSynchronize(cuda_stream[dev_num]);
        }
    }
}

namespace mycub {
    void *d_key_alt_buf[MAX_NGPUS];
    unsigned int key_alt_buf_bytes[MAX_NGPUS];
    void *d_value_alt_buf[MAX_NGPUS];
    unsigned int value_alt_buf_bytes[MAX_NGPUS];
    void *d_temp_storage[MAX_NGPUS];
    size_t temp_storage_bytes[MAX_NGPUS];
    void *d_temp_storage2[MAX_NGPUS];
    size_t temp_storage_bytes2[MAX_NGPUS];
    bool cub_initted;

    void cub_init() {
        // std::cout <<"CUB init" << std::endl;
        for (int q = 0; q < MAX_NGPUS; q++) {
            d_key_alt_buf[q] = NULL;
            key_alt_buf_bytes[q] = 0;
            d_value_alt_buf[q] = NULL;
            value_alt_buf_bytes[q] = 0;
            d_temp_storage[q] = NULL;
            temp_storage_bytes[q] = 0;
            d_temp_storage2[q] = NULL;
            temp_storage_bytes2[q] = 0;
        }
        cub_initted = true;
    }

    void cub_close() {
        for (int q = 0; q < MAX_NGPUS; q++) {
            if (d_key_alt_buf[q]) safe_cuda(cudaFree(d_key_alt_buf[q]));
            if (d_value_alt_buf[q]) safe_cuda(cudaFree(d_value_alt_buf[q]));
            if (d_temp_storage[q]) safe_cuda(cudaFree(d_temp_storage[q]));
            if (d_temp_storage2[q]) safe_cuda(cudaFree(d_temp_storage2[q]));
            d_temp_storage[q] = NULL;
            d_temp_storage2[q] = NULL;
        }
        cub_initted = false;
    }

    void sort_by_key_int(thrust::device_vector<int> &keys, thrust::device_vector<int> &values) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        cudaStream_t this_stream = cuda_stream[dev_num];
        int SIZE = keys.size();
        //int *d_key_alt_buf, *d_value_alt_buf;
        if (key_alt_buf_bytes[dev_num] < sizeof(int) * SIZE) {
            if (d_key_alt_buf[dev_num]) safe_cuda(cudaFree(d_key_alt_buf[dev_num]));
            safe_cuda(cudaMalloc(&d_key_alt_buf[dev_num], sizeof(int) * SIZE));
            key_alt_buf_bytes[dev_num] = sizeof(int) * SIZE;
        }
        if (value_alt_buf_bytes[dev_num] < sizeof(int) * SIZE) {
            if (d_value_alt_buf[dev_num]) safe_cuda(cudaFree(d_value_alt_buf[dev_num]));
            safe_cuda(cudaMalloc(&d_value_alt_buf[dev_num], sizeof(int) * SIZE));
            value_alt_buf_bytes[dev_num] = sizeof(int) * SIZE;
        }
        cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys.data()), (int *) d_key_alt_buf[dev_num]);
        cub::DoubleBuffer<int> d_values(thrust::raw_pointer_cast(values.data()), (int *) d_value_alt_buf[dev_num]);

        // Determine temporary device storage requirements for sorting operation
        if (!d_temp_storage[dev_num]) {
            cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys,
                                            d_values, SIZE, 0, sizeof(int) * 8, this_stream);
            // Allocate temporary storage for sorting operation
            safe_cuda(cudaMalloc(&d_temp_storage[dev_num], temp_storage_bytes[dev_num]));
        }
        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys,
                                        d_values, SIZE, 0, sizeof(int) * 8, this_stream);
        // Sorted keys and values are referenced by d_keys.Current() and d_values.Current()
    }


}
