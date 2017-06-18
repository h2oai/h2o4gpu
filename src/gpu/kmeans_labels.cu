// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#include "kmeans_labels.h"
#include <cublas_v2.h>
#include <cfloat>

cudaStream_t cuda_stream[16];
namespace kmeans {
  namespace detail {

    cublasHandle_t cublas_handle[16];

    void labels_init() {
      cublasStatus_t stat;
      cudaError_t err;
      int dev_num;
      cudaGetDevice(&dev_num);
      stat = cublasCreate(&detail::cublas_handle[dev_num]);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
        exit(1);
      }
      err = cudaStreamCreate(&cuda_stream[dev_num]);
      if (err != cudaSuccess) {
        std::cout << "Stream creation failed" << std::endl;
        exit(1);
      }
      cublasSetStream(cublas_handle[dev_num], cuda_stream[dev_num]);
      mycub::cub_init();
    }

    void labels_close() {
      int dev_num;
      cudaGetDevice(&dev_num);
      cublasDestroy(cublas_handle[dev_num]);
      cudaStreamDestroy(cuda_stream[dev_num]);
      mycub::cub_close();
    }

    void streamsync(int dev_num) {
      cudaStreamSynchronize(cuda_stream[dev_num]);
    }

    template<>
      void calculate_distances<double>(int n, int d, int k,
          thrust::device_vector<double>& data,
          thrust::device_vector<double>& centroids,
          thrust::device_vector<double>& data_dots,
          thrust::device_vector<double>& centroid_dots,
          thrust::device_vector<double>& pairwise_distances) {
        detail::make_self_dots(k, d, centroids, centroid_dots);
        detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);
        //||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
        //pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
        //The dgemm calculates x.y for all x and y, so alpha = -2.0
        double alpha = -2.0;
        double beta = 1.0;
        //If the data were in standard column major order, we'd do a
        //centroids * data ^ T
        //But the data is in row major order, so we have to permute
        //the arguments a little
        int dev_num;
        cudaGetDevice(&dev_num);
        cublasStatus_t stat =
          cublasDgemm(detail::cublas_handle[dev_num],
              CUBLAS_OP_T, CUBLAS_OP_N,
              n, k, d, &alpha,
              thrust::raw_pointer_cast(data.data()),
              d,//Has to be n or d
              thrust::raw_pointer_cast(centroids.data()),
              d,//Has to be k or d
              &beta,
              thrust::raw_pointer_cast(pairwise_distances.data()),
              n); //Has to be n or k

        if (stat != CUBLAS_STATUS_SUCCESS) {
          std::cout << "Invalid Dgemm" << std::endl;
          exit(1);
        }
      }

    template<>
      void calculate_distances<float>(int n, int d, int k,
          thrust::device_vector<float>& data,
          thrust::device_vector<float>& centroids,
          thrust::device_vector<float>& data_dots,
          thrust::device_vector<float>& centroid_dots,
          thrust::device_vector<float>& pairwise_distances) {
        detail::make_self_dots(k, d, centroids, centroid_dots);
        detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);
        //||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
        //pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
        //The dgemm calculates x.y for all x and y, so alpha = -2.0
        float alpha = -2.0;
        float beta = 1.0;
        //If the data were in standard column major order, we'd do a
        //centroids * data ^ T
        //But the data is in row major order, so we have to permute
        //the arguments a little
        int dev_num;
        cudaGetDevice(&dev_num);
        cublasStatus_t stat =
          cublasSgemm(detail::cublas_handle[dev_num],
              CUBLAS_OP_T, CUBLAS_OP_N,
              n, k, d, &alpha,
              thrust::raw_pointer_cast(data.data()),
              d,//Has to be n or d
              thrust::raw_pointer_cast(centroids.data()),
              d,//Has to be k or d
              &beta,
              thrust::raw_pointer_cast(pairwise_distances.data()),
              n); //Has to be n or k

        if (stat != CUBLAS_STATUS_SUCCESS) {
          std::cout << "Invalid Sgemm" << std::endl;
          exit(1);
        }
      }


  }
}
namespace mycub {
  void *d_key_alt_buf[16];
  unsigned int key_alt_buf_bytes[16];
  void *d_value_alt_buf[16];
  unsigned int value_alt_buf_bytes[16];
  void *d_temp_storage[16];
  size_t temp_storage_bytes[16];
  void *d_temp_storage2[16];
  size_t temp_storage_bytes2[16];
  bool cub_initted;
  void cub_init() {
    std::cout <<"CUB init" << std::endl;
    for (int q=0; q<16; q++) {
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
    for (int q=0; q<16; q++) {
      if(d_key_alt_buf[q]) cudaFree(d_key_alt_buf[q]);
      if(d_value_alt_buf[q]) cudaFree(d_value_alt_buf[q]);
      if(d_temp_storage[q]) cudaFree(d_temp_storage[q]);
      if(d_temp_storage2[q]) cudaFree(d_temp_storage2[q]);
      d_temp_storage[q] = NULL;
      d_temp_storage2[q] = NULL;
    }
    cub_initted = false;
  }
  void sort_by_key_int(thrust::device_vector<int>& keys, thrust::device_vector<int>& values) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaStream_t this_stream = cuda_stream[dev_num]; 
    int SIZE = keys.size();
    //int *d_key_alt_buf, *d_value_alt_buf;
    if (key_alt_buf_bytes[dev_num] < sizeof(int)*SIZE) {
      if (d_key_alt_buf[dev_num]) cudaFree(d_key_alt_buf[dev_num]);
      cudaMalloc(&d_key_alt_buf[dev_num], sizeof(int)*SIZE);
      key_alt_buf_bytes[dev_num] = sizeof(int)*SIZE;
    }
    if (value_alt_buf_bytes[dev_num] < sizeof(int)*SIZE) {
      if (d_value_alt_buf[dev_num]) cudaFree(d_value_alt_buf[dev_num]);
      cudaMalloc(&d_value_alt_buf[dev_num], sizeof(int)*SIZE);
      value_alt_buf_bytes[dev_num] = sizeof(int)*SIZE;
    }
    cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys.data()), (int*)d_key_alt_buf[dev_num]);
    cub::DoubleBuffer<int> d_values(thrust::raw_pointer_cast(values.data()), (int*)d_value_alt_buf[dev_num]);

    // Determine temporary device storage requirements for sorting operation
    if (!d_temp_storage[dev_num]) {
      cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys, 
          d_values, SIZE, 0, sizeof(int)*8, this_stream);
      // Allocate temporary storage for sorting operation
      cudaMalloc(&d_temp_storage[dev_num], temp_storage_bytes[dev_num]);
    }
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys, 
        d_values, SIZE, 0, sizeof(int)*8, this_stream);
    // Sorted keys and values are referenced by d_keys.Current() and d_values.Current()


  }
}
