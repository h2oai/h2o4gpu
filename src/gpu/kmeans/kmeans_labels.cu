/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#include "kmeans_labels.h"
#include <cublas_v2.h>
#include <cfloat>
#include <unistd.h>
#include "kmeans_general.h"

cudaStream_t cuda_stream[MAX_NGPUS];

namespace kmeans {
namespace detail {

template<typename T>
struct absolute_value {
  __host__ __device__
  void operator()(T &x) const {
    x = (x > 0 ? x : -x);
  }
};

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
  mycub::cub_init(dev_num);
}

void labels_close() {
  int dev_num;
  safe_cuda(cudaGetDevice(&dev_num));
  safe_cublas(cublasDestroy(cublas_handle[dev_num]));
  safe_cuda(cudaStreamDestroy(cuda_stream[dev_num]));
  mycub::cub_close(dev_num);
}

void streamsync(int dev_num) {
  cudaStreamSynchronize(cuda_stream[dev_num]);
}

/**
 * Matrix multiplication: alpha * A^T * B + beta * C
 * Optimized for tall and skinny matrices
 *
 * @tparam float_t
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 * @param n
 * @param d
 * @param k
 * @param max_block_rows
 * @return
 */
template<typename float_t>
__global__ void matmul(const float_t *A, const float_t *B, float_t *C,
                       const float_t alpha, const float_t beta, int n, int d, int k, int max_block_rows) {
  extern __shared__ __align__(sizeof(float_t)) unsigned char my_smem[];
  float_t *shared = reinterpret_cast<float_t *>(my_smem);

  float_t *s_A = shared;
  float_t *s_B = shared + max_block_rows * d;

  for (int i = threadIdx.x; i < d * k; i += blockDim.x) {
    s_B[i] = B[i];
  }

  int block_start_row_index = blockIdx.x * max_block_rows;
  int block_rows = max_block_rows;

  if (blockIdx.x == gridDim.x - 1 && n % max_block_rows != 0) {
    block_rows = n % max_block_rows;
  }

  for (int i = threadIdx.x; i < d * block_rows; i += blockDim.x) {
    s_A[i] = alpha * A[d * block_start_row_index + i];
  }

  __syncthreads();

  float_t elem_c = 0;

  int col_c = threadIdx.x % k;
  int abs_row_c = block_start_row_index + threadIdx.x / k;
  int row_c = threadIdx.x / k;

  // Thread/Block combination either too far for data array
  // Or is calculating for index that should be calculated in a different blocks - in some edge cases
  // "col_c * n + abs_row_c" can yield same result in different thread/block combinations
  if (abs_row_c >= n || threadIdx.x >= block_rows * k) {
    return;
  }

  for (int i = 0; i < d; i++) {
    elem_c += s_B[d * col_c + i] * s_A[d * row_c + i];
  }

  C[col_c * n + abs_row_c] = beta * C[col_c * n + abs_row_c] + elem_c;

}

template<>
void calculate_distances<double>(int verbose, int q, size_t n, int d, int k,
                                 thrust::device_vector<double> &data,
                                 size_t data_offset,
                                 thrust::device_vector<double> &centroids,
                                 thrust::device_vector<double> &data_dots,
                                 thrust::device_vector<double> &centroid_dots,
                                 thrust::device_vector<double> &pairwise_distances) {
  detail::make_self_dots(k, d, centroids, centroid_dots);
  detail::make_all_dots(n, k, data_offset, data_dots, centroid_dots, pairwise_distances);

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
  safe_cuda(cudaGetDevice(&dev_num));

  if (k <= 16 && d <= 64) {
    const int BLOCK_SIZE_MUL = 128;
    int block_rows = std::min(BLOCK_SIZE_MUL / k, n);
    int grid_size = std::ceil(static_cast<double>(n) / block_rows);

    int shared_size_B = d * k * sizeof(double);
    int shared_size_A = block_rows * d * sizeof(double);

    matmul << < grid_size, BLOCK_SIZE_MUL, shared_size_B + shared_size_A >> > (
        thrust::raw_pointer_cast(data.data() + data_offset * d),
            thrust::raw_pointer_cast(centroids.data()),
            thrust::raw_pointer_cast(pairwise_distances.data()),
            alpha, beta, n, d, k, block_rows
    );

    safe_cuda(cudaDeviceSynchronize());
  } else {
    cublasStatus_t stat = safe_cublas(cublasDgemm(detail::cublas_handle[dev_num],
                                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                                  n, k, d, &alpha,
                                                  thrust::raw_pointer_cast(data.data() + data_offset * d),
                                                  d,//Has to be n or d
                                                  thrust::raw_pointer_cast(centroids.data()),
                                                  d,//Has to be k or d
                                                  &beta,
                                                  thrust::raw_pointer_cast(pairwise_distances.data()),
                                                  n)); //Has to be n or k

    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << "Invalid Dgemm" << std::endl;
      exit(1);
    }
  }

  thrust::for_each(pairwise_distances.begin(),
                   pairwise_distances.end(),
                   absolute_value<double>()); // in-place transformation to ensure all distances are positive indefinite

  #if(CHECK)
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  #endif
}

template<>
void calculate_distances<float>(int verbose, int q, size_t n, int d, int k,
                                thrust::device_vector<float> &data,
                                size_t data_offset,
                                thrust::device_vector<float> &centroids,
                                thrust::device_vector<float> &data_dots,
                                thrust::device_vector<float> &centroid_dots,
                                thrust::device_vector<float> &pairwise_distances) {
  detail::make_self_dots(k, d, centroids, centroid_dots);
  detail::make_all_dots(n, k, data_offset, data_dots, centroid_dots, pairwise_distances);

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
  safe_cuda(cudaGetDevice(&dev_num));

  if (k <= 16 && d <= 64) {
    const int BLOCK_SIZE_MUL = 128;
    int block_rows = std::min(BLOCK_SIZE_MUL / k, n);
    int grid_size = std::ceil(static_cast<float>(n) / block_rows);

    int shared_size_B = d * k * sizeof(float);
    int shared_size_A = block_rows * d * sizeof(float);

    matmul << < grid_size, BLOCK_SIZE_MUL, shared_size_B + shared_size_A >> > (
        thrust::raw_pointer_cast(data.data() + data_offset * d),
            thrust::raw_pointer_cast(centroids.data()),
            thrust::raw_pointer_cast(pairwise_distances.data()),
            alpha, beta, n, d, k, block_rows
    );
    safe_cuda(cudaDeviceSynchronize());
  } else {
    cublasStatus_t stat = safe_cublas(cublasSgemm(detail::cublas_handle[dev_num],
                                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                                  n, k, d, &alpha,
                                                  thrust::raw_pointer_cast(data.data() + data_offset * d),
                                                  d,//Has to be n or d
                                                  thrust::raw_pointer_cast(centroids.data()),
                                                  d,//Has to be k or d
                                                  &beta,
                                                  thrust::raw_pointer_cast(pairwise_distances.data()),
                                                  n)); //Has to be n or k

    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << "Invalid Sgemm" << std::endl;
      exit(1);
    }
  }

  thrust::for_each(pairwise_distances.begin(),
                   pairwise_distances.end(),
                   absolute_value<float>()); // in-place transformation to ensure all distances are positive indefinite

  #if(CHECK)
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  #endif
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

void cub_init(int dev) {
  d_key_alt_buf[dev] = NULL;
  key_alt_buf_bytes[dev] = 0;
  d_value_alt_buf[dev] = NULL;
  value_alt_buf_bytes[dev] = 0;
  d_temp_storage[dev] = NULL;
  temp_storage_bytes[dev] = 0;
  d_temp_storage2[dev] = NULL;
  temp_storage_bytes2[dev] = 0;
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

void cub_close(int dev) {
  if (d_key_alt_buf[dev]) safe_cuda(cudaFree(d_key_alt_buf[dev]));
  if (d_value_alt_buf[dev]) safe_cuda(cudaFree(d_value_alt_buf[dev]));
  if (d_temp_storage[dev]) safe_cuda(cudaFree(d_temp_storage[dev]));
  if (d_temp_storage2[dev]) safe_cuda(cudaFree(d_temp_storage2[dev]));
  d_temp_storage[dev] = NULL;
  d_temp_storage2[dev] = NULL;
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
  cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num],
                                  d_keys, d_values, SIZE, 0, sizeof(int) * 8, this_stream);
  // Sorted keys and values are referenced by d_keys.Current() and d_values.Current()

  keys.data() = thrust::device_pointer_cast(d_keys.Current());
  values.data() = thrust::device_pointer_cast(d_values.Current());
}

}