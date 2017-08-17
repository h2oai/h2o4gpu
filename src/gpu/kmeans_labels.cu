// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#include <unistd.h>
#include "kmeans_labels.h"
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>
#include <cublas_v2.h>
#include <cfloat>
#include "include/kmeans_general.h"

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
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

extern cudaStream_t cuda_stream[MAX_NGPUS];

template<unsigned int i>
extern __global__ void debugMark() {};

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
            mycub::cub_init();
        }

        void labels_close() {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cublas(cublasDestroy(cublas_handle[dev_num]));
            safe_cuda(cudaStreamDestroy(cuda_stream[dev_num]));
            mycub::cub_close();
        }

        template<typename T>
        void memcpy(thrust::host_vector <T, std::allocator<T>> &H,
                    thrust::device_vector <T, thrust::device_malloc_allocator<T>> &D) {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(H.data()),
                                      thrust::raw_pointer_cast(D.data()),
                                      sizeof(T) * D.size(), cudaMemcpyDeviceToHost, cuda_stream[dev_num]));
        }

        template<typename T>
        void memcpy(thrust::device_vector <T, thrust::device_malloc_allocator<T>> &D,
                    thrust::host_vector <T, std::allocator<T>> &H) {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(D.data()),
                                      thrust::raw_pointer_cast(H.data()),
                                      sizeof(T) * H.size(), cudaMemcpyHostToDevice, cuda_stream[dev_num]));
        }

        template<typename T>
        void memcpy(thrust::device_vector <T, thrust::device_malloc_allocator<T>> &Do,
                    thrust::device_vector <T, thrust::device_malloc_allocator<T>> &Di) {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(Do.data()),
                                      thrust::raw_pointer_cast(Di.data()),
                                      sizeof(T) * Di.size(), cudaMemcpyDeviceToDevice, cuda_stream[dev_num]));
        }

        template<typename T>
        void memzero(thrust::device_vector <T, thrust::device_malloc_allocator<T>> &D) {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cuda(
                    cudaMemsetAsync(thrust::raw_pointer_cast(D.data()), 0, sizeof(T) * D.size(), cuda_stream[dev_num]));
        }

        void streamsync(int dev_num) {
            cudaStreamSynchronize(cuda_stream[dev_num]);
        }

        template<typename T>
        __global__ void self_dots(int n, int d, T *data, T *dots) {
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
        void make_self_dots(int n, int d, thrust::device_vector <T> &data, thrust::device_vector <T> &dots) {
            int dev_num;
#define MAX_BLOCK_THREADS0 256
            const int GRID_SIZE = (n - 1) / MAX_BLOCK_THREADS0 + 1;
            safe_cuda(cudaGetDevice(&dev_num));
            self_dots << < GRID_SIZE, MAX_BLOCK_THREADS0, 0, cuda_stream[dev_num] >> >
                                                             (n, d, thrust::raw_pointer_cast(data.data()),
                                                                     thrust::raw_pointer_cast(dots.data()));
#if(CHECK)
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
#endif

        }

#define MAX_BLOCK_THREADS 32

        template<typename T>
        __global__ void all_dots(int n, int k, T *data_dots, T *centroid_dots, T *dots) {
            __shared__
            T local_data_dots[MAX_BLOCK_THREADS];
            __shared__
            T local_centroid_dots[MAX_BLOCK_THREADS];
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
        void make_all_dots(int n, int k, thrust::device_vector <T> &data_dots,
                           thrust::device_vector <T> &centroid_dots,
                           thrust::device_vector <T> &dots) {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            const int BLOCK_THREADSX = MAX_BLOCK_THREADS; // BLOCK_THREADSX*BLOCK_THREADSY<=1024 on modern arch's (sm_61)
            const int BLOCK_THREADSY = MAX_BLOCK_THREADS;
            const int GRID_SIZEX = (n - 1) / BLOCK_THREADSX + 1; // on old arch's this has to be less than 2^16=65536
            const int GRID_SIZEY = (k - 1) / BLOCK_THREADSY + 1; // this has to be less than 2^16=65536
            //        printf("pre all_dots: %d %d %d %d\n",GRID_SIZEX,GRID_SIZEY,BLOCK_THREADSX,BLOCK_THREADSY); fflush(stdout);
            all_dots << <
            dim3(GRID_SIZEX, GRID_SIZEY),
                    dim3(BLOCK_THREADSX, BLOCK_THREADSY), 0,
                    cuda_stream[dev_num] >> > (n, k, thrust::raw_pointer_cast(data_dots.data()),
                            thrust::raw_pointer_cast(centroid_dots.data()),
                            thrust::raw_pointer_cast(dots.data()));
#if(CHECK)
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
#endif
        };

        template<>
        void calculate_distances<double>(int verbose, int q, int n, int d, int k,
                                         thrust::device_vector<double> &data,
                                         thrust::device_vector<double> &centroids,
                                         thrust::device_vector<double> &data_dots,
                                         thrust::device_vector<double> &centroid_dots,
                                         thrust::device_vector<double> &pairwise_distances) {
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
            safe_cuda(cudaGetDevice(&dev_num));
            cublasStatus_t stat =
            safe_cublas(cublasDgemm(detail::cublas_handle[dev_num],
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, k, d, &alpha,
                                    thrust::raw_pointer_cast(data.data()),
                                    d,//Has to be n or d
                                    thrust::raw_pointer_cast(centroids.data()),
                                    d,//Has to be k or d
                                    &beta,
                                    thrust::raw_pointer_cast(pairwise_distances.data()),
                                    n)); //Has to be n or k

            thrust::for_each(pairwise_distances.begin(), pairwise_distances.end(),
                             absolute_value<double>()); // in-place transformation to ensure all distances are positive indefinite

            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Invalid Dgemm" << std::endl;
                exit(1);
            }

        }

        template<>
        void calculate_distances<float>(int verbose, int q, int n, int d, int k,
                                        thrust::device_vector<float> &data,
                                        thrust::device_vector<float> &centroids,
                                        thrust::device_vector<float> &data_dots,
                                        thrust::device_vector<float> &centroid_dots,
                                        thrust::device_vector<float> &pairwise_distances) {
            detail::make_self_dots(k, d, centroids, centroid_dots);
            detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);

            if (verbose) {
                thrust::host_vector<float> h_data_dots = data_dots;
                thrust::host_vector<float> h_centroid_dots = centroid_dots;
                thrust::host_vector<float> h_pairwise_distances = pairwise_distances;

                for (int i = 0; i < n; i++) {
                    if (i % 1 == 0) {
                        fprintf(stderr, "0 q=%d data_dots[%d]=%g\n", q, i, h_data_dots[i]);
                        fflush(stderr);
                    }
                }
                for (int i = 0; i < k; i++) {
                    fprintf(stderr, "0 q=%d centroid_dots[%d]=%g\n", q, i, h_centroid_dots[i]);
                    fflush(stderr);
                }
                for (int i = 0; i < n * k; i++) {
                    if (i % 1 == 0) {
                        fprintf(stderr, "0 q=%d pairwise_distances[%d]=%g\n", q, i, h_pairwise_distances[i]);
                        fflush(stderr);
                    }
                }
            }

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
            // http://docs.nvidia.com/cuda/cublas/index.html#axzz4kgBuzSr6
            cublasStatus_t stat;
            if (0) {
                int M = n;
                int N = k;
                int K = d;
                int lda = K;
                int ldb = N;
                int ldc = M;
                fprintf(stderr, "%d x %d : data size=%zu\n", lda, M, data.size());
                fflush(stderr);
                fprintf(stderr, "%d x %d : centroids size=%zu\n", ldb, K, centroids.size());
                fflush(stderr);
                fprintf(stderr, "%d x %d : pairwise_distances size=%zu\n", ldc, N, pairwise_distances.size());
                fflush(stderr);
                stat =
                safe_cublas(cublasSgemm(detail::cublas_handle[dev_num],
                                        CUBLAS_OP_T, CUBLAS_OP_T,
                                        M, N, K, &alpha,
                                        thrust::raw_pointer_cast(
                                                data.data()), // <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
                                        lda,
                                        thrust::raw_pointer_cast(
                                                centroids.data()), // <type> array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
                                        ldb,
                                        &beta,
                                        thrust::raw_pointer_cast(
                                                pairwise_distances.data()), // <type> array of dimensions ldc x n with ldc>=max(1,m).
                                        ldc));
            } else if (0) {
                int M = n;
                int N = k;
                int K = d;
                int lda = M;
                int ldb = K;
                int ldc = M;
                fprintf(stderr, "A2 %d x %d : data size=%zu\n", lda, K, data.size());
                fflush(stderr);
                fprintf(stderr, "B2 %d x %d : centroids size=%zu\n", ldb, N, centroids.size());
                fflush(stderr);
                fprintf(stderr, "C2 %d x %d : pairwise_distances size=%zu\n", ldc, N, pairwise_distances.size());
                fflush(stderr);
                stat =
                safe_cublas(cublasSgemm(detail::cublas_handle[dev_num],
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        M, N, K, &alpha,
                                        thrust::raw_pointer_cast(
                                                data.data()), // <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
                                        lda,
                                        thrust::raw_pointer_cast(
                                                centroids.data()), // <type> array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
                                        ldb,
                                        &beta,
                                        thrust::raw_pointer_cast(
                                                pairwise_distances.data()), // <type> array of dimensions ldc x n with ldc>=max(1,m).
                                        ldc));
            } else {
                int M = n; // rows in op(A) and C
                int N = k; // cols in op(B) and C
                int K = d; // cols in op(A) and op(B)
                int lda = K;
                int ldb = K; // http://docs.nvidia.com/cuda/cublas/index.html#axzz4kgBuzSr6 has mistake, transa should have been transb
                //see http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html#gafe51bacb54592ff5de056acabd83c260
                int ldc = M;
                if (verbose >= 2) {
                    fprintf(stderr, "A3 %d x %d -> %d x %d : data size=%zu\n", K, M, M, K, data.size());
                    fflush(stderr);
                    fprintf(stderr, "B3 %d x %d -> %d x %d : centroids size=%zu\n", K, N, K, N, centroids.size());
                    fflush(stderr);
                    fprintf(stderr, "C3 %d x %d : pairwise_distances size=%zu\n", M, N, pairwise_distances.size());
                    fflush(stderr);
                    fflush(stderr);
                    //sleep(5);
                }
                stat =
                safe_cublas(cublasSgemm(detail::cublas_handle[dev_num],
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        M, N, K, &alpha,
                                        thrust::raw_pointer_cast(data.data()),
                                        lda,//Has to be n or d
                                        thrust::raw_pointer_cast(centroids.data()),
                                        ldb,//Has to be k or d
                                        &beta,
                                        thrust::raw_pointer_cast(pairwise_distances.data()),
                                        ldc)); //Has to be n or k
                if (verbose >= 2) {
                    fprintf(stderr, "After cublasSgemm\n");
                    fflush(stderr);
                    //sleep(5);
                }

                thrust::for_each(pairwise_distances.begin(), pairwise_distances.end(),
                                 absolute_value<float>()); // in-place transformation to ensure all distances are positive indefinite
                if (verbose) {
                    thrust::host_vector<float> h_data = data;
                    thrust::host_vector<float> h_centroids = centroids;
                    thrust::host_vector<float> h_pairwise_distances = pairwise_distances;

                    for (int i = 0; i < M * K; i++) {
                        if (i % 1 == 0) {
                            fprintf(stderr, "q=%d data[%d]=%g\n", q, i, h_data[i]);
                            fflush(stderr);
                        }
                    }
                    for (int i = 0; i < K * N; i++) {
                        fprintf(stderr, "q=%d centroids[%d]=%g\n", q, i, h_centroids[i]);
                    }
                    for (int i = 0; i < M * N; i++) {
                        if (i % 1 == 0) {
                            fprintf(stderr, "q=%d pairwise_distances[%d]=%g\n", q, i, h_pairwise_distances[i]);
                            fflush(stderr);
                        }
                    }
                }

            }
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Invalid Sgemm" << std::endl;
                exit(1);
            }
        }

        template<typename T>
        __global__ void make_new_labels(int n, int k, T *pairwise_distances,
                                        int *labels, int *changes,
                                        T *distances) {
            T min_distance = FLT_MAX; //std::numeric_limits<T>::max(); // might be ok TODO FIXME
            T min_idx = -1;
            int global_id = threadIdx.x + blockIdx.x * blockDim.x;
            if (global_id < n) {
                int old_label = labels[global_id];
                for (int c = 0; c < k; c++) {
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
                     thrust::device_vector <T> &pairwise_distances,
                     thrust::device_vector<int> &labels,
                     thrust::device_vector <T> &distances,
                     int *d_changes) {
            int dev_num;
            safe_cuda(cudaGetDevice(&dev_num));
            safe_cuda(cudaMemsetAsync(d_changes, 0, sizeof(int), cuda_stream[dev_num]));
#define MAX_BLOCK_THREADS2 256
            const int GRID_SIZE = (n - 1) / MAX_BLOCK_THREADS2 + 1;
            make_new_labels << < GRID_SIZE, MAX_BLOCK_THREADS2, 0, cuda_stream[dev_num] >> > (
                    n, k,
                            thrust::raw_pointer_cast(pairwise_distances.data()),
                            thrust::raw_pointer_cast(labels.data()),
                            d_changes,
                            thrust::raw_pointer_cast(distances.data()));
#if(CHECK)
            gpuErrchk(cudaPeekAtLastError());
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

    template<typename T, typename U>
    void sort_by_key(thrust::device_vector <T> &keys, thrust::device_vector <U> &values) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        cudaStream_t this_stream = cuda_stream[dev_num];
        int SIZE = keys.size();
        if (key_alt_buf_bytes[dev_num] < sizeof(T) * SIZE) {
            if (d_key_alt_buf[dev_num]) safe_cuda(cudaFree(d_key_alt_buf[dev_num]));
            safe_cuda(cudaMalloc(&d_key_alt_buf[dev_num], sizeof(T) * SIZE));
            key_alt_buf_bytes[dev_num] = sizeof(T) * SIZE;
            std::cout << "Malloc key_alt_buf" << std::endl;
        }
        if (value_alt_buf_bytes[dev_num] < sizeof(U) * SIZE) {
            if (d_value_alt_buf[dev_num]) safe_cuda(cudaFree(d_value_alt_buf[dev_num]));
            safe_cuda(cudaMalloc(&d_value_alt_buf[dev_num], sizeof(U) * SIZE));
            value_alt_buf_bytes[dev_num] = sizeof(U) * SIZE;
            std::cout << "Malloc value_alt_buf" << std::endl;
        }
        cub::DoubleBuffer <T> d_keys(thrust::raw_pointer_cast(keys.data()), (T *) d_key_alt_buf[dev_num]);
        cub::DoubleBuffer <U> d_values(thrust::raw_pointer_cast(values.data()), (U *) d_value_alt_buf[dev_num]);
        cudaError_t err;

        // Determine temporary device storage requirements for sorting operation
        //if (temp_storage_bytes[dev_num] == 0) {
        void *d_temp;
        size_t temp_bytes;
        err = cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_bytes, d_keys,
                                              d_values, SIZE, 0, sizeof(T) * 8, this_stream);
        // Allocate temporary storage for sorting operation
        safe_cuda(cudaMalloc(&d_temp, temp_bytes));
        d_temp_storage[dev_num] = d_temp;
        temp_storage_bytes[dev_num] = temp_bytes;
        std::cout << "Malloc temp_storage. " << temp_storage_bytes[dev_num] << " bytes" << std::endl;
        std::cout << "d_temp_storage[" << dev_num << "] = " << d_temp_storage[dev_num] << std::endl;
        if (err) {
            std::cout << "Error " << err << " in SortPairs 1" << std::endl;
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        //}
        // Run sorting operation
        err = cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys,
                                              d_values, SIZE, 0, sizeof(T) * 8, this_stream);
        if (err) std::cout << "Error in SortPairs 2" << std::endl;
        //cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys,
        //                                d_values, SIZE, 0, sizeof(T)*8, this_stream);

    }

    template<typename T>
    void sum_reduce(thrust::device_vector <T> &values, T *sum) {
        int dev_num;
        safe_cuda(cudaGetDevice(&dev_num));
        if (!d_temp_storage2[dev_num]) {
            cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num],
                                   thrust::raw_pointer_cast(values.data()),
                                   sum, values.size(), cuda_stream[dev_num]);
            // Allocate temporary storage for sorting operation
            safe_cuda(cudaMalloc(&d_temp_storage2[dev_num], temp_storage_bytes2[dev_num]));
        }
        cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num],
                               thrust::raw_pointer_cast(values.data()),
                               sum, values.size(), cuda_stream[dev_num]);
    }
}
