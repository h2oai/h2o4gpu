#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include <unistd.h>
#include "h2ogpumlkmeans.h"
#include "kmeans.h"
#include <random>
#include <algorithm>
#include <vector>
#include "include/kmeans_general.h"
#include <csignal>

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)


template<typename T>
void fill_array(T &array, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            array[i * n + j] = (i % 2) * 3 + j;
        }
    }
}

template<typename T>
void random_data(int verbose, thrust::device_vector <T> &array, int m, int n) {
    thrust::host_vector <T> host_array(m * n);
    for (int i = 0; i < m * n; i++) {
        host_array[i] = (T) rand() / (T) RAND_MAX;
    }
    array = host_array;
}

template<typename T>
void nonrandom_data(int verbose, const char ord, thrust::device_vector <T> &array, const T *srcdata, int q, int n,
                    int npergpu, int d) {
    thrust::host_vector <T> host_array(npergpu * d);
    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        int indexi, indexj;
        for (int i = 0; i < npergpu * d; i++) {
#if(1)
            indexi = i % d; // col
            indexj = i / d + q * npergpu; // row (shifted by which gpu)
            //      host_array[i] = srcdata[indexi*n + indexj];
            host_array[i] = srcdata[indexi * n + indexj];
#else
            indexj = i%d;
            indexi = i/d;
            //      host_array[i] = srcdata[indexi*n + indexj];
            host_array[i] = srcdata[indexi*d + indexj];
#endif
        }
#if(DEBUGKMEANS)
        for(int i = 0; i < npergpu; i++) {
          for(int j = 0; j < d; j++) {
            fprintf(stderr,"q=%d initdata[%d,%d]=%g\n",q,i,j,host_array[i*d+j]); fflush(stderr);
          }
        }
#endif
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < npergpu * d; i++) {
            host_array[i] = srcdata[q * npergpu * d + i]; // shift by which gpu
        }
    }
    array = host_array;
}

template<typename T>
void nonrandom_data_new(int verbose, std::vector<int> v, const char ord, thrust::device_vector <T> &array,
                        const T *srcdata, int q, int n, int npergpu, int d) {
    thrust::host_vector <T> host_array(npergpu * d);

    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        for (int i = 0; i < npergpu; i++) {
            for (int j = 0; j < d; j++) {
                host_array[i * d + j] = srcdata[v[q * npergpu + i] + j * n]; // shift by which gpu
            }
        }
#if(DEBUGKMEANS)
        for(int i = 0; i < npergpu; i++) {
          for(int j = 0; j < d; j++) {
            fprintf(stderr,"q=%d initdata[%d,%d]=%g\n",q,i,j,host_array[i*d+j]); fflush(stderr);
          }
        }
#endif
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < npergpu; i++) {
            for (int j = 0; j < d; j++) {
                host_array[i * d + j] = srcdata[v[q * npergpu + i] * d + j]; // shift by which gpu
            }
        }
    }
    array = host_array;
}

void random_labels(int verbose, thrust::device_vector<int> &labels, int n, int k) {
    thrust::host_vector<int> host_labels(n);
    for (int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}

void nonrandom_labels(int verbose, const char ord, thrust::device_vector<int> &labels, const int *srclabels,
                      int q, int n, int npergpu) {
    thrust::host_vector<int> host_labels(npergpu);
    int d = 1; // only 1 dimension
    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "labels COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        int indexi, indexj;
        for (int i = 0; i < npergpu; i++) {
#if(1)
            indexi = i % d; // col
            indexj = i / d + q * npergpu; // row (shifted by which gpu)
            //      host_labels[i] = srclabels[indexi*n + indexj];
            host_labels[i] = srclabels[indexi * n + indexj];
#else
            indexj = i%d;
            indexi = i/d;
            //      host_labels[i] = srclabels[indexi*n + indexj];
            host_labels[i] = srclabels[indexi*d + indexj];
#endif
        }
    } else {
        if (verbose) {
            fprintf(stderr, "labels ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < npergpu; i++) {
            host_labels[i] = srclabels[q * npergpu * d + i]; // shift by which gpu
        }
    }
    labels = host_labels;
}

template<typename T>
void random_centroids(int verbose, int seed, const char ord,
                      thrust::device_vector <T> &array, const T *srcdata,
                      int q, int n, int npergpu, int d, int k) {
    thrust::host_vector <T> host_array(k * d);
    if (seed >= 0) {
    } else {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        seed = rd();
    }
    std::mt19937 gen(seed);
    //  std::uniform_int_distribution<>dis(0, npergpu-1); // random i in range from 0..npergpu-1
    std::uniform_int_distribution<> dis(0, n - 1); // random i in range from 0..n-1 (i.e. only 1 gpu gets centroids)

    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            int reali = dis(gen); // + q*npergpu; // row sampled (called indexj above)
            for (int j = 0; j < d; j++) { // cols
                host_array[i * d + j] = srcdata[reali + j * n];
#if(DEBUGKMEANS)
                fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,reali,host_array[i*d+j]); fflush(stderr);
#endif
            }
        }
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            int reali = dis(gen); // + q*npergpu ; // row sampled
            for (int j = 0; j < d; j++) { // cols
                host_array[i * d + j] = srcdata[reali * d + j];
            }
        }
    }
    array = host_array;
}

template<typename T>
void random_centroids_new(int verbose, std::vector<int> v, const char ord, thrust::device_vector <T> &array,
                          const T *srcdata, int q, int n, int npergpu, int d, int k) {
    thrust::host_vector <T> host_array(k * d);

    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            for (int j = 0; j < d; j++) { // cols
                host_array[i * d + j] = srcdata[v[i] + j * n];
#if(DEBUGKMEANS)
                fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,v[i],host_array[i*d+j]); fflush(stderr);
#endif
            }
        }
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            for (int j = 0; j < d; j++) { // cols
                host_array[i * d + j] = srcdata[v[i] * d + j];
            }
        }
    }
    array = host_array;
}

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace h2ogpumlkmeans {
    volatile std::atomic_int flaggpu(0);

    inline void my_function_gpu(int sig) { // can be called asynchronously
        fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
        flaggpu = 1;
    }

    template<typename T>
    H2OGPUMLKMeans<T>::H2OGPUMLKMeans(const T *A, int k, int n, int d) {
        _A = A;
        _k = k;
        _n = n;
        _d = d;
    }

    template<typename T>
    int kmeans_fit(int verbose, int seed, int gpu_idtry, int n_gputry,
                   size_t rows, size_t cols, const char ord,
                   int k, int max_iterations, int init_from_labels,
                   int init_labels, int init_data, T threshold,
                   const T *srcdata, const int *srclabels, void **pred_centroids, void **pred_labels) {
        if (verbose) {
            std::cout << " Start makePtr_dense." << std::endl;
        }

        // init random seed if use the C function rand()
        if (seed >= 0) {
            srand(seed);
        } else {
            srand(unsigned(time(NULL)));
        }

        if (verbose) {
            std::cout << "Seed: " << seed << std::endl;
        }
        // no more clusters than rows
        if (k > rows) {
            k = static_cast<int>(rows);
            fprintf(stderr, "Number of clusters adjusted to be equal to number of rows.\n");
            fflush(stderr);
        }

        if(rows>std::numeric_limits<int>::max()){
            fprintf(stderr,"rows>%d now implemented\n",std::numeric_limits<int>::max());
            fflush(stderr);
            exit(0);
        }

        int n = rows;
        int d = cols;

        std::signal(SIGINT, my_function_gpu);
        std::signal(SIGTERM, my_function_gpu);

        int printsrcdata = 0;
        if (printsrcdata) {
            for (unsigned int ii = 0; ii < n; ii++) {
                for (unsigned int jj = 0; jj < d; jj++) {
                    fprintf(stderr, "%2g ", srcdata[ii * d + jj]);
                }
                fprintf(stderr, " |  ");
            }
            fflush(stderr);
        }

        // no more gpus than visible gpus
        int n_gpuvis;
        cudaGetDeviceCount(&n_gpuvis);
        int n_gpu;
        n_gpu = std::min(n_gpuvis, n_gputry);

        // also no more than rows
        n_gpu = std::min(n_gpu, n);

        if (verbose) {
            std::cout << n_gpu << " gpus." << std::endl;
        }

        int gpu_id;
        gpu_id = gpu_idtry % n_gpuvis;

        // setup GPU list to use
        std::vector<int> dList(n_gpu);
        for (int idx = 0; idx < n_gpu; idx++) {
            int device_idx = (gpu_id + idx) % n_gpuvis;
            dList[idx] = device_idx;
        }

        double t0t = timer<double>();
        thrust::device_vector <T> *data[n_gpu];
        thrust::device_vector<int> *labels[n_gpu];
        thrust::device_vector <T> *d_centroids[n_gpu];
        thrust::device_vector <T> *distances[n_gpu];

        if (verbose) {
            fprintf(stderr, "Before allocation");
            fflush(stderr);
            //sleep(5);
        }

        for (int q = 0; q < n_gpu; q++) {
            CUDACHECK(cudaSetDevice(dList[q]));
            data[q] = new thrust::device_vector<T>(n / n_gpu * d);
            labels[q] = new thrust::device_vector<int>(n / n_gpu * d); // TODO should this size really be multiplied by d??
            d_centroids[q] = new thrust::device_vector<T>(k * d);
            distances[q] = new thrust::device_vector<T>(n);
        }

        if (verbose) {
            std::cout << "Number of points: " << n << std::endl;
            std::cout << "Number of dimensions: " << d << std::endl;
            std::cout << "Number of clusters: " << k << std::endl;
            std::cout << "Max. number of iterations: " << max_iterations << std::endl;
            std::cout << "Stopping threshold: " << threshold << std::endl;
            //sleep(5);
        }

        // setup random sequence for sampling data
        //      std::random_device rd;
        //      std::mt19937 g(rd());
        std::vector<int> v(n);
        std::iota(std::begin(v), std::end(v), 0); // Fill with 0, 1, ..., 99.

        if (seed >= 0) {
            std::shuffle(v.begin(), v.end(), std::default_random_engine(seed));
        } else {
            std::random_shuffle(v.begin(), v.end());
        }

        for (int q = 0; q < n_gpu; q++) {
            CUDACHECK(cudaSetDevice(dList[q]));
            if (verbose) { std::cout << "Copying data to device: " << dList[q] << std::endl; }

            if (init_labels == 0) { // random
                random_labels(verbose, *labels[q], n / n_gpu, k);
            } else {
                nonrandom_labels(verbose, ord, *labels[q], &srclabels[0], q, n, n / n_gpu);
            }

            if (init_data == 0) { // random (for testing)
                random_data<T>(verbose, *data[q], n / n_gpu, d);
            } else if (init_data == 1) { // shard by row
                nonrandom_data(verbose, ord, *data[q], &srcdata[0], q, n, n / n_gpu, d);
            } else { // shard by randomly (without replacement) selected by row
                nonrandom_data_new(verbose, v, ord, *data[q], &srcdata[0], q, n, n / n_gpu, d);
            }
        }

        // get non-random centroids on 1 gpu, then share with rest.
        if (init_from_labels == 0) {
            int masterq = 0;
            CUDACHECK(cudaSetDevice(dList[masterq]));
            //random_centroids(ord, *centroids[masterq], &srcdata[0], masterq, n, n/n_gpu, d, k);
            random_centroids_new(verbose, v, ord, *d_centroids[masterq], &srcdata[0], masterq, n, n / n_gpu, d, k);
            int bytecount = d * k * sizeof(T); // all centroids

            // copy centroids to rest of gpus asynchronously
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
                cudaStreamSynchronize(*(streams[q]));
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
        }

        double timetransfer = static_cast<double>(timer<double>() - t0t);

        if (verbose) {
            fprintf(stderr, "Before kmeans() call\n");
            fflush(stderr);
        }

        double t0 = timer<double>();

        int status = kmeans::kmeans<T>(verbose, &flaggpu, n, d, k, data, labels, d_centroids, distances, dList, n_gpu,
                                       max_iterations, init_from_labels, threshold);
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
        thrust::host_vector <T> *ctr = new thrust::host_vector<T>(*d_centroids[0]);
        // TODO FIXME: When do delete this ctr memory?
        //      cudaMemcpy(ctr->data().get(), centroids[0]->data().get(), sizeof(T)*k*d, cudaMemcpyDeviceToHost);
        *pred_centroids = ctr->data();

        // copy assigned labels
        thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(0);
        for (int q = 0; q < n_gpu; q++) {
            h_labels->insert(h_labels->end(), labels[q]->begin(), labels[q]->end());
        }
        *pred_labels = h_labels->data();

        // debug
        int printcenters = verbose > 2;
        if (printcenters) {
            for (unsigned int ii = 0; ii < k; ii++) {
                fprintf(stderr, "ii=%d of k=%d ", ii, k);
                for (unsigned int jj = 0; jj < d; jj++) {
                    fprintf(stderr, "%g ", (*ctr)[d * ii + jj]);
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }
        }

        for (int q = 0; q < n_gpu; q++) {
            delete(data[q]);
            delete(labels[q]);
            delete(d_centroids[q]);
            delete(distances[q]);
        }

        return 0;
    }

    template<typename T>
    int kmeans_predict(int verbose, int gpu_idtry, int n_gputry,
                       size_t rows, size_t cols,
                       const char ord, int k,
                       const T* srcdata, const T* centroids, void** pred_labels) {
        if (rows > std::numeric_limits<int>::max()) {
            fprintf(stderr, "rows > %d not implemented\n", std::numeric_limits<int>::max());
            fflush(stderr);
            exit(0);
        }

        int n = rows;
        int m = cols;
        std::signal(SIGINT, my_function_gpu);
        std::signal(SIGTERM, my_function_gpu);

        // no more gpus than visible gpus
        int n_gpuvis;
        cudaGetDeviceCount(&n_gpuvis);
        int n_gpu;
        n_gpu = std::min(n_gpuvis, n_gputry);

        // also no more than rows
        n_gpu = std::min(n_gpu, n);

        std::cout << n_gpu << " gpus." << std::endl;

        int gpu_id;
        gpu_id = gpu_idtry % n_gpuvis;

        // setup GPU list to use
        std::vector<int> dList(n_gpu);
        for (int idx = 0; idx < n_gpu; idx++) {
            int device_idx = (gpu_id + idx) % n_gpuvis;
            dList[idx] = device_idx;
        }

        double t0t = timer<double>();
        thrust::device_vector <T> *d_data[n_gpu];
        thrust::device_vector<int> *d_labels[n_gpu];
        thrust::device_vector<T> *d_centroids[n_gpu];
        thrust::device_vector<T> *pairwise_distances[n_gpu];
        thrust::device_vector<T> *data_dots[n_gpu];
        thrust::device_vector<T> *centroid_dots[n_gpu];
        thrust::device_vector<T> *distances[n_gpu];
        int *d_changes[n_gpu];


        for (int q = 0; q < n_gpu; q++) {
            CUDACHECK(cudaSetDevice(dList[q]));
            kmeans::detail::labels_init();

            data_dots[q] = new thrust::device_vector <T>(n/n_gpu);
            centroid_dots[q] = new thrust::device_vector<T>(k);
            pairwise_distances[q] = new thrust::device_vector<T>(n / n_gpu * k);

            d_centroids[q] = new thrust::device_vector<T>(k * m);
            d_data[q] = new thrust::device_vector<T>(n/n_gpu * m);
            distances[q] = new thrust::device_vector<T>(n);
            d_labels[q] = new thrust::device_vector<int>(n/n_gpu);

            cudaMalloc(&d_changes[q], sizeof(int));

            // Move centroids from host memory to GPU
            std::cout << "Copying centroids and data to device: " << dList[q] << std::endl;
            nonrandom_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, m);

            nonrandom_data(verbose, ord, *d_data[q], &srcdata[0], q, n, n/n_gpu, m);

            kmeans::detail::make_self_dots(n/n_gpu, m, *d_data[q], *data_dots[q]);

            kmeans::detail::calculate_distances(verbose, q, n/n_gpu, m, k,
                *d_data[q], *d_centroids[q], *data_dots[q],
                *centroid_dots[q], *pairwise_distances[q]);

            kmeans::detail::relabel(n/n_gpu, k, *pairwise_distances[q], *d_labels[q], *distances[q], d_changes[q]);
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
            delete(d_labels[q]);
            delete(pairwise_distances[q]);
            delete(data_dots[q]);
            delete(centroid_dots[q]);
            delete(d_centroids[q]);
            delete(d_data[q]);
            delete(distances[q]);
        }

        return 0;
    }

    template<typename T>
    int kmeans_transform(int verbose,
                                int gpu_idtry, int n_gputry,
                                size_t rows, size_t cols, const char ord, int k,
                                const T* srcdata, const T* centroids,
                                void **preds) {
        if (rows > std::numeric_limits<int>::max()) {
            fprintf(stderr, "rows > %d not implemented\n", std::numeric_limits<int>::max());
            fflush(stderr);
            exit(0);
        }

        int n = rows;
        int m = cols;
        std::signal(SIGINT, my_function_gpu);
        std::signal(SIGTERM, my_function_gpu);

        // no more gpus than visible gpus
        int n_gpuvis;
        cudaGetDeviceCount(&n_gpuvis);
        int n_gpu;
        n_gpu = std::min(n_gpuvis, n_gputry);

        // also no more than rows
        n_gpu = std::min(n_gpu, n);

        std::cout << n_gpu << " gpus." << std::endl;

        int gpu_id = gpu_idtry % n_gpuvis;

        // setup GPU list to use
        std::vector<int> dList(n_gpu);
        for (int idx = 0; idx < n_gpu; idx++) {
            int device_idx = (gpu_id + idx) % n_gpuvis;
            dList[idx] = device_idx;
        }

        thrust::device_vector <T> *d_data[n_gpu];
        thrust::device_vector<T> *d_centroids[n_gpu];
        thrust::device_vector<T> *d_pairwise_distances[n_gpu];
        thrust::device_vector<T> *data_dots[n_gpu];
        thrust::device_vector<T> *centroid_dots[n_gpu];

        for (int q = 0; q < n_gpu; q++) {
            CUDACHECK(cudaSetDevice(dList[q]));
            kmeans::detail::labels_init();

            data_dots[q] = new thrust::device_vector <T>(n/n_gpu);
            centroid_dots[q] = new thrust::device_vector<T>(k);
            d_pairwise_distances[q] = new thrust::device_vector<T>(n / n_gpu * k);

            d_centroids[q] = new thrust::device_vector<T>(k * m);
            d_data[q] = new thrust::device_vector<T>(n/n_gpu * m);

            // Move centroids from host memory to GPU
            std::cout << "Copying centroids and data to device: " << dList[q] << std::endl;
            nonrandom_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, m);

            nonrandom_data(verbose, ord, *d_data[q], &srcdata[0], q, n, n/n_gpu, m);

            kmeans::detail::make_self_dots(n/n_gpu, m, *d_data[q], *data_dots[q]);

            kmeans::detail::calculate_distances(verbose, q, n/n_gpu, m, k,
                *d_data[q], *d_centroids[q], *data_dots[q],
                *centroid_dots[q], *d_pairwise_distances[q]);
        }

        // Move the resulting labels into host memory from all devices
        thrust::host_vector<T> *h_pairwise_distances = new thrust::host_vector<T>(0);
        for (int q = 0; q < n_gpu; q++) {
            h_pairwise_distances->insert(h_pairwise_distances->end(), d_pairwise_distances[q]->begin(), d_pairwise_distances[q]->end());
        }
        *preds = h_pairwise_distances->data();

        for (int q = 0; q < n_gpu; q++) {
            safe_cuda(cudaSetDevice(dList[q]));
            kmeans::detail::labels_close();
            delete(d_pairwise_distances[q]);
            delete(data_dots[q]);
            delete(centroid_dots[q]);
            delete(d_centroids[q]);
            delete(d_data[q]);
        }



        return 0;
    }

    template<typename T>
    int makePtr_dense(int dopredict, int verbose, int seed, int gpu_idtry, int n_gputry, size_t rows, size_t cols,
                      const char ord, int k, int max_iterations, int init_from_labels, int init_labels, int init_data,
                      T threshold, const T *srcdata, const int *srclabels, const T *centroids,
                      void **pred_centroids, void **pred_labels) {
        if (dopredict == 0) {
            return kmeans_fit(verbose, seed, gpu_idtry, n_gputry, rows, cols,
                              ord, k, max_iterations, init_from_labels, init_labels, init_data, threshold,
                              srcdata, srclabels, pred_centroids, pred_labels);
        } else {
            return kmeans_predict(verbose, gpu_idtry, n_gputry, rows, cols,
                                  ord, k,
                                  srcdata, centroids, pred_labels);
        }
    }

    template int
    makePtr_dense<float>(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t rows, size_t cols,
                         const char ord, int k, int max_iterations, int init_from_labels, int init_labels,
                         int init_data, float threshold, const float *srcdata, const int *srclabels,
                         const float *centroids, void **pred_centroids, void **pred_labels);

    template int
    makePtr_dense<double>(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t rows, size_t cols,
                          const char ord, int k, int max_iterations, int init_from_labels, int init_labels,
                          int init_data, double threshold, const double *srcdata, const int *srclabels,
                          const double *centroids, void **pred_centroids, void **pred_labels);

    template int kmeans_fit<float>(int verbose, int seed, int gpu_idtry, int n_gputry,
                                   size_t rows, size_t cols,
                                   const char ord, int k, int max_iterations,
                                   int init_from_labels, int init_labels, int init_data, float threshold,
                                   const float *srcdata, const int *srclabels,
                                   void **pred_centroids, void **pred_labels);

    template int kmeans_fit<double>(int verbose, int seed, int gpu_idtry, int n_gputry,
                                    size_t rows, size_t cols,
                                    const char ord, int k, int max_iterations,
                                    int init_from_labels, int init_labels, int init_data, double threshold,
                                    const double *srcdata, const int *srclabels,
                                    void **pred_centroids, void **pred_labels);

    template int kmeans_predict<float>(int verbose, int gpu_idtry, int n_gputry,
                                        size_t rows, size_t cols,
                                        const char ord, int k,
                                        const float* srcdata, const float* centroids, void **pred_labels);

    template int kmeans_predict<double>(int verbose, int gpu_idtry, int n_gputry,
                                        size_t rows, size_t cols,
                                        const char ord, int k,
                                        const double *srcdata, const double *centroids, void **pred_labels);

    template int kmeans_transform<float>(int verbose,
                                         int gpu_id, int n_gpu,
                                         size_t m, size_t n, const char ord, int k,
                                         const float * src_data, const float * centroids,
                                         void **preds);

    template int kmeans_transform<double>(int verbose,
                                         int gpu_id, int n_gpu,
                                         size_t m, size_t n, const char ord, int k,
                                         const double * src_data, const double * centroids,
                                         void **preds);

// Explicit template instantiation.
#if !defined(H2OGPUML_DOUBLE) || H2OGPUML_DOUBLE == 1

    template
    class H2OGPUMLKMeans<double>;

#endif

#if !defined(H2OGPUML_SINGLE) || H2OGPUML_SINGLE == 1

    template
    class H2OGPUMLKMeans<float>;

#endif

}  // namespace h2ogpumlkmeans

#ifdef __cplusplus
extern "C" {
#endif

// Fit and Predict
int make_ptr_float_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                          const char ord, int k, int max_iterations, int init_from_labels, int init_labels,
                          int init_data, float threshold, const float *srcdata, const int *srclabels,
                          const float *centroids, void **pred_centroids, void **pred_labels) {
    return h2ogpumlkmeans::makePtr_dense<float>(dopredict, verbose, seed, gpu_id, n_gpu, mTrain, n, ord, k,
                                                max_iterations, init_from_labels, init_labels, init_data, threshold,
                                                srcdata, srclabels, centroids, pred_centroids, pred_labels);
}

int make_ptr_double_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                           const char ord, int k, int max_iterations, int init_from_labels, int init_labels,
                           int init_data, double threshold, const double *srcdata, const int *srclabels,
                           const double *centroids, void **pred_centroids, void **pred_labels) {
    return h2ogpumlkmeans::makePtr_dense<double>(dopredict, verbose, seed, gpu_id, n_gpu, mTrain, n, ord, k,
                                                 max_iterations, init_from_labels, init_labels, init_data, threshold,
                                                 srcdata, srclabels, centroids, pred_centroids, pred_labels);
}

// Transform
int kmeans_transform_float(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const float * src_data, const float * centroids,
                           void **preds) {
    return h2ogpumlkmeans::kmeans_transform<float>(verbose, gpu_id, n_gpu, m, n, ord, k, src_data, centroids, preds);
}

int kmeans_transform_double(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const double * src_data, const double * centroids,
                           void **preds) {
    return h2ogpumlkmeans::kmeans_transform<double>(verbose, gpu_id, n_gpu, m, n, ord, k, src_data, centroids, preds);
}

#ifdef __cplusplus
}
#endif