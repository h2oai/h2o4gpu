/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <iostream>
#include <cstdlib>
#include "h2o4gpukmeans.h"
#include <random>
#include <algorithm>
#include <vector>
//#include "mkl.h"
#include "cblas.h"
#include <atomic>
#include <csignal>

#define VERBOSE 1

#include "h2o4gpukmeans_kmeanscpu.h"

//  FIXME: This kmeans back-end is not working.
//  FIXME: not really using seed.  Need to follow like gpu code.

template<typename T>
void random_data(int verbose, std::vector <T> &array, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        array[i] = (T) rand() / (T) RAND_MAX;
    }
}

template<typename T>
void nonrandom_data(int verbose, const char ord, std::vector <T> &array, const T *srcdata, int q, int n, int npercpu,
                    int d) {
    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        int indexi, indexj;
        for (int i = 0; i < npercpu * d; i++) {
#if(1)
            indexi = i % d; // col
            indexj = i / d + q * npercpu; // row (shifted by which cpu)
            //      array[i] = srcdata[indexi*n + indexj];
            array[i] = srcdata[indexi * n + indexj];
#else
            indexj = i%d;
            indexi = i/d;
            //      array[i] = srcdata[indexi*n + indexj];
            array[i] = srcdata[indexi*d + indexj];
#endif
        }
#if(DEBUGKMEANS)
        for(int i = 0; i < npercpu; i++) {
          for(int j = 0; j < d; j++) {
            fprintf(stderr,"q=%d initdata[%d,%d]=%g\n",q,i,j,array[i*d+j]); fflush(stderr);
          }
        }
#endif
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < npercpu * d; i++) {
            array[i] = srcdata[q * npercpu * d + i]; // shift by which cpu
        }
    }
}

template<typename T>
void
nonrandom_data_new(int verbose, std::vector<int> v, const char ord, std::vector <T> &array, const T *srcdata, int q,
                   int n, int npercpu, int d) {

    if (ord == 'c') {
        if (verbose) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        for (int i = 0; i < npercpu; i++) {
            for (int j = 0; j < d; j++) {
                array[i * d + j] = srcdata[v[q * npercpu + i] + j * n]; // shift by which cpu
            }
        }
#if(DEBUGKMEANS)
        for(int i = 0; i < npercpu; i++) {
          for(int j = 0; j < d; j++) {
            fprintf(stderr,"q=%d initdata[%d,%d]=%g\n",q,i,j,array[i*d+j]); fflush(stderr);
          }
        }
#endif
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < npercpu; i++) {
            for (int j = 0; j < d; j++) {
                array[i * d + j] = srcdata[v[q * npercpu + i] * d + j]; // shift by which cpu
            }
        }
    }
}

void random_labels(int verbose, std::vector<int> &labels, int n, int k) {
    for (int i = 0; i < n; i++) {
        labels[i] = rand() % k;
    }
}

template<typename T>
void random_centroids(int verbose, const char ord, std::vector <T> &array, const T *srcdata, int q, int n, int npercpu,
                      int d, int k) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    //  std::uniform_int_distribution<>dis(0, npercpu-1); // random i in range from 0..npercpu-1
    std::uniform_int_distribution<> dis(0, n - 1); // random i in range from 0..n-1 (i.e. only 1 cpu gets centroids)

    if (ord == 'c') {
        if (verbose) {
            if (verbose) {
                fprintf(stderr, "COL ORDER -> ROW ORDER\n");
                fflush(stderr);
            }
        }
        for (int i = 0; i < k; i++) { // rows
            int reali = dis(gen); // + q*npercpu; // row sampled (called indexj above)
            for (int j = 0; j < d; j++) { // cols
                array[i * d + j] = srcdata[reali + j * n];
#if(DEBUGKMEANS)
                fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,reali,array[i*d+j]); fflush(stderr);
#endif
            }
        }
    } else {
        if (verbose) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            int reali = dis(gen); // + q*npercpu ; // row sampled
            for (int j = 0; j < d; j++) { // cols
                array[i * d + j] = srcdata[reali * d + j];
            }
        }
    }
}

template<typename T>
void random_centroids_new(int verbose, std::vector<int> v, const char ord, std::vector <T> &array,
                          const T *srcdata, int q, int n, int npercpu, int d, int k) {
    if (ord == 'c') {
        if (VERBOSE) {
            fprintf(stderr, "COL ORDER -> ROW ORDER\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            for (int j = 0; j < d; j++) { // cols
                array[i * d + j] = srcdata[v[i] + j * n];
#if(DEBUGKMEANS)
                fprintf(stderr,"q=%d initcent[%d,%d reali=%d]=%g\n",q,i,j,v[i],array[i*d+j]); fflush(stderr);
#endif
            }
        }
    } else {
        if (VERBOSE) {
            fprintf(stderr, "ROW ORDER not changed\n");
            fflush(stderr);
        }
        for (int i = 0; i < k; i++) { // rows
            for (int j = 0; j < d; j++) { // cols
                array[i * d + j] = srcdata[v[i] * d + j];
            }
        }
    }
}

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace h2o4gpukmeans {
    volatile std::atomic_int flag(0);

    inline void my_function(int sig) { // can be called asynchronously
        fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
        flag = 1;
    }

    template<typename T>
    H2O4GPUKMeansCPU<T>::H2O4GPUKMeansCPU(const T *A, int k, int n, int d) {
        _A = A;
        _k = k;
        _n = n;
        _d = d;
    }

    template<typename T>
    int kmeans_fit(int verbose, int seed, int cpu_idtry, int n_cputry,
                   size_t rows, size_t cols, const char ord, int k, int max_iterations,
                   int init_from_data, T threshold,
                   const T *srcdata, T **pred_centroids, int **pred_labels) {
        if (rows > std::numeric_limits<int>::max()) {
            fprintf(stderr, "rows > %d not implemented\n", std::numeric_limits<int>::max());
            fflush(stderr);
            exit(0);
        }

        int n = rows;
        int d = cols;
        std::signal(SIGINT, my_function);
        std::signal(SIGTERM, my_function);

        if (verbose) {
            for (int ii = 0; ii < n; ii++) {
                for (int jj = 0; jj < d; jj++) {
                    fprintf(stderr, "%2g ", srcdata[ii * d + jj]);
                }
                fprintf(stderr, " |  ");
            }
            fflush(stderr);
        }

        int n_cpu = 1; // ignore try
        int cpu_id = 0; // ignore try
        int n_cpuvis = n_cpu; // fake

        // setup CPU list to use
        std::vector<int> dList(n_cpu);
        for (int idx = 0; idx < n_cpu; idx++) {
            int device_idx = (cpu_id + idx) % n_cpuvis;
            dList[idx] = device_idx;
        }

        double t0t = timer<double>();
        std::vector <T> *data[n_cpu];
        std::vector<int> *labels[n_cpu];
        std::vector <T> *l_centroids[n_cpu];
        std::vector <T> *distances[n_cpu];
        for (int q = 0; q < n_cpu; q++) {
            data[q] = new std::vector<T>(n / n_cpu * d);
            labels[q] = new std::vector<int>(n / n_cpu * d);
            l_centroids[q] = new std::vector<T>(k * d);
            distances[q] = new std::vector<T>(n);
        }

        std::cout << "Number of points: " << n << std::endl;
        std::cout << "Number of dimensions: " << d << std::endl;
        std::cout << "Number of clusters: " << k << std::endl;
        std::cout << "Max. number of iterations: " << max_iterations << std::endl;
        std::cout << "Stopping threshold: " << threshold << std::endl;

        // setup random sequence for sampling data
        //      std::random_device rd;
        //      std::mt19937 g(rd());
        std::vector<int> v(n);
        std::iota(std::begin(v), std::end(v), 0); // Fill with 0, 1, ..., 99.
        std::random_shuffle(v.begin(), v.end());

        for (int q = 0; q < n_cpu; q++) {
            nonrandom_data(verbose, ord, *data[q], &srcdata[0], q, n, n / n_cpu, d);
        }
        // get non-random centroids on 1 cpu, then share with rest.
        if (init_from_data == 0) {
            int masterq = 0;
            //random_centroids(verbose, ord, *centroids[masterq], &srcdata[0], masterq, n, n/n_cpu, d, k);
            random_centroids_new(verbose, v, ord, *l_centroids[masterq], &srcdata[0], masterq, n, n / n_cpu, d, k);
#if(DEBUGKMEANS)
            for (int q = 0; q < n_cpu; q++) {
				std::vector<T> h_centroidq=*l_centroids[q];
				for(int ii=0;ii<k*d;ii++){
				  fprintf(stderr,"q=%d initcent[%d]=%g\n",q,ii,h_centroidq[ii]); fflush(stderr);
				}
            }
#endif
        }
        double timetransfer = static_cast<double>(timer<double>() - t0t);


        double t0 = timer<double>();
        int masterq = 0;
        kmeans::kmeans<T>(verbose, &flag, n, d, k, *data[masterq], *labels[masterq],
                          *l_centroids[masterq], max_iterations, init_from_data, threshold);

        double timefit = static_cast<double>(timer<double>() - t0);

        std::cout << "  Time fit: " << timefit << " s" << std::endl;
        fprintf(stderr, "Timetransfer: %g Timefit: %g\n", timetransfer, timefit);
        fflush(stderr);

        // copy result of centroids (sitting entirely on each device) back to host
        std::vector <T> *ctr = new std::vector<T>(*l_centroids[0]);
        *pred_centroids = ctr->data();

        std::vector <int> *lbls = new std::vector<int>(*labels[0]);
        *pred_labels = lbls->data();

        // debug
        int printcenters = 0;
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

        // done with CPU data
        for (int q = 0; q < n_cpu; q++) {
            delete (data[q]);
            delete (labels[q]);
            delete (l_centroids[q]);
            delete (distances[q]);
        }

        return 0;
    }

    template<typename T>
    int kmeans_predict(int verbose, int cpu_idtry, int n_cputry,
                       size_t rows, size_t cols,
                       const char ord, int k,
                       const T *srcdata, const T *centroids, int **pred_labels) {
        if (rows > std::numeric_limits<int>::max()) {
            fprintf(stderr, "rows>%d now implemented\n", std::numeric_limits<int>::max());
            fflush(stderr);
            exit(0);
        }

        int n = rows;
        int m = cols;
        std::signal(SIGINT, my_function);
        std::signal(SIGTERM, my_function);

        int n_cpu = 1;
        int cpu_id = 0;
        int n_cpuvis = n_cpu;

        std::vector<int> dList(n_cpu);
        for (int idx = 0; idx < n_cpu; idx++) {
            int device_idx = (cpu_id * idx) % n_cpuvis;
            dList[idx] = device_idx;
        }

        std::vector <T> *data[n_cpu];
        std::vector<int> *labels[n_cpu];

        std::vector<T> *l_centroids = new std::vector<T>(k * m);
        std::vector<T> *pairwise_distances[n_cpu];
        std::vector<T> *data_dots[n_cpu];
        std::vector<T> *centroid_dots[n_cpu];

        nonrandom_data(verbose, 'r', *l_centroids, &centroids[0], 0, k, k, m);

        for (int q = 0; q < n_cpu; q++) {
            data[q] = new std::vector<T>(n/n_cpu * m);
            nonrandom_data(verbose, ord, *data[q], &srcdata[0], q, n, n/n_cpu, m);

            data_dots[q] = new std::vector<T>(n/n_cpu);
            centroid_dots[q] = new std::vector<T>(k);

            pairwise_distances[q] = new std::vector<T>(n/n_cpu * k);

            kmeans::self_dot(*data[q], n/n_cpu, m, *data_dots[q]);

            kmeans::compute_distances(*data[q], *data_dots[q], n/n_cpu, m, *l_centroids, *centroid_dots[q],
                                    k, *pairwise_distances[q]);

            labels[q] = new std::vector<int>(n/n_cpu);
            kmeans::relabel(*data[q], n/n_cpu, *pairwise_distances[q], k, *labels[q]);
        }

        std::vector<int> *ctr = new std::vector<int>(*labels[0]);
        *pred_labels = ctr->data();

        for (int q = 0; q < n_cpu; q++) {
            delete(data[q]);
//            delete(labels[q]);
            delete(pairwise_distances[q]);
            delete(data_dots[q]);
            delete(centroid_dots[q]);
        }

        return 0;
    }

    template<typename T>
    int kmeans_transform(int verbose,
                            int gpu_id, int n_gpu,
                            size_t rows, size_t cols, const char ord, int k,
                            const T* srcdata, const T* centroids,
                            T **preds) {
        if (rows > std::numeric_limits<int>::max()) {
            fprintf(stderr, "rows>%d now implemented\n", std::numeric_limits<int>::max());
            fflush(stderr);
            exit(0);
        }

        int n = rows;
        int m = cols;
        std::signal(SIGINT, my_function);
        std::signal(SIGTERM, my_function);

        int n_cpu = 1;
        int cpu_id = 0;
        int n_cpuvis = n_cpu;

        std::vector<int> dList(n_cpu);
        for (int idx = 0; idx < n_cpu; idx++) {
            int device_idx = (cpu_id * idx) % n_cpuvis;
            dList[idx] = device_idx;
        }

        std::vector <T> *data[n_cpu];

        std::vector<T> *l_centroids = new std::vector<T>(k * m);
        std::vector<T> *pairwise_distances[n_cpu];
        std::vector<T> *data_dots[n_cpu];
        std::vector<T> *centroid_dots[n_cpu];

        nonrandom_data(verbose, 'r', *l_centroids, &centroids[0], 0, k, k, m);

        for (int q = 0; q < n_cpu; q++) {
            data[q] = new std::vector<T>(n/n_cpu * m);
            nonrandom_data(verbose, ord, *data[q], &srcdata[0], q, n, n/n_cpu, m);

            data_dots[q] = new std::vector<T>(n/n_cpu);
            centroid_dots[q] = new std::vector<T>(k);

            pairwise_distances[q] = new std::vector<T>(n/n_cpu * k);

            kmeans::self_dot(*data[q], n/n_cpu, m, *data_dots[q]);

            kmeans::compute_distances(*data[q], *data_dots[q], n/n_cpu, m, *l_centroids, *centroid_dots[q],
                                    k, *pairwise_distances[q]);
        }

        std::vector<T> *ctr = new std::vector<T>(*pairwise_distances[0]);
        *preds = ctr->data();

        for (int q = 0; q < n_cpu; q++) {
            delete(data[q]);
            delete(pairwise_distances[q]);
            delete(data_dots[q]);
            delete(centroid_dots[q]);
        }

        return 0;
    }

    template<typename T>
    int makePtr_dense(int dopredict, int verbose, int seed, int cpu_idtry, int n_cputry, size_t rows, size_t cols,
                      const char ord, int k, int max_iterations, int init_from_data,
                      T threshold, const T *srcdata, const T *centroids,
                      T **pred_centroids, int **pred_labels) {
        if (dopredict == 0) {
            return kmeans_fit(verbose, seed, cpu_idtry, n_cputry, rows, cols,
                              ord, k, max_iterations, init_from_data, threshold,
                              srcdata, pred_centroids, pred_labels);
        } else {
            return kmeans_predict(verbose, cpu_idtry, n_cputry, rows, cols,
                                  ord, k,
                                  srcdata, centroids, pred_labels);
        }
    }

    template
    int makePtr_dense<float>(int dopredict, int verbose, int seed, int cpu_idtry, int n_cputry, size_t rows, size_t cols,
                             const char ord, int k, int max_iterations, int init_from_data,
                             float threshold, const float *srcdata,
                             const float *centroids, float **pred_centroids, int **pred_labels);

    template
    int makePtr_dense<double>(int dopredict, int verbose, int seed, int cpu_idtry, int n_cputry, size_t rows, size_t cols,
                              const char ord, int k, int max_iterations, int init_from_data,
                              double threshold, const double *srcdata,
                              const double *centroids, double **pred_centroids, int **pred_labels);

    template
    int kmeans_fit<float>(int verbose, int seed, int cpu_idtry, int n_cputry,
                          size_t rows, size_t cols,
                          const char ord, int k, int max_iterations,
                          int init_from_data, float threshold,
                          const float *srcdata, float **pred_centroids, int **pred_labels);

    template
    int kmeans_fit<double>(int verbose, int seed, int cpu_idtry, int n_cputry,
                           size_t rows, size_t cols,
                           const char ord, int k, int max_iterations,
                           int init_from_data, double threshold,
                           const double *srcdata, double **pred_centroids, int **pred_labels);

    template
    int kmeans_predict<float>(int verbose, int cpu_idtry, int n_cputry,
                              size_t rows, size_t cols,
                              const char ord, int k,
                              const float *srcdata, const float *centroid, int **pred_labels);

    template
    int kmeans_predict<double>(int verbose, int cpu_idtry, int n_cputry,
                               size_t rows, size_t cols,
                               const char ord, int k,
                               const double *srcdata, const double *centroid, int **pred_labels);

    template
    int kmeans_transform<float>(int verbose,
                                int gpu_id, int n_gpu,
                                size_t m, size_t n, const char ord, int k,
                                const float * src_data, const float * centroids,
                                float **preds);

    template
    int kmeans_transform<double>(int verbose,
                                 int gpu_id, int n_gpu,
                                 size_t m, size_t n, const char ord, int k,
                                 const double * src_data, const double * centroids,
                                 double **preds);

// Explicit template instantiation.
#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE == 1

    template
    class H2O4GPUKMeansCPU<double>;

#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE == 1

    template
    class H2O4GPUKMeansCPU<float>;

#endif

}  // namespace h2o4gpukmeans

// Fit and Predict
int make_ptr_float_kmeans(int dopredict, int verbose, int seed, int cpu_id, int n_cpu, size_t mTrain, size_t n,
                          const char ord, int k, int max_iterations, int init_from_data,
                          float threshold, const float *srcdata,
                          const float *centroids, float **pred_centroids, int **pred_labels) {
    return h2o4gpukmeans::makePtr_dense<float>(dopredict, verbose, seed, cpu_id, n_cpu, mTrain, n, ord, k,
                                                max_iterations, init_from_data, threshold,
                                                srcdata, centroids,
                                                pred_centroids, pred_labels);
}

int make_ptr_double_kmeans(int dopredict, int verbose, int seed, int cpu_id, int n_cpu, size_t mTrain, size_t n,
                           const char ord, int k, int max_iterations, int init_from_data,
                           double threshold, const double *srcdata,
                           const double *centroids, double **pred_centroids, int **pred_labels) {
    return h2o4gpukmeans::makePtr_dense<double>(dopredict, verbose, seed, cpu_id, n_cpu, mTrain, n, ord, k,
                                                 max_iterations, init_from_data, threshold,
                                                 srcdata, centroids,
                                                 pred_centroids, pred_labels);
}

// Transform
int kmeans_transform_float(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const float * src_data, const float * centroids,
                           float **preds) {
    return h2o4gpukmeans::kmeans_transform<float>(verbose, gpu_id, n_gpu, m, n, ord, k, src_data, centroids, preds);
}

int kmeans_transform_double(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const double * src_data, const double * centroids,
                           double **preds) {
    return h2o4gpukmeans::kmeans_transform<double>(verbose, gpu_id, n_gpu, m, n, ord, k, src_data, centroids, preds);
}


