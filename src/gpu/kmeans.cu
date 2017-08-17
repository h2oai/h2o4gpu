// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#include <signal.h>
#include <string>
#include <sstream>
#include <thrust/reduce.h>
#include "include/kmeans_general.h"
#include "include/kmeans.h"

namespace kmeans {

    template<typename T>
    int kmeans(
            int verbose,
            volatile std::atomic_int *flag,
            int n, int d, int k,
            thrust::device_vector <T> **data,
            thrust::device_vector<int> **labels,
            thrust::device_vector <T> **centroids,
            thrust::device_vector <T> **distances,
            std::vector<int> dList,
            int n_gpu,
            int max_iterations,
            int init_from_labels,
            double threshold) {

        thrust::device_vector <T> *data_dots[MAX_NGPUS];
        thrust::device_vector <T> *centroid_dots[MAX_NGPUS];
        thrust::device_vector <T> *pairwise_distances[MAX_NGPUS];
        thrust::device_vector<int> *labels_copy[MAX_NGPUS];
        thrust::device_vector<int> *range[MAX_NGPUS];
        thrust::device_vector<int> *indices[MAX_NGPUS];
        thrust::device_vector<int> *counts[MAX_NGPUS];
        thrust::host_vector <T> h_centroids(k * d);
        thrust::host_vector <T> h_centroids_tmp(k * d);
        int h_changes[MAX_NGPUS], *d_changes[MAX_NGPUS];
        T h_distance_sum[MAX_NGPUS], *d_distance_sum[MAX_NGPUS];

        for (int q = 0; q < n_gpu; q++) {

            if (verbose) {
                fprintf(stderr, "Before kmeans() Allocation: gpu: %d\n", q);
                fflush(stderr);
            }

            safe_cuda(cudaSetDevice(dList[q]));
            safe_cuda(cudaMalloc(&d_changes[q], sizeof(int)));
            safe_cuda(cudaMalloc(&d_distance_sum[q], sizeof(T)));
            detail::labels_init();

            try {
                data_dots[q] = new thrust::device_vector<T>(n / n_gpu);
                centroid_dots[q] = new thrust::device_vector<T>(k);
                pairwise_distances[q] = new thrust::device_vector<T>(n / n_gpu * k);
                labels_copy[q] = new thrust::device_vector<int>(n / n_gpu * d);
                range[q] = new thrust::device_vector<int>(n / n_gpu);
                counts[q] = new thrust::device_vector<int>(k);
                indices[q] = new thrust::device_vector<int>(n / n_gpu);
            }
            catch (thrust::system_error &e) {
                // output an error message and exit
                std::stringstream ss;
                ss << "Unable to allocate memory for gpu: " << q << " n/n_gpu: " << n / n_gpu << " k: " << k << " d: "
                   << d << " error: " << e.what() << std::endl;
                return (-1);
            }
            catch (std::bad_alloc &e) {
                // output an error message and exit
                std::stringstream ss;
                ss << "Unable to allocate memory for gpu: " << q << " n/n_gpu: " << n / n_gpu << " k: " << k << " d: "
                   << d << " error: " << e.what() << std::endl;
                return (-1);
            }

            if (verbose) {
                fprintf(stderr, "Before Create and save range for initializing labels: gpu: %d\n", q);
                fflush(stderr);
            }
            //Create and save "range" for initializing labels
            thrust::copy(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(n / n_gpu),
                         (*range[q]).begin());

            if (verbose) {
                fprintf(stderr, "Before make_self_dots: gpu: %d\n", q);
                fflush(stderr);
            }

            detail::make_self_dots(n / n_gpu, d, *data[q], *data_dots[q]);
            if (init_from_labels) {
                if (verbose) {
                    fprintf(stderr, "Before find_centroids: gpu: %d\n", q);
                    fflush(stderr);
                }
                detail::find_centroids(q, n / n_gpu, d, k, *data[q], *labels[q], *centroids[q], *range[q], *indices[q],
                                       *counts[q]);
            }
        }

        if (verbose) {
            fprintf(stderr, "Before kmeans() Iterations\n");
            fflush(stderr);
        }
        int i = 0;
        bool done = false;
        for (; i < max_iterations; i++) {
            if (*flag) continue;
            //Average the centroids from each device (same as averaging centroid updates)
            if (n_gpu > 1) {
                for (int p = 0; p < k * d; p++) h_centroids[p] = 0.0;
                for (int q = 0; q < n_gpu; q++) {
                    safe_cuda(cudaSetDevice(dList[q]));
                    detail::memcpy(h_centroids_tmp, *centroids[q]);
                    detail::streamsync(dList[q]);
                    for (int p = 0; p < k * d; p++) h_centroids[p] += h_centroids_tmp[p];
                }
                for (int p = 0; p < k * d; p++) h_centroids[p] /= n_gpu;
                //Copy the averaged centroids to each device
                for (int q = 0; q < n_gpu; q++) {
                    safe_cuda(cudaSetDevice(dList[q]));
                    detail::memcpy(*centroids[q], h_centroids);
                }
            }
            for (int q = 0; q < n_gpu; q++) {
                safe_cuda(cudaSetDevice(dList[q]));
                if (verbose) {
                    fprintf(stderr, "q=%d\n", q);
                    fflush(stderr);
                }
                detail::calculate_distances(verbose, q, n / n_gpu, d, k,
                                            *data[q], *centroids[q], *data_dots[q],
                                            *centroid_dots[q], *pairwise_distances[q]);

                detail::relabel(n / n_gpu, k, *pairwise_distances[q], *labels[q], *distances[q], d_changes[q]);
                //TODO remove one memcpy
                detail::memcpy(*labels_copy[q], *labels[q]);
                detail::find_centroids(q, n / n_gpu, d, k, *data[q], *labels[q], *centroids[q], *range[q], *indices[q],
                                       *counts[q]);
                detail::memcpy(*labels[q], *labels_copy[q]);
                mycub::sum_reduce(*distances[q], d_distance_sum[q]);
            }

            // whether to perform per iteration check
            // TODO pass as parameter
            int docheck = 1;
            if (docheck) {
                double distance_sum = 0.0;
                int moved_points = 0.0;
                for (int q = 0; q < n_gpu; q++) {
                    safe_cuda(cudaSetDevice(dList[q])); //  unnecessary
                    safe_cuda(cudaMemcpyAsync(h_changes + q, d_changes[q], sizeof(int), cudaMemcpyDeviceToHost,
                                              cuda_stream[q]));
                    safe_cuda(cudaMemcpyAsync(h_distance_sum + q, d_distance_sum[q], sizeof(T), cudaMemcpyDeviceToHost,
                                              cuda_stream[q]));
                    detail::streamsync(dList[q]);
                    if (verbose >= 2) {
                        std::cout << "Device " << dList[q] << ":  Iteration " << i << " produced " << h_changes[q]
                                  << " changes and the total_distance is " << h_distance_sum[q] << std::endl;
                    }
                    distance_sum += h_distance_sum[q];
                    moved_points += h_changes[q];
                }
                if (i > 0) {
                    double fraction = (double) moved_points / n;
#define NUMSTEP 10
                    if (verbose > 1 && (i <= 1 || i % NUMSTEP == 0)) {
                        std::cout << "Iteration: " << i << ", moved points: " << moved_points << std::endl;
                    }
                    if (fraction < threshold) {
                        if (verbose) { std::cout << "Threshold triggered. Terminating early." << std::endl; }
                        done = true;
                    }
                }
            }
            if (*flag) {
                fprintf(stderr, "Signal caught. Terminated early.\n");
                fflush(stderr);
                *flag = 0; // set flag
                done = true;
            }
            if (done) break;
        }
        for (int q = 0; q < n_gpu; q++) {
            safe_cuda(cudaSetDevice(dList[q]));
            safe_cuda(cudaFree(d_changes[q]));
            safe_cuda(cudaFree(d_distance_sum[q]));
            detail::labels_close();
            delete(pairwise_distances[q]);
            delete(data_dots[q]);
            delete(centroid_dots[q]);
            delete(labels_copy[q]);
            delete(range[q]);
            delete(counts[q]);
            delete(indices[q]);
        }
        if (verbose) {
            fprintf(stderr, "Iterations: %d\n", i);
            fflush(stderr);
        }
        return 0;
    }
}
