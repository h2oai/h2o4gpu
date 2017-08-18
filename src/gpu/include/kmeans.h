// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <atomic>
#include <signal.h>
#include <string>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "centroids.h"
#include "kmeans_labels.h"
#include "kmeans_general.h"

template<typename T>
void print_array(T& array, int m, int n) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      typename T::value_type value = array[i * n + j];
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
}

namespace kmeans {

  //! kmeans clusters data into k groups
  /*! 

    \param n Number of data points
    \param d Number of dimensions
    \param k Number of clusters
    \param data Data points, in row-major order. This vector must have
    size n * d, and since it's in row-major order, data point x occupies
    positions [x * d, (x + 1) * d) in the vector. The vector is passed
    by reference since it is shared with the caller and not copied.
    \param labels Cluster labels. This vector has size n.
    The vector is passed by reference since it is shared with the caller
    and not copied.
    \param centroids Centroid locations, in row-major order. This
    vector must have size k * d, and since it's in row-major order,
    centroid x occupies positions [x * d, (x + 1) * d) in the
    vector. The vector is passed by reference since it is shared
    with the caller and not copied.
    \param distances Distances from points to centroids. This vector has
    size n. It is passed by reference since it is shared with the caller
    and not copied.
    \param init_from_labels If true, the labels need to be initialized
    before calling kmeans. If false, the centroids need to be
    initialized before calling kmeans. Defaults to true, which means
    the labels must be initialized.
    \param threshold This controls early termination of the kmeans
    iterations. If the ratio of points being reassigned to a different
    centroid is less than the threshold, than the iterations are
    terminated. Defaults to 1e-3.
    \param max_iterations Maximum number of iterations to run
    \return The number of iterations actually performed.
   */

  template<typename T>
    int kmeans(
    	int verbose,
        volatile std::atomic_int * flag,
        int n, int d, int k,
        thrust::device_vector<T>** data,
        thrust::device_vector<int>** labels,
        thrust::device_vector<T>** centroids,
        thrust::device_vector<T>** distances,
        std::vector<int> dList,
        int n_gpu,
        int max_iterations,
        int init_from_labels=0,
        double threshold=1e-3) {
    
      thrust::device_vector<T> *data_dots[MAX_NGPUS];
      thrust::device_vector<T> *centroid_dots[MAX_NGPUS];
      thrust::device_vector<T> *pairwise_distances[MAX_NGPUS];
      thrust::device_vector<int> *labels_copy[MAX_NGPUS];
      thrust::device_vector<int> *range[MAX_NGPUS];
      thrust::device_vector<int> *indices[MAX_NGPUS];
      thrust::device_vector<int> *counts[MAX_NGPUS];


#if(DEBUGKMEANS)
      // debug
      thrust::host_vector<T> *h_data_dots[MAX_NGPUS];
      thrust::host_vector<T> *h_centroid_dots[MAX_NGPUS];
      thrust::host_vector<T> *h_pairwise_distances[MAX_NGPUS];
      thrust::host_vector<int> *h_labels_copy[MAX_NGPUS];
      thrust::host_vector<int> *h_range[MAX_NGPUS];
      thrust::host_vector<int> *h_indices[MAX_NGPUS];
      thrust::host_vector<int> *h_counts[MAX_NGPUS];
#endif

      
      thrust::host_vector<T> h_centroids( k * d );
      thrust::host_vector<T> h_centroids_tmp( k * d );
      int h_changes[MAX_NGPUS], *d_changes[MAX_NGPUS];
      T h_distance_sum[MAX_NGPUS], *d_distance_sum[MAX_NGPUS];


      

      for (int q = 0; q < n_gpu; q++) {

        if(verbose){
        	fprintf(stderr,"Before kmeans() Allocation: gpu: %d\n",q); fflush(stderr);
        }
        safe_cuda(cudaSetDevice(dList[q]));
        safe_cuda(cudaMalloc(&d_changes[q], sizeof(int)));
        safe_cuda(cudaMalloc(&d_distance_sum[q], sizeof(T)));
        detail::labels_init();
        
        try {
			data_dots[q] = new thrust::device_vector <T>(n/n_gpu);
			centroid_dots[q] = new thrust::device_vector<T>(k);
			pairwise_distances[q] = new thrust::device_vector<T>(n/n_gpu * k);
			labels_copy[q] = new thrust::device_vector<int>(n/n_gpu * d);
			range[q] = new thrust::device_vector<int>(n/n_gpu);
			counts[q] = new thrust::device_vector<int>(k);
			indices[q] = new thrust::device_vector<int>(n/n_gpu);
        }
        catch(thrust::system_error &e) {
            // output an error message and exit
        	std::stringstream ss;
            ss << "Unable to allocate memory for gpu: " << q << " n/n_gpu: " << n/n_gpu << " k: " << k << " d: " << d << " error: " << e.what() << std::endl;
            return(-1);
            // throw std::runtime_error(ss.str());
        }
        catch(std::bad_alloc &e) {
		    // output an error message and exit
        	std::stringstream ss;
			ss << "Unable to allocate memory for gpu: " << q << " n/n_gpu: " << n/n_gpu << " k: " << k << " d: " << d << " error: " << e.what() << std::endl;
			return(-1);
			//throw std::runtime_error(ss.str());
		}

#if(DEBUGKMEANS)
        // debug
        h_data_dots[q] = new thrust::host_vector <T>(n/n_gpu);
        h_centroid_dots[q] = new thrust::host_vector<T>(k*d);
        h_pairwise_distances[q] = new thrust::host_vector<T>(n/n_gpu * k);
        h_labels_copy[q] = new thrust::host_vector<int>(n/n_gpu * d);
        h_range[q] = new thrust::host_vector<int>(n/n_gpu);
        h_counts[q] = new thrust::host_vector<int>(k);
        h_indices[q] = new thrust::host_vector<int>(n/n_gpu);
#endif
        
        if(verbose){
        	fprintf(stderr,"Before Create and save range for initializing labels: gpu: %d\n",q); fflush(stderr);
        }
        //Create and save "range" for initializing labels
        thrust::copy(thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n/n_gpu), 
            (*range[q]).begin());

        if(verbose){
        	fprintf(stderr,"Before make_self_dots: gpu: %d\n",q); fflush(stderr);
        }

        detail::make_self_dots(n/n_gpu, d, *data[q], *data_dots[q]);
        if (init_from_labels) {
          if(verbose){
        	  fprintf(stderr,"Before find_centroids: gpu: %d\n",q); fflush(stderr);
          }
          detail::find_centroids(q, n/n_gpu, d, k, *data[q], *labels[q], *centroids[q], *range[q], *indices[q], *counts[q]);
        }
      }

      if(verbose){
    	 fprintf(stderr,"Before kmeans() Iterations\n");
    	 fflush(stderr);
      }
      int i=0;
      bool done=false;
      for(; i < max_iterations; i++) {
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
            detail::memcpy(*centroids[q],h_centroids);
          }
        }
        for (int q = 0; q < n_gpu; q++) {
        	safe_cuda(cudaSetDevice(dList[q]));
#if(DEBUGKMEANS)
          fprintf(stderr,"q=%d\n",q); fflush(stderr);
#endif
          detail::calculate_distances(verbose, q, n/n_gpu, d, k,
              *data[q], *centroids[q], *data_dots[q],
              *centroid_dots[q], *pairwise_distances[q]);

#if(DEBUGKMEANS)
          *h_pairwise_distances[0] = *pairwise_distances[0];
          size_t countpos=0;
          size_t countneg=0;
          for(int ll=0;ll<(*h_pairwise_distances[0]).size();ll++){
            T result=(*h_pairwise_distances[0])[ll];
            if(result>0) countpos++;
            if(result<0) countneg++;
          }
          fprintf(stderr,"countpos=%zu countneg=%zu\n",countpos,countneg); fflush(stderr);
#endif

          detail::relabel(n/n_gpu, k, *pairwise_distances[q], *labels[q], *distances[q], d_changes[q]);
          //TODO remove one memcpy
          detail::memcpy(*labels_copy[q], *labels[q]);
          detail::find_centroids(q, n/n_gpu, d, k, *data[q], *labels[q], *centroids[q], *range[q], *indices[q], *counts[q]);
          detail::memcpy(*labels[q], *labels_copy[q]);
          //T d_distance_sum[q] = thrust::reduce(distances[q].begin(), distances[q].end())
          mycub::sum_reduce(*distances[q], d_distance_sum[q]);
        }

        // whether to perform per iteration check
        int docheck=1;
        if(docheck){
          double distance_sum = 0.0;
          int moved_points = 0.0;
          for (int q = 0; q < n_gpu; q++) {
        	safe_cuda(cudaSetDevice(dList[q])); //  unnecessary
        	safe_cuda(cudaMemcpyAsync(h_changes+q, d_changes[q], sizeof(int), cudaMemcpyDeviceToHost, cuda_stream[q]));
        	safe_cuda(cudaMemcpyAsync(h_distance_sum+q, d_distance_sum[q], sizeof(T), cudaMemcpyDeviceToHost, cuda_stream[q]));
            detail::streamsync(dList[q]);
            if(verbose>=2){
            	std::cout << "Device " << dList[q] << ":  Iteration " << i << " produced " << h_changes[q]
                << " changes and the total_distance is " << h_distance_sum[q] << std::endl;
            }
            distance_sum += h_distance_sum[q];
            moved_points += h_changes[q];
          }
          if (i > 0) {
            double fraction = (double)moved_points / n;
#define NUMSTEP 10
            if(verbose>1 && (i<=1 || i%NUMSTEP==0)){
              std::cout << "Iteration: " << i << ", moved points: " << moved_points << std::endl;
            }
            if (fraction < threshold) {
              if(verbose){ std::cout << "Threshold triggered. Terminating early." << std::endl; }
                done=true;
            }
          }
        }
        if (*flag) {
          fprintf(stderr, "Signal caught. Terminated early.\n"); fflush(stderr);
          *flag = 0; // set flag
            done=true;
        }
          if(done) break;
      }
      for (int q = 0; q < n_gpu; q++) {
    	safe_cuda(cudaSetDevice(dList[q]));
    	safe_cuda(cudaFree(d_changes[q]));
        detail::labels_close();
        delete(pairwise_distances[q]);
        delete(data_dots[q]);
        delete(centroid_dots[q]);
        delete(labels_copy[q]);
        delete(range[q]);
        delete(counts[q]);
        delete(indices[q]);
      }


      if(verbose){
    	  fprintf(stderr,"Iterations: %d\n",i);
    	  fflush(stderr);
      }
      return 0;
    }
}
