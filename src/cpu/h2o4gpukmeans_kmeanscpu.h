/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#define MAX_NCPUS 1

namespace kmeans {

template<typename T>
  void self_dot(std::vector<T> array_in, int n, int dim,
                std::vector<T>& dots) {
    for (int pt = 0; pt<n; pt++) {
      T sum = 0.0;
      for (int i=0; i<dim; i++) {
        sum += array_in[pt*dim+i]*array_in[pt*dim+i];
      }
      dots[pt] = sum;
    }
  }



template<typename T>
  void find_centroids(std::vector<T> array_in, int n, int dim,
                      std::vector<int> labels_in,
                      std::vector<T>& centroids, int k) {
    std::vector<int> members(k); //Number of points in each cluster

    std::fill(members.begin(), members.end(), 0);
    if(0){ // see gpu code for comments
      std::fill(centroids.begin(), centroids.end(), 0.0);
    }
    
    //Add all vectors in the cluster
    for(int pt=0; pt<n; pt++) {
      int this_cluster = labels_in[pt];
      members[this_cluster]++;
      for (int i=0; i<dim; i++) centroids[this_cluster*dim+i] +=
                                  array_in[pt*dim+i];
    }
    //Divide by the number of points in the cluster
    for(int cluster=0; cluster < k; cluster++) {
      if (dim < 6) std::cout << cluster << "(" << members[cluster] << " members):  ";
      for (int i=0; i<dim; i++) {
        centroids[cluster*dim+i] /= members[cluster];
        if (dim < 6) std::cout << centroids[cluster*dim+i] << "  ";
      }
      if (dim < 6) std::cout << std::endl;
    }
  }

  void compute_distances(std::vector<double> data_in,
                         std::vector<double> data_dots_in,
                         int n, int dim, std::vector<double> centroids_in,
                         std::vector<double> centroid_dots, int k,
                         std::vector<double>& pairwise_distances) {
    self_dot(centroids_in, k, dim, centroid_dots);
    for (int nn=0; nn<n; nn++)
      for (int c=0; c<k; c++) {
                 pairwise_distances[nn*k+c] = data_dots_in[nn] +
                   centroid_dots[c];
      }
    double alpha = -2.0;
    double beta = 1.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, k,
                dim, alpha, &data_in[0], dim, &centroids_in[0], dim,
                beta, &pairwise_distances[0], k);
  }

  void compute_distances(std::vector<float> data_in,
                         std::vector<float> data_dots_in,
                         int n, int dim, std::vector<float> centroids_in,
                         std::vector<float> centroid_dots, int k,
                         std::vector<float>& pairwise_distances) {
    self_dot(centroids_in, k, dim, centroid_dots);
    for (int nn=0; nn<n; nn++)
      for (int c=0; c<k; c++) {
                 pairwise_distances[nn*k+c] = data_dots_in[nn] +
                   centroid_dots[c];
      }
    float alpha = -2.0;
    float beta = 1.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, k,
                dim, alpha, &data_in[0], dim, &centroids_in[0], dim,
                beta, &pairwise_distances[0], k);
  }

  template<typename T>
  int relabel(std::vector<T> data_in, int n,
              std::vector<T> pairwise_distances_in,
              int k, std::vector<int>& labels) {
    int changes = 0;
    for (int nn=0; nn<n; nn++) {
      T min = pairwise_distances_in[nn*k];
      int idx = 0;
      for (int cc=1; cc<k; cc++) {
        T this_dist = pairwise_distances_in[nn*k+cc];
        if (this_dist < min) {
          idx=cc;
          min=this_dist;
        }
      }
      if (labels[nn] != idx) {
        changes ++;
        labels[nn] = idx;
      }
    }
    return changes;
  }

  template<typename T>
  int kmeans(int verbose,
             volatile std::atomic_int * flag,
             int n, int d, int k,
             std::vector<T> &data,
             std::vector<int> &labels,
             std::vector<T> &centroids,
             int max_iterations,
             int init_from_data=0,
             double threshold=1e-3) {
      

    // TRANSLATE to CPU CODE
    std::vector<T> data_dots(n);
    std::vector<T> centroid_dots(k);
    std::vector<T> pairwise_distances(k * n);
    std::vector<int> labels_copy(n);
    
    // rest is original CPU code
    self_dot(data, n, d, data_dots);

    //Let the first k points be the centroids of the clusters
    //    memcpy(&centroids[0], &data[0], sizeof(T)*k*d);

    int i;
    for(i=0; i<max_iterations; i++) {
      compute_distances(data, data_dots, n, d, centroids, centroid_dots,
                        k, pairwise_distances);

      int moved_points = relabel(data, n, pairwise_distances, k, labels);
      std::cout <<std::endl << "*** Iteration " << i << " ***" << std::endl;
      std::cout << moved_points << " points moved between clusters." << std::endl;

      if (i > 0) {
        double fraction = (double)moved_points / n;
#define NUMSTEP 10
        if(VERBOSE || VERBOSE==0 && i%NUMSTEP==0){
          std::cout << "Iteration: " << i << ", moved points: " << moved_points << std::endl;
        }
        if (fraction < threshold || 0 == moved_points) {
          std::cout << "Threshold triggered. Terminating early." << std::endl;
          return i + 1;
        }
      }
      if (*flag) {
        fprintf(stderr, "Signal caught. Terminated early.\n"); fflush(stderr);
        *flag = 0; // set flag
      }
    
      
      find_centroids(data, n, d, labels, centroids, k);
    }

    return i;

  }


}
