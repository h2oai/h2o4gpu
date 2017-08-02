#include <cstdio>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include "examples.h"
#include "reader.h"
#include "timer.h"

#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include <iostream>
//#include "cuda.h"
#include <cstdlib>
#include <random>
#include "h2ogpumlkmeans.h"

typedef float real_t;
//typedef double real_t;

int main(int argc, char **argv) {
  using namespace std;

  int max_iterations = 10000;
  int k = 100;  // clusters
  double threshold = 1e-3;  // relative improvement
  int n_gpu=1;
  //  cudaGetDeviceCount(&n_gpu);
  std::cout << n_gpu << " gpus." << std::endl;
  size_t rows = n_gpu*100000;  // rows
  size_t cols = 100;  // cols

  int gpu_id = 0;

  void* preds;

#if 0
  // creates random data inside
  real_t a = 0;
  h2ogpumlkmeans::H2OGPUMLKMeans<real_t>(&a, k, rows, cols).Solve();
#else
  //user-given data
  std::vector<real_t> data(rows*cols);
  for (unsigned int i=0;i<rows;i++) {
    for (unsigned int j = 0; j < cols; j++) {
      data[i * cols + j] = static_cast<real_t>(drand48());
    }
  }
  std::vector<int> labels(rows);
  for (unsigned int i=0;i<rows;i++) {
    labels[i] = static_cast<int>(rand() % k);
  }
  int verbose=1;
  const char ord='r';
  int init_from_labels=1;//true as set above, otherwise internally will set initial centroids "smartly"
  int init_labels=1; // randomly select from input
  int init_data=2; // randomly (without replacement) select from input
  int seed=12345;
  int dopredict = 0;

  std::vector<real_t> centroids(1);
  centroids[0] = static_cast<real_t>(drand48());

  h2ogpumlkmeans::makePtr_dense<real_t>(dopredict, verbose, seed, gpu_id, n_gpu, rows, cols, ord, k, max_iterations, init_from_labels, init_labels, init_data, threshold, &data[0], &labels[0], &centroids[0], &preds);

  // report something about centroids that site in res as k*cols data block
#endif

  fflush(stdout);
  fflush(stderr);

  return 0;
}
