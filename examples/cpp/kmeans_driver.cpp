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
#include <thrust/device_vector.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include <random>
#include "h2oaikmeans.h"

typedef float real_t;
//typedef double real_t;

int main(int argc, char **argv) {
  using namespace std;

  int max_iterations = 10000;
  int k = 100;  // clusters
  double threshold = 1e-3;  // relative improvement
  int n_gpu;
  cudaGetDeviceCount(&n_gpu);
  std::cout << n_gpu << " gpus." << std::endl;
  size_t rows = n_gpu*100000;  // rows
  size_t cols = 100;  // cols

  void* res = 0;

#if 0
  // creates random data inside
  real_t a = 0;
  h2oaikmeans::H2OAIKMeans<real_t>(&a, k, rows, cols).Solve();
#else
  //user-given data
  std::vector<real_t> data(rows*cols);
  std::vector<int> labels(rows*cols);
  for (unsigned int i=0;i<rows;i++) {
    for (unsigned int j = 0; j < cols; j++) {
      data[i * cols + j] = static_cast<real_t>(drand48());
      labels[i * cols + j] = static_cast<int>(rand() % k);
    }
  }
  h2oaikmeans::makePtr_dense<real_t>(n_gpu, rows, cols, 'r', k, max_iterations, threshold, &data[0], &labels[0], &res);

  // report something about centroids that site in res as k*cols data block
#endif

  fflush(stdout);
  fflush(stderr);

  return 0;
}
