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
#include "h2oaikmeans.h"

typedef float real_t;
//typedef double real_t;

int main(int argc, char **argv) {
  using namespace std;

  int max_iterations = 10000;
  int k = 100;  // clusters
  double thresh = 1e-3;  // relative improvement

  int n_gpu;
  cudaGetDeviceCount(&n_gpu);
  std::cout << n_gpu << " gpus." << std::endl;

  real_t a=0; //TODO: send real data over
  size_t rows = 260753;  // rows
  size_t cols = 298;  // cols

  h2oaikmeans::H2OAIKMeans<real_t>(&a, k, rows, cols).Solve();

  fflush(stdout);
  fflush(stderr);

  return 0;
}