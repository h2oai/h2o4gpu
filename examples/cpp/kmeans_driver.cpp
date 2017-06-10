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

  real_t a=0; //TODO: send real data over
  size_t rows = 260753;  // rows
  size_t cols = 298;  // cols

  void* res = 0;
  int n_gpu=2;
  std::vector<real_t> data(rows*cols);

  // works - makes random data
  //h2oaikmeans::H2OAIKMeans<real_t>(&a, k, rows, cols).Solve();

  // TODO: FIXME (one or the other)
  // h2oaikmeans::makePtr_dense<float>(n_gpu, rows, cols, 'r', k, &data[0], &res);
  // make_ptr_float_kmeans(n_gpu, rows, cols, 'r', k, &data[0], &res);

  fflush(stdout);
  fflush(stderr);

  return 0;
}