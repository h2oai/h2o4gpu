#include <cstdio>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include "examples.h"
#include "reader.h"
#include "timer.h"

typedef float real_t;

int main(int argc, char **argv) {
  using namespace std;
  double t;

  size_t rows=0, cols=0;


  if (argc!=10) {
    printf("usage: %s %s", argv[0], " <file.txt> <DoSharedA> <nThreads> <nGPUs> <nLambdas> <nAlphas> <intercept?1:0> <standardize?1:0> <validFraction>\n");
    exit(-1);
  }

  int ai=0;
  char *filename=argv[++ai];
  int sharedA=atoi(argv[++ai]);
  int nThreads=atoi(argv[++ai]);
  int nGPUs=atoi(argv[++ai]);
  int nLambdas=atoi(argv[++ai]);
  int nAlphas=atoi(argv[++ai]);
  int intercept=atoi(argv[++ai]);
  int standardize=atoi(argv[++ai]);
  double validationFraction=atof(argv[++ai]);

  std::vector<real_t> A;
  std::vector<real_t> b;
  cout << "START FILL DATA\n" << endl;
  double t0 = timer<double>();
  fillData(rows,cols,filename, A, b);
  double t1 = timer<double>();
  cout << "END FILL DATA. Took " << t1 - t0 << " secs" << endl;

  

  printf("\nElastic Net: rows=%d cols=%d sharedA=%d nThreads=%d nGPUs=%d nlambdas=%d nalphas=%d intercept=%d standardize=%d validFraction=%g\n",rows,cols,sharedA, nThreads, nGPUs,nLambdas,nAlphas,intercept,standardize,validationFraction);
  t = ElasticNet<real_t>(A, b, sharedA, nThreads, nGPUs, nLambdas, nAlphas, intercept, standardize, validationFraction);
  printf("\nElastic Net: rows=%d cols=%d sharedA=%d nThreads=%d nGPUs=%d nlambdas=%d nalphas=%d intercept=%d standardize=%d validFraction=%g time=%e secs\n",rows,cols,sharedA, nThreads, nGPUs,nLambdas,nAlphas,intercept,standardize,validationFraction,t);

  fflush(stdout);
  fflush(stderr);
  


  return 0;
}

