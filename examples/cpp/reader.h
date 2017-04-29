#ifndef __READER_H__
#define __READER_H__

#include <fstream>
#include <random>
#include <cstring>
#include <iostream>
#include <sstream>

#include <iterator>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

template<typename T>
void splitData(const std::vector<T>& A, const std::vector<T>& b, const std::vector<T>& w,
               std::vector<T>& trainX, std::vector<T>& trainY, std::vector<T>& trainW,
               std::vector<T>& validX, std::vector<T>& validY, std::vector<T>& validW,
               double validFraction, int intercept) {
  using namespace std;

  cout << "START TRAIN/VALID SPLIT" << endl;
// Split A/b into train/valid, via head/tail
  size_t m = b.size();
  size_t mValid = static_cast<size_t>(validFraction * static_cast<double>(m));
  size_t mTrain = m - mValid;
  size_t n = A.size() / m;

// If intercept == 1, add one extra column at the end, all constant 1s
  n += intercept;

  trainX.resize(mTrain * n); // TODO FIXME: Should just point trainX to part of A to save memory
  trainY.resize(mTrain);
  trainW.resize(mTrain);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < mTrain; ++i) { //rows
    trainY[i] = b[i];
    trainW[i] = w[i];
//      cout << "y[" << i << "] = " << trainY[i] << endl;
    for (int j = 0; j < n - intercept; ++j) { //cols
      trainX[i * n + j] = A[i * (n-intercept) + j];
//        cout << "X[" << i << ", " << j << "] = " << trainX[i*n+j] << endl;
    }
    if (intercept) {
      trainX[i * n + n - 1] = 1;
    }
  }
  if (mValid > 0) {
    validX.resize(mValid * n);
    validY.resize(mValid);
    validW.resize(mValid);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mValid; ++i) { //rows
      validY[i] = b[mTrain + i];
      validW[i] = w[mTrain + i];
      for (int j = 0; j < n - intercept; ++j) { //cols
        validX[i * n + j] = A[(mTrain + i) * (n-intercept) + j];
      }
      if (intercept) {
        validX[i * n + n - 1] = 1;
      }
    }
  }
  cout << "END TRAIN/VALID SPLIT" << endl;
  fflush(stdout);
}



template<typename T>
int fillData(size_t m, size_t n, // only used if name.empty()
             const std::string &file,
             std::vector<T>& A, std::vector<T>& b, std::vector<T>& w) {
  std::default_random_engine generator;
  std::uniform_real_distribution <T> u_dist(static_cast<T>(0),
                                            static_cast<T>(1));
  std::normal_distribution <T> n_dist(static_cast<T>(0),
                                      static_cast<T>(1));

// READ-IN DATA
  if (!file.empty()) {

    std::ifstream ifs(file);
    std::string line;
    size_t rows=0;
    size_t cols = 0;
    while (std::getline(ifs, line)) {
      if (rows==0) {
        std::string buf;
        std::stringstream ss(line);
        while (ss >> buf) cols++;
      }
      //std::cout << line << std::endl;
      rows++;
    }
    cols--; //don't count target column

    printf("rows: %d\n", rows); fflush(stdout);
    printf("cols (w/o response): %d\n", cols); fflush(stdout);
    ifs.close();

    size_t m = rows;
    size_t n = cols;
    A.resize(n*m);
    b.resize(m);
    w.resize(m);
#ifdef _OPENMP
#pragma omp parallel
{
#endif
    std::ifstream ifs(file);
    if (!ifs) {
      fprintf(stderr, "Cannot read file.\n");
      exit(1);
    } else {
      // Expect space-separated file with response in last column, no header
      int maxlen = (n + 1) * 30;
      char line[maxlen];
      const char sep[] = " ,";

      size_t i=0;
#ifdef _OPENMP
      int id = omp_get_thread_num();
      int nth = omp_get_num_threads();
      size_t len = m / nth;
      size_t from = id*len;
      size_t to = (id+1)*len;
      if (to > m) to = m;
      if (id==nth-1) to = m;
      //#pragma omp critical
      //      {
      //        std::cout << "thread " << id << " reads lines " << from << "..." << to << std::endl;
      //      }
#else
      size_t from = 0;
      size_t to = m;
#endif
      for (; i < from; ++i) ifs.getline(line, maxlen);
      for (; i < to; ++i) {
        ifs.getline(line, maxlen);
        char *pch;
        char *savePtr;
        pch = strtok_r(line, sep, &savePtr);
        int j = 0;
        T val = static_cast<T>(atof(pch));
        A[i * n + j] = val;
        j++;
        while (pch != NULL) {
          pch = strtok_r(NULL, sep, &savePtr);
          if (pch != NULL) {
            val = static_cast<T>(atof(pch));
            if (j < n) {
              A[i * n + j] = val;
            } else {
              b[i] = val;
              w[i] = 1.0; // just constant weight
            }
            j++;
          }
        }
      }
    }
#ifdef _OPENMP
  }
#endif

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m * n; ++i) {
      if (!std::isfinite(A[i])) fprintf(stderr, "NF: A[%d]=%g\n", i, A[i]);
    }
    for (unsigned int i = 0; i < m; ++i) {
      if (!std::isfinite(b[i])) fprintf(stderr, "b[%d]=%g\n", i, b[i]);
      if (!std::isfinite(w[i])) fprintf(stderr, "w[%d]=%g\n", i, w[i]);
    }
  } else {
    // GENERATE DATA
    for (unsigned int i = 0; i < m * n; ++i)
      A[i] = n_dist(generator);

    std::vector <T> x_true(n);
    for (unsigned int i = 0; i < n; ++i)
      x_true[i] = u_dist(generator) < static_cast<T>(0.8)
                  ? static_cast<T>(0) : n_dist(generator) / static_cast<T>(std::sqrt(n));

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m; ++i) // rows
      for (unsigned int j = 0; j < n; ++j) // columns
        b[i] += A[i * n + j] * x_true[j]; // C(0-indexed) row-major order

    for (unsigned int i = 0; i < m; ++i)
      w[i] = 1.0; // constant weight
    for (unsigned int i = 0; i < m; ++i)
      b[i] += static_cast<T>(0.5) * n_dist(generator);

  }
  return(0);
}
#endif
