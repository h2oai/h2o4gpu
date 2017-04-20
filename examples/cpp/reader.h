#ifndef __READER_H__
#define __READER_H__

#include <fstream>
#include <random>
#include <cstring>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

template<typename T>
int fillData(int generate, std::string name, size_t rows, size_t cols, T* A, T* b) {
  std::default_random_engine generator;
  std::uniform_real_distribution <T> u_dist(static_cast<T>(0),
                                            static_cast<T>(1));
  std::normal_distribution <T> n_dist(static_cast<T>(0),
                                      static_cast<T>(1));

  size_t m = rows;
  size_t n = cols;

// READ-IN DATA
  if (generate == 0) {
    size_t R = m; // rows
    size_t C = n + 1; // columns
#ifdef _OPENMP
#pragma omp parallel
{
#endif
    std::ifstream ifs(name);
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
      size_t len = R / nth;
      size_t from = id*len;
      size_t to = (id+1)*len;
      if (to > R) to = R;
      if (id==nth-1) to = R;
#pragma omp critical
      {
        std::cout << "thread " << id << " reads lines " << from << "..." << to << std::endl;
      }
#else
      size_t from = 0;
      size_t to = R;
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
      b[i] += static_cast<T>(0.5) * n_dist(generator);

  }
  return(0);
}
#endif
