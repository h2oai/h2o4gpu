#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include <vector>

template <typename T>
void row2csr(const std::vector<T> &A, size_t m, size_t n,
             std::vector<T> *val, std::vector<int> *row_ptr,
             std::vector<int> *col_ind) {
  size_t nnz = 0;
  for (size_t i = 0; i < m; ++i) {
    row_ptr->push_back(static_cast<int>(nnz));
    for (size_t j = 0; j < n; ++j) {
      if (A[static_cast<size_t>(i * n + j)] != static_cast<T>(0)) {
        val->push_back(A[static_cast<size_t>(i * n + j)]);
        col_ind->push_back(static_cast<int>(j));
        nnz++;
      }
    }
  }
  row_ptr->push_back(static_cast<int>(nnz));
}

template <typename T>
void col2csc(const std::vector<T> &A, size_t m, size_t n,
             std::vector<T> *val, std::vector<int> *col_ptr,
             std::vector<int> *row_ind) {
  size_t nnz = 0;
  for (size_t j = 0; j < n; ++j) {
    col_ptr->push_back(static_cast<int>(nnz));
    for (size_t i = 0; i < m; ++i) {
      if (A[static_cast<size_t>(i + j * m)] != static_cast<T>(0)) {
        val->push_back(A[static_cast<size_t>(i + j * m)]);
        row_ind->push_back(static_cast<int>(i));
        nnz++;
      }
    }
  }
  col_ptr->push_back(static_cast<int>(nnz));
}

#endif  // TEST_UTIL_H_

